import time
from typing import List, Optional, Tuple, Union
from layers.configs import RetNetConfig
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.utils import logging
from layers.Xpos import XPOS
from layers.snippets import get_gpu_memory_usage
import gc

logger = logging.get_logger(__name__)


# helper functions
def split_chunks(*tensors, size, dim=0):
    return [torch.split(x, size, dim=dim) for x in tensors]


def split_heads(tensors, bsz, seqlen, num_heads):
    assert isinstance(tensors, (tuple, list))
    return [x.view(bsz, seqlen, num_heads, -1).transpose(1, 2) for x in tensors]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MultiScaleRetention(nn.Module):
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model,
                             config.qk_dim * 2 + config.v_dim,
                             bias=config.use_bias_in_msr)
        self.silu = nn.SiLU()
        self.gated = nn.Linear(config.d_model, config.v_dim, bias=False)
        self.proj = nn.Linear(config.v_dim, config.d_model, bias=config.use_bias_in_msr_out)
        self.gn = nn.GroupNorm(num_groups=config.num_heads, num_channels=config.v_dim, affine=False)
        self.xpos = XPOS(config.qk_dim)

        # initialize gamma, found
        if config.use_default_gamma:
            gamma = 1 - 2**(-5 - torch.arange(0, config.num_heads, dtype=torch.float))
        else:
            # you should update the gamma value based on your dataset, if the time span is large, make gamma close to 1
            # s = torch.log(torch.tensor(1 / 32))
            s = torch.log(torch.tensor(1 / 64))
            # s = torch.log(torch.tensor(1 / 128))
            e = torch.log(torch.tensor(1 / 512))
            gamma = 1 - torch.exp(torch.linspace(s, e, config.num_heads))  # [h,]
        self.decay = nn.Parameter(gamma, requires_grad=False) # tensor([0.9688, 0.9876, 0.9951, 0.9980])

    def get_parallel_decay_mask(self, length, t, retention_mask=None):
        t = t.to(self.decay.device)
        # Compute the pairwise time differences for each sequence in the batch
        time_diff = t.unsqueeze(2) - t.unsqueeze(1)  # [batch_size, seq_len, seq_len]
        # Expand the decay parameter to match the batch size and sequence length
        decay_mask = self.decay.view(1, -1, 1, 1) ** time_diff.unsqueeze(1)  # [batch_size, num_heads, seq_len, seq_len]
        # Apply lower triangular mask to ensure causality
        decay_mask = torch.tril(decay_mask, diagonal=0)  # [batch_size, num_heads, seq_len, seq_len]
        if retention_mask is not None:
            # Expand the retention mask to match the decay mask dimensions
            retention_mask = retention_mask.float().view(-1, 1, 1, length)
            decay_mask = decay_mask * retention_mask
        else:
            decay_mask = decay_mask

        return decay_mask

    # def get_recurrent_decay(self):
    #     decay = self.decay.view(1, self.config.num_heads, 1, 1)
    #     return decay

    # def get_chunkwise_decay(self, chunk_size, retention_mask=None):
    #     # within chunk decay
    #     decay_mask = self.get_parallel_decay_mask(chunk_size, retention_mask=retention_mask)
    #     # decay of the chunk
    #     chunk_decay = self.decay.view(1, self.config.num_heads, 1, 1)**chunk_size
    #     # cross-chunk decay
    #     exponent = torch.arange(chunk_size, dtype=torch.float,
    #                             device=decay_mask.device).unsqueeze(0) + 1
    #     inner_decay = (self.decay.unsqueeze(-1)**exponent).view(1, self.config.num_heads, chunk_size, 1)
    #     return decay_mask, chunk_decay, inner_decay

    def parallel_retention(self, q, k, v, decay_mask):
        """
        q,  # bsz * num_heads * len * qk_dim
        k,  # bsz * num_heads * len * qk_dim
        v,  # bsz * num_heads * len * v_dim
        decay_mask,  # (1 or bsz) * num_heads * len * len
        """
        # [b, h, t, t]
        retention = q @ k.transpose(-1, -2) * k.size(-1)**-0.5  # (scaled dot-product) found
        retention = retention * decay_mask
        output = retention @ v #


        # kv cache
        current_kv = k.unsqueeze(-1) * v.unsqueeze(-2)
        # [bsz, num_heads, qk_dim, v_dim]
        intra_decay = decay_mask[:, :, -1, :, None, None]
        current_kv = (current_kv * intra_decay).sum(2)
        return output, current_kv, retention

    # def recurrent_retention(self, q, k, v, past_key_value=None, decay=None, retention_mask=None):
    #     """
    #     q, k, v, # bsz * num_heads * 1 * qkv_dim
    #     past_key_value, # bsz * num_heads * qk_dim * v_dim
    #     decay # num_heads * 1 * 1
    #     retention_mask # bsz * 1
    #     """
    #     past_key_value = past_key_value if past_key_value is not None else 0
    #     decay = decay if decay is not None else 0
    #     retention_mask = retention_mask.view(-1, 1, 1, 1) if retention_mask is not None else 1
    #     # (b, h, d_k, d_v)
    #     current_kv = decay * past_key_value + retention_mask * (k.transpose(-1, -2) @ v)
    #     output = q @ current_kv * k.size(-1)**-0.5  # (b, h, 1, d_v)
    #     return output, current_kv

    # def chunkwise_retention(self,
    #                         q,
    #                         k,
    #                         v,
    #                         decay_mask,
    #                         past_key_value=None,
    #                         chunk_decay=None,
    #                         inner_decay=None
    #                         ):
    #     """
    #     q, k, v,  # bsz * num_heads * chunk_size * qkv_dim
    #     past_key_value,  # bsz * num_heads * qk_dim * v_dim
    #     decay_mask,  # 1 * num_heads * chunk_size * chunk_size
    #     chunk_decay,  # 1 * num_heads * 1 * 1
    #     inner_decay,  # 1 * num_heads * chunk_size * 1
    #     """
    #     # [bsz, num_heads, chunk_size, chunk_size]
    #     retention = q @ k.transpose(-1, -2) * k.size(-1)**-0.5
    #     retention = retention * decay_mask
    #     inner_retention = retention @ v  # [bsz, num_heads, chunk_size, v_dim]
    #
    #     if past_key_value is None:
    #         cross_retention = 0
    #         past_chunk = 0
    #     else:
    #         cross_retention = (q @ past_key_value) * inner_decay * k.size(-1)**-0.5
    #         past_chunk = chunk_decay * past_key_value
    #
    #     # [bsz, num_heads, chunk_size, v_dim]
    #     retention = inner_retention + cross_retention
    #     # [bsz, num_heads, chunk_size, qk_dim, v_dim]
    #     current_kv = k.unsqueeze(-1) * v.unsqueeze(-2)
    #     # [bsz, num_heads, qk_dim, v_dim]
    #     intra_decay = decay_mask[:, :, -1, :, None, None] # NOTE: intra_decay is omitted in the paper; but this detail is important
    #     current_kv = (current_kv * intra_decay).sum(2) # it is same to 8.9 paper # very confused about intra_decay
    #     current_kv = past_chunk + current_kv
        # return retention, current_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        retention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        forward_impl: str = 'parallel',
        sequence_offset: Optional[int] = 0,
        chunk_size: Optional[int] = None,
        output_retentions: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        B, T, H = hidden_states.size()

        q, k, v = self.qkv(hidden_states).split(
            [self.config.qk_dim, self.config.qk_dim, self.config.v_dim], dim=-1)

        q, k = self.xpos.rotate_queries_and_keys(q, k, offset=sequence_offset) # xpos for q and k
        q, k, v = split_heads((q, k, v), B, T, self.config.num_heads)

        # retention
        if forward_impl == 'parallel':
            decay_mask = self.get_parallel_decay_mask(T, t, retention_mask=retention_mask)
            retention_out, curr_kv, retention_weights = self.parallel_retention(q, k, v, decay_mask)
            torch.cuda.empty_cache()
            gc.collect()
        # elif forward_impl == 'recurrent':
        #     decay = self.get_recurrent_decay()
        #     retention_out, curr_kv = self.recurrent_retention(q, k, v,
        #                                                       past_key_value=past_key_value,
        #                                                       decay=decay,
        #                                                       retention_mask=retention_mask)
        # elif forward_impl == 'chunkwise':
        #     assert chunk_size is not None
        #     q_chunks, k_chunks, v_chunks = split_chunks(q, k, v, size=chunk_size, dim=2)
        #     if retention_mask is not None:
        #         retention_mask_chunks = split_chunks(retention_mask, size=chunk_size, dim=1)[0]
        #     ret_chunks = []
        #     for i, (_q, _k, _v) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
        #         csz = _q.size(2)
        #         ret_mask = retention_mask_chunks[i] if retention_mask is not None else None
        #         decay_mask, chunk_decay, inner_decay = self.get_chunkwise_decay(csz, retention_mask=ret_mask)
        #         out_chunk, past_key_value = self.chunkwise_retention(_q, _k, _v,
        #                                                              decay_mask,
        #                                                              past_key_value=past_key_value,
        #                                                              chunk_decay=chunk_decay,
        #                                                              inner_decay=inner_decay)
                # # Memory usage after processing the current chunk
                # gpu_mem_usage = get_gpu_memory_usage()
                # print("GPU memory usage after %d th chunk:" % i)
                # print("Total GPU Memory: {} MiB".format(gpu_mem_usage['total']))
                # print("Used GPU Memory: {} MiB".format(gpu_mem_usage['used']))
                # print("Free GPU Memory: {} MiB".format(gpu_mem_usage['free']))
                # ret_chunks.append(out_chunk)
            # [bsz, num_heads, seqlen, v_dim]
            # retention_out = torch.cat(ret_chunks, dim=2)
            # curr_kv = past_key_value
        else:
            raise ValueError(f'forward_impl {forward_impl} not supported.')
        # concat heads
        retention_out = retention_out.transpose(1, 2).contiguous().view(B, T, self.config.v_dim)
        # group norm (merge batch, length dimension -> group norm -> split back)
        normed = self.gn(retention_out.view(B * T, self.config.v_dim))
        normed = normed.view(B, T, self.config.v_dim)
        # out gate & proj
        out = self.silu(self.gated(hidden_states)) * normed

        outputs = (self.proj(out), curr_kv) # project v_dim to d_model
        if output_retentions:
            outputs += (retention_weights,) if forward_impl == 'parallel' else (None,)
        return outputs


class RetNetBlock(nn.Module):
    '''
    A decoder layer of RetNet consisting of a multi-scale retention (MSR) and MLP layer
    '''
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config

        # multi-scale retention, output shape is [bsz, seq_len, d_model]
        self.msr = MultiScaleRetention(config)

        # 2-layer MLP, output shape is [bsz, seq_len, d_model]
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_proj_size, bias=config.use_bias_in_mlp),
            nn.GELU(),
            nn.Linear(config.ffn_proj_size, config.d_model, bias=config.use_bias_in_mlp),
        )

        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        retention_mask: Optional[torch.Tensor] = None,
        forward_impl: str = 'parallel',
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        sequence_offset: Optional[int] = 0,
        # chunk_size: Optional[int] = None,
        output_retentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        msr_outs = self.msr(self.ln1(hidden_states),
                            t=t,
                            retention_mask=retention_mask,
                            past_key_value=past_key_value,
                            forward_impl=forward_impl,
                            sequence_offset=sequence_offset,
                            # chunk_size=chunk_size,
                            output_retentions=output_retentions)
        # Original MSR layer with residual connection
        msr = msr_outs[0]
        curr_kv = msr_outs[1]
        x = hidden_states + msr

        # MLP layer
        y = x + self.ffn(self.ln2(x))

        outputs = (y, curr_kv)

        if output_retentions:
            outputs += (msr_outs[2],)
        return outputs




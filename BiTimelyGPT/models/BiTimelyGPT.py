import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Conv_layers import *
from layers.BiTimelyGPT_layers import RetentionBlock
from layers.heads import *
from layers.Embed import PatchEmbedding, ValueEmbedding
from typing import List, Optional, Tuple, Union
from layers.snippets import get_gpu_memory_usage, SigmoidRange


class BiTimelyGPT(nn.Module):
    '''
    Temporal Generative Pre-training leverages recurrent-form transformer architecture for multi-variate time series
    '''
    def __init__(self, configs, head_type='pretrain'):
        super(BiTimelyGPT, self).__init__()

        # load parameters
        self.n_layers = configs.num_layers
        self.n_heads = configs.num_heads
        self.d_model = configs.d_model
        self.qk_dim = configs.qk_dim
        self.v_dim = configs.v_dim if configs.v_dim else self.qk_dim
        self.dropout = configs.dropout

        self.n_output = configs.n_output
        # self.seq_len = configs.seq_len
        # self.label_len = configs.label_len
        # self.pred_len = configs.pred_len

        # the start token for shifted right
        self.sos = torch.nn.Parameter(torch.zeros(self.d_model))
        nn.init.normal_(self.sos)
        # the end token for shifted right
        self.eos = torch.nn.Parameter(torch.zeros(self.d_model))
        nn.init.normal_(self.eos)

        # Integration of ConvSubampling for embedding
        self.conv_subsampling = Conv1dSubampling(in_channels=self.n_output, out_channels=self.d_model, reduce_time_layers=2)
        # Add ConvUpampling for upsampling, note that the argument intermediate_channels is not used
        self.conv_upsampling = Conv1dUpsampling(hidden_dim=self.d_model, reduce_time_layers=2)

        self.input_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(p=self.dropout)
        )

        # the stacked decoder layer
        self.blocks = nn.ModuleList([RetentionBlock(configs) for _ in range(self.n_layers)])

        # output layer
        self.ln_f = nn.LayerNorm(self.d_model)  # Layer Normalization
        self.head_type = head_type
        if self.head_type == "pretrain":
            # we suppose the token is [batch_size x seq_len x n_output]
            self.head = PretrainHead(self.d_model, self.n_output)
        elif self.head_type == "clf":
            self.head = ClfHead(self.d_model, self.n_output)
        elif self.head_type == "regr":
            self.head = RegrHead(self.d_model, self.n_output)
        else:
            raise ValueError("Invalid head_type provided.")

        self.gradient_checkpointing = configs.use_grad_ckp

    def forward(self,
                X, y,
                retention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                forward_impl: Optional[str] = 'chunkwise', # chunkwise
                chunk_size: Optional[int] = None,
                sequence_offset: Optional[int] = 0,
                output_retentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = True,
                ):
        # Use ConvSubsampling as tokenizer, input_project as embedding layer
        X, X_tokens = self.conv_subsampling(X)
        hidden_states = self.input_projection(X)
        batch_size, seq_len, dim = X.shape

        # Add the SOS and EOS token to the input sequence
        sos_token = self.sos.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape [batch_size, 1, d_model]
        # Add the SOS token for shifted right,  so the sequence length will be seq_len + 1
        hidden_states = torch.cat([sos_token, hidden_states], dim=1)
        eos_token = self.eos.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape [batch_size, 1, d_model]
        # Add the SOS token for shifted left,  so the sequence length will be seq_len + 2
        hidden_states = torch.cat([hidden_states, eos_token], dim=1)

        if retention_mask is None: # what is the usage of rentention mask
            # not sure whether we need to mask the first token (SOS token)
            retention_mask = torch.ones((batch_size, seq_len+2), dtype=torch.bool, device=X.device) # batch_size x token_num

        hidden_states_NTP = None
        hidden_states_PTP = None
        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        present_key_values = ()  # To store current key-value pairs
        for l, block in enumerate(self.blocks):
            # For even layer, we actually do not need for backward attention, just reverse each sequence to make it reverse.
            # It is equal to perform backward attention on a forward sequence.
            # reverse the hidden states of tokens for each layer except the first layer (the first layer is forward direction)
            if l > 0:
                # reverse the sequence of hidden states
                hidden_states = torch.flip(hidden_states, [1])
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            # Use gradient checkpointing for the forward pass of the block
            if self.gradient_checkpointing and self.training:
                def custom_forward(*inputs):
                    return block(*inputs, sequence_offset, chunk_size, output_retentions)

                block_outputs = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    retention_mask,
                    forward_impl,
                    past_key_value,
                )
            else:
                block_outputs = block(hidden_states,
                                      retention_mask=retention_mask,
                                      forward_impl=forward_impl,
                                      past_key_value=past_key_value,
                                      sequence_offset=sequence_offset,
                                      chunk_size=chunk_size,
                                      output_retentions=output_retentions)
            hidden_states = block_outputs[0]
            present_key_values += (block_outputs[1],)
            if (l+1) == self.n_layers - 1: # use the hidden states from the second last layer for next token prediction
                hidden_states_NTP = hidden_states
            elif (l+1) == self.n_layers: # use the hidden states from the last layer for previous token prediction
                hidden_states_PTP = hidden_states
            if output_retentions:
                all_retentions += (block_outputs[2],)
            torch.cuda.empty_cache()
            gc.collect()
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.head_type == 'pretrain':
            # Apply the custom head on the hidden states for output
            X_NTK = self.ln_f(hidden_states_NTP)
            outputs_NTK = self.head(X_NTK)
            X_PTK = self.ln_f(hidden_states_PTP)
            outputs_PTK = self.head(X_PTK)
            return self.compute_pretrain_loss(outputs_NTK, outputs_PTK, X_tokens) # return pre-trained loss
        elif self.head_type == 'cls':
            # Apply the custom head on the SOS token
            X = self.ln_f(hidden_states_PTP[:, 0, :])
            outputs = self.head(X)
            return self.compute_classify_loss(outputs, y) # return classification loss
        elif self.head_type == 'regr':
            # Apply the custom head on the SOS token
            X = self.ln_f(hidden_states_PTP[:, 0, :])
            outputs = self.head(X)
            return self.compute_regr_loss(outputs, y) # return regression loss

    def compute_pretrain_loss(self, next_token_predictions, previous_token_predictions, token_targets):
        """
        Compute the loss of the pre-training task (next token prediction)
        """
        self.mse_loss = nn.MSELoss()

        # For next-token prediction, the start token will be used to predict the word token;
        # thereby we use the first N tokens out of the N+2 to predict the next tokens.
        next_token_loss = self.mse_loss(next_token_predictions[:, :-2, :], token_targets)
        # For previous-token prediction, the end token will be used to predict the last word token
        # thereby we use the last N tokens out of the N+2 to predict the previous tokens.
        previous_token_loss = self.mse_loss(previous_token_predictions[:, 2:, :], token_targets)

        return next_token_loss + previous_token_loss

    def compute_regr_loss(self, regr_predictions, regr_targets):
        """
        Compute the loss of the regression task
        """
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        # Ensure regr_targets is a float tensor and has the same shape as regr_predictions
        regr_targets = regr_targets.float().view_as(regr_predictions)
        # regr_loss = self.mse_loss(regr_predictions, regr_targets)
        regr_loss = F.l1_loss(regr_predictions, regr_targets)  # L1 loss is equivalent to MAE
        return regr_loss

    def compute_cls_loss(self, cls_logits, cls_targets):
        """
        Compute the loss of classification task
        """
        # compute cross entropy loss
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # make prediction
        probs = F.softmax(cls_logits, dim=1)
        predicted = torch.argmax(probs, dim=1)

        # compute Accuracy
        correct = (predicted == cls_targets).float()
        accuracy = correct.mean() * 100.0

        return accuracy


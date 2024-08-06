import yaml
from transformers.configuration_utils import PretrainedConfig


def load_config_from_yaml(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = RetNetConfig.from_dict(config)
    return config


class RetNetConfig(PretrainedConfig):
    model_type = "retnet"

    def __init__(self,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 d_model: int = 200,
                 qk_dim: int = 200, #not
                 v_dim: int = 400, #not
                 ffn_proj_size: int = 800,
                 use_bias_in_msr: bool = False,
                 use_bias_in_mlp: bool = True,
                 use_bias_in_msr_out: bool = False,
                 use_default_gamma: bool = False,
                 initializer_range: float = 0.02, # not init right now
                 is_decoder: bool = True,
                 output_retentions: bool = False,
                 use_cache: bool = True,
                 forward_impl: str = 'parallel',  # not set
                 **kwargs):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.ffn_proj_size = ffn_proj_size
        self.use_bias_in_msr = use_bias_in_msr
        self.use_bias_in_mlp = use_bias_in_mlp
        self.use_bias_in_msr_out = use_bias_in_msr_out
        self.use_default_gamma = use_default_gamma
        self.initializer_range = initializer_range
        self.output_retentions = output_retentions
        self.forward_impl = forward_impl

        super().__init__(is_decoder=is_decoder,
                         use_cache=use_cache,
                         **kwargs)

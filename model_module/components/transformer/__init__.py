from .encoder import Encoder
from .decoder import Decoder
from .timesformer.timesformer import TimeSformer


def get_transformer(idim, odim, transformer_cfg):
    encoder = Encoder(
        idim=idim,
        attention_dim=transformer_cfg.adim,
        attention_heads=transformer_cfg.aheads,
        linear_units=transformer_cfg.eunits,
        num_blocks=transformer_cfg.elayers,
        input_layer=transformer_cfg.transformer_embed_layer,
        dropout_rate=transformer_cfg.dropout_rate,
        positional_dropout_rate=transformer_cfg.dropout_rate,
        attention_dropout_rate=transformer_cfg.transformer_attn_dropout_rate,
        encoder_attn_layer_type=transformer_cfg.transformer_encoder_attn_layer_type,
        macaron_style=transformer_cfg.macaron_style,
        use_cnn_module=transformer_cfg.use_cnn_module,
        cnn_module_kernel=transformer_cfg.cnn_module_kernel,
        zero_triu=transformer_cfg.zero_triu,
        talking_head=transformer_cfg.talking_head
    )

    if transformer_cfg.mtlalpha < 1:
        decoder = Decoder(
            odim=odim,
            attention_dim=transformer_cfg.ddim,
            attention_heads=transformer_cfg.dheads,
            linear_units=transformer_cfg.dunits,
            num_blocks=transformer_cfg.dlayers,
            dropout_rate=transformer_cfg.dropout_rate,
            positional_dropout_rate=transformer_cfg.dropout_rate,
            self_attention_dropout_rate=transformer_cfg.transformer_attn_dropout_rate,
            src_attention_dropout_rate=transformer_cfg.transformer_attn_dropout_rate,
        )
    else:
        decoder = None
    
    return encoder, decoder

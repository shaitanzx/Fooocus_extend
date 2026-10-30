import torch

from ldm_patched.ldm.modules.attention import optimized_attention
from ldm_patched.modules import utils as ldm_utils


# Порядок и имена соответствуют checkpoint LayerDiffusion:
# hookers.layers.0 ... hookers.layers.31
MODULE_MAPPING_SD15 = {
    0: "input_blocks.1.1.transformer_blocks.0.attn1",
    1: "input_blocks.1.1.transformer_blocks.0.attn2",
    2: "input_blocks.2.1.transformer_blocks.0.attn1",
    3: "input_blocks.2.1.transformer_blocks.0.attn2",
    4: "input_blocks.4.1.transformer_blocks.0.attn1",
    5: "input_blocks.4.1.transformer_blocks.0.attn2",
    6: "input_blocks.5.1.transformer_blocks.0.attn1",
    7: "input_blocks.5.1.transformer_blocks.0.attn2",
    8: "input_blocks.7.1.transformer_blocks.0.attn1",
    9: "input_blocks.7.1.transformer_blocks.0.attn2",
    10: "input_blocks.8.1.transformer_blocks.0.attn1",
    11: "input_blocks.8.1.transformer_blocks.0.attn2",

    12: "output_blocks.3.1.transformer_blocks.0.attn1",
    13: "output_blocks.3.1.transformer_blocks.0.attn2",
    14: "output_blocks.4.1.transformer_blocks.0.attn1",
    15: "output_blocks.4.1.transformer_blocks.0.attn2",
    16: "output_blocks.5.1.transformer_blocks.0.attn1",
    17: "output_blocks.5.1.transformer_blocks.0.attn2",
    18: "output_blocks.6.1.transformer_blocks.0.attn1",
    19: "output_blocks.6.1.transformer_blocks.0.attn2",
    20: "output_blocks.7.1.transformer_blocks.0.attn1",
    21: "output_blocks.7.1.transformer_blocks.0.attn2",
    22: "output_blocks.8.1.transformer_blocks.0.attn1",
    23: "output_blocks.8.1.transformer_blocks.0.attn2",
    24: "output_blocks.9.1.transformer_blocks.0.attn1",
    25: "output_blocks.9.1.transformer_blocks.0.attn2",
    26: "output_blocks.10.1.transformer_blocks.0.attn1",
    27: "output_blocks.10.1.transformer_blocks.0.attn2",
    28: "output_blocks.11.1.transformer_blocks.0.attn1",
    29: "output_blocks.11.1.transformer_blocks.0.attn2",

    30: "middle_block.1.transformer_blocks.0.attn1",
    31: "middle_block.1.transformer_blocks.0.attn2",
}


class LoRALinearLayer(torch.nn.Module):
    """
    LayerDiffusion rank-256 linear LoRA.

    The original Linear module is kept as a non-registered reference.
    This preserves the checkpoint key layout.
    """

    def __init__(self, in_features, out_features, rank=256, original=None):
        super().__init__()
        self.down = torch.nn.Linear(in_features, rank, bias=False)
        self.up = torch.nn.Linear(rank, out_features, bias=False)

        # A list prevents registering the original model layer as a child module.
        self.original = [original]

    def forward(self, x):
        original = self.original[0]
        weight = original.weight.to(device=x.device, dtype=x.dtype)
        bias = original.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)

        down_weight = self.down.weight.to(device=x.device, dtype=x.dtype)
        up_weight = self.up.weight.to(device=x.device, dtype=x.dtype)

        merged_weight = weight + torch.mm(up_weight, down_weight)
        return torch.nn.functional.linear(x, merged_weight, bias)


class AttentionSharingUnit(torch.nn.Module):
    """
    Wrapper replacing one SD1.5 CrossAttention module.

    LayerDiffusion's released Attention Injection checkpoint uses one frame.
    Fooocus does not provide Forge's cond_mark/cond_overwrite arguments, so
    this port handles the single-frame Attention Injection path only.
    """

    def __init__(self, attention_module, rank=256):
        super().__init__()

        self.heads = attention_module.heads

        # Keep the original attention module as a non-registered reference.
        self.original_module = [attention_module]

        self.to_q_lora = torch.nn.ModuleList([
            LoRALinearLayer(
                attention_module.to_q.in_features,
                attention_module.to_q.out_features,
                rank,
                attention_module.to_q,
            )
        ])
        self.to_k_lora = torch.nn.ModuleList([
            LoRALinearLayer(
                attention_module.to_k.in_features,
                attention_module.to_k.out_features,
                rank,
                attention_module.to_k,
            )
        ])
        self.to_v_lora = torch.nn.ModuleList([
            LoRALinearLayer(
                attention_module.to_v.in_features,
                attention_module.to_v.out_features,
                rank,
                attention_module.to_v,
            )
        ])
        self.to_out_lora = torch.nn.ModuleList([
            LoRALinearLayer(
                attention_module.to_out[0].in_features,
                attention_module.to_out[0].out_features,
                rank,
                attention_module.to_out[0],
            )
        ])

        hidden_size = attention_module.to_k.out_features

        # Names and shapes must match hookers.layers.* in the checkpoint.
        self.temporal_i = torch.nn.Linear(hidden_size, hidden_size)
        self.temporal_n = torch.nn.LayerNorm(
            hidden_size,
            elementwise_affine=True,
            eps=1e-6,
        )
        self.temporal_q = torch.nn.Linear(hidden_size, hidden_size)
        self.temporal_k = torch.nn.Linear(hidden_size, hidden_size)
        self.temporal_v = torch.nn.Linear(hidden_size, hidden_size)
        self.temporal_o = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x, context=None, value=None, mask=None):
        # Fooocus calls this replacement with [batch, sequence, channels].
        # Attention Injection is single-frame, so no frame rearrangement is needed.
        context = x if context is None else context
        value = context if value is None else value

        q = self.to_q_lora[0](x)
        k = self.to_k_lora[0](context)
        v = self.to_v_lora[0](value)

        attended = optimized_attention(q, k, v, self.heads, mask=mask)

        out = self.to_out_lora[0](attended)
        out = self.original_module[0].to_out[1](out)

        # The upstream implementation adds temporal attention after the normal
        # attention output. With frames=1, the temporal sequence length is one.
        modified = x + out
        temporal = self.temporal_i(self.temporal_n(modified))

        batch, sequence, channels = temporal.shape
        temporal = temporal.reshape(batch * sequence, 1, channels)

        temporal_q = self.temporal_q(temporal)
        temporal_k = self.temporal_k(temporal)
        temporal_v = self.temporal_v(temporal)

        temporal = optimized_attention(
            temporal_q,
            temporal_k,
            temporal_v,
            self.heads,
        )
        temporal = self.temporal_o(temporal)
        temporal = temporal.reshape(batch, sequence, channels)

        modified = modified + temporal

        # BasicTransformerBlock adds the returned value to its residual input.
        return modified - x


class HookerLayers(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class SD15AttentionSharingPatcher(torch.nn.Module):
    def __init__(self, target_unet):
        super().__init__()

        diffusion_model = target_unet.model.diffusion_model
        units = []

        for index in range(32):
            layer_path = MODULE_MAPPING_SD15[index]
            original_attention = ldm_utils.get_attr(
                diffusion_model,
                layer_path,
            )

            unit = AttentionSharingUnit(original_attention)
            units.append(unit)

            # Fooocus ModelPatcher installs these by nested attribute path.
            target_unet.add_object_patch(
                "diffusion_model." + layer_path,
                unit,
            )

        self.hookers = HookerLayers(units)


def apply_sd15_transparent_attention(target_unet, state_dict):
    patcher = SD15AttentionSharingPatcher(target_unet)

    # Fail explicitly if the downloaded file is not the expected SD1.5 checkpoint.
    patcher.load_state_dict(state_dict, strict=True)
    patcher.eval()

    return patcher

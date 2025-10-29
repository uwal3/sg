import torch
import torch.nn as nn

from diffusers.models.unets.unet_2d_blocks import AttnUpDecoderBlock2D


class SkinDecoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()

        self.block_out_channels = [256, 128, 64]
        self.layers_per_block = 2
        self.attention_head_dim = 32

        self.conv_in = nn.Conv2d(
            in_channels, self.block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList()

        prev_output_channel = self.block_out_channels[0]

        for i, out_ch in enumerate(self.block_out_channels):
            block = AttnUpDecoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=prev_output_channel,
                out_channels=out_ch,
                add_upsample=False,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                attention_head_dim=self.attention_head_dim,
                temb_channels=None,
            )
            self.blocks.append(block)
            prev_output_channel = out_ch

        self.norm_out = nn.GroupNorm(32, self.block_out_channels[-1])
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(
            self.block_out_channels[-1], out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)
        return {"sample": h}

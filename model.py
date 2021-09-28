import math
from packaging import version

import torch
from torch import nn
from torch.nn import functional as F

from stylegan2.model import StyledConv, Blur, EqualLinear, EqualConv2d, ScaledLeakyReLU
from stylegan2.op import FusedLeakyReLU

from models.resnet import resnet50

class EqualConvTranspose2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=(1, 3, 3, 1),
        bias=True,
        activate=True,
        padding="zero",
    ):
        layers = []

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2

        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))

                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class StyledResBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, style_dim, upsample, blur_kernel=(1, 3, 3, 1)
    ):
        super().__init__()

        self.conv1 = StyledConv(
            in_channel,
            out_channel,
            3,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
        )

        self.conv2 = StyledConv(out_channel, out_channel, 3, style_dim)

        if upsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel,
                out_channel,
                1,
                upsample=upsample,
                blur_kernel=blur_kernel,
                bias=False,
                activate=False,
            )

        else:
            self.skip = None

    def forward(self, input, style, noise=None):
        out = self.conv1(input, style, noise)
        out = self.conv2(out, style, noise)

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        return (out + skip) / math.sqrt(2)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        downsample,
        padding="zero",
        blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, out_channel, 3, padding=padding)

        self.conv2 = ConvLayer(
            out_channel,
            out_channel,
            3,
            downsample=downsample,
            padding=padding,
            blur_kernel=blur_kernel,
        )

        if downsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel,
                out_channel,
                1,
                downsample=downsample,
                blur_kernel=blur_kernel,
                bias=False,
                activate=False,
            )

        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        # print(out.shape)

        return (out + skip) / math.sqrt(2)

class Stem(nn.Module):
    def __init__(
        self,
        channel,
        gray=False
    ):
        super().__init__()
        if gray:
            self.stem = ConvLayer(1, channel, 3)
        else:
            self.stem = ConvLayer(3, channel, 3)

    def forward(self, input):
        out = self.stem(input)

        return out

class StrBranch(nn.Module):
    def __init__(self, channel, structure_channel=8):
        super().__init__()
        
        scale1 = []
        in_channel = channel
        for i in range(0,1):
            ch = channel * (2 ** i)
            scale1.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale1 = nn.Sequential(*scale1)
        
        scale2 = []
        for i in range(1,2):
            ch = channel * (2 ** i)
            scale2.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale2 = nn.Sequential(*scale2)

        scale3 = []
        for i in range(2,4):
            ch = channel * (2 ** i)
            scale3.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale3 = nn.Sequential(*scale3)

        self.structure = nn.Sequential(
            ConvLayer(ch, ch, 1), ConvLayer(ch, structure_channel, 1)
        )

    def forward(self, input, multi_out):
        scale1 = self.scale1(input)
        scale2 = self.scale2(scale1)
        scale3 = self.scale3(scale2)
        structure = self.structure(scale3)

        if multi_out:
            return scale1, scale2, scale3, structure
        else:
            return structure

class CifarStrBranch(nn.Module):
    def __init__(self, channel, structure_channel=8):
        super().__init__()
        
        scale1 = []
        in_channel = channel
        for i in range(0,1):
            ch = channel * (2 ** i)
            scale1.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale1 = nn.Sequential(*scale1)
        
        scale2 = []
        for i in range(1,2):
            ch = channel * (2 ** i)
            scale2.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale2 = nn.Sequential(*scale2)

        scale3 = []
        for i in range(2,3):
            ch = channel * (2 ** i)
            scale3.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale3 = nn.Sequential(*scale3)

        self.structure = nn.Sequential(
            ConvLayer(ch, ch, 1), ConvLayer(ch, structure_channel, 1)
        )

    def forward(self, input, multi_out):
        scale1 = self.scale1(input)
        scale2 = self.scale2(scale1)
        scale3 = self.scale3(scale2)
        structure = self.structure(scale3)

        if multi_out:
            return scale1, scale2, scale3, structure
        else:
            return structure

class TexBranch(nn.Module):
    def __init__(self, channel, texture_channel=8):
        super().__init__()
        
        scale1 = []
        in_channel = channel
        for i in range(0,1):
            ch = channel * (2 ** i)
            scale1.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale1 = nn.Sequential(*scale1)
        
        scale2 = []
        for i in range(1,2):
            ch = channel * (2 ** i)
            scale2.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale2 = nn.Sequential(*scale2)

        scale3 = []
        for i in range(2,4):
            ch = channel * (2 ** i)
            scale3.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale3 = nn.Sequential(*scale3)

        self.texture = nn.Sequential(
            ConvLayer(ch, ch * 2, 3, downsample=True, padding="valid"),
            ConvLayer(ch * 2, ch * 4, 3, downsample=True, padding="valid"),
            nn.AdaptiveAvgPool2d(1),
            ConvLayer(ch * 4, texture_channel, 1),
        )

    def forward(self, input, multi_out=True):
        scale1 = self.scale1(input)
        scale2 = self.scale2(scale1)
        scale3 = self.scale3(scale2)
        texture =  torch.flatten(self.texture(scale3), 1)

        if multi_out:
            return scale1, scale2, scale3, texture
        else:
            return texture

class CifarTexBranch(nn.Module):
    def __init__(self, channel, texture_channel=128):
        super().__init__()
        
        scale1 = []
        in_channel = channel
        for i in range(0,1):
            ch = channel * (2 ** i)
            scale1.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale1 = nn.Sequential(*scale1)
        
        scale2 = []
        for i in range(1,2):
            ch = channel * (2 ** i)
            scale2.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale2 = nn.Sequential(*scale2)

        scale3 = []
        for i in range(2,3):
            ch = channel * (2 ** i)
            scale3.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale3 = nn.Sequential(*scale3)

        self.texture = nn.Sequential(
            ConvLayer(ch, ch * 2, 2, downsample=False, padding="valid"),
            nn.AdaptiveAvgPool2d(1),
            ConvLayer(ch * 2, texture_channel, 1),
        )

    def forward(self, input, multi_out=True):
        scale1 = self.scale1(input)
        scale2 = self.scale2(scale1)
        scale3 = self.scale3(scale2)
        texture =  torch.flatten(self.texture(scale3), 1)

        if multi_out:
            return scale1, scale2, scale3, texture
        else:
            return texture

class PyramidEncoder(nn.Module):
    def __init__(self, channel, structure_channel=8, texture_channel=2048, gray=False):
        super().__init__()

        self.stem = Stem(channel, gray=gray)
        self.str_branch = StrBranch(channel, structure_channel)
        self.tex_branch = TexBranch(channel, texture_channel)

    def forward(self, input, run_str=True, run_tex=True, multi_str=False, multi_tex=True):
        structures = None
        textures = None
        out = self.stem(input)
        if run_str:
            structures = self.str_branch(out, multi_out=multi_str)
        if run_tex:
            textures = self.tex_branch(out, multi_out=multi_tex)
        return structures, textures

class CifarPyramidEncoder(nn.Module):
    def __init__(self, channel, structure_channel=8, texture_channel=128, gray=False):
        super().__init__()

        self.stem = Stem(channel, gray=gray)
        self.str_branch = CifarStrBranch(channel, structure_channel)
        self.tex_branch = CifarTexBranch(channel, texture_channel)

    def forward(self, input, run_str=True, run_tex=True, multi_str=False, multi_tex=True):
        structures = None
        textures = None
        out = self.stem(input)
        if run_str:
            structures = self.str_branch(out, multi_out=multi_str)
        if run_tex:
            textures = self.tex_branch(out, multi_out=multi_tex)
        return structures, textures

class Encoder(nn.Module):
    def __init__(
        self,
        channel,
        structure_channel=8,
        texture_channel=2048,
        blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        stem = [ConvLayer(3, channel, 1)]

        in_channel = channel
        for i in range(1, 5):
            ch = channel * (2 ** i)
            stem.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch

        self.stem = nn.Sequential(*stem)

        self.structure = nn.Sequential(
            ConvLayer(ch, ch, 1), ConvLayer(ch, structure_channel, 1)
        )

        self.texture = nn.Sequential(
            ConvLayer(ch, ch * 2, 3, downsample=True, padding="valid"),
            ConvLayer(ch * 2, ch * 4, 3, downsample=True, padding="valid"),
            nn.AdaptiveAvgPool2d(1),
            ConvLayer(ch * 4, texture_channel, 1),
        )

    def forward(self, input):
        out = self.stem(input)

        structure = self.structure(out)
        texture = torch.flatten(self.texture(out), 1)

        return structure, texture

class LightEncoder(nn.Module):
    def __init__(
        self,
        channel,
        structure_channel=8,
        texture_channel=1024,
        blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        stem = [ConvLayer(3, channel, 1)]

        in_channel = channel
        for i in range(1, 5):
            ch = channel * (2 ** i)
            stem.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch

        self.stem = nn.Sequential(*stem)

        self.structure = nn.Sequential(
            ConvLayer(ch, ch//4, 1), ConvLayer(ch//4, structure_channel, 1)
        )

        self.texture = nn.Sequential(
            ConvLayer(ch, ch, 3, downsample=True, padding="valid"),
            ConvLayer(ch, ch * 2, 3, downsample=True, padding="valid"),
            nn.AdaptiveAvgPool2d(1),
            ConvLayer(ch * 2, texture_channel, 1),
        )

    def forward(self, input):
        out = self.stem(input)

        structure = self.structure(out)
        texture = torch.flatten(self.texture(out), 1)

        return structure, texture

class Generator(nn.Module):
    def __init__(
        self,
        channel,
        structure_channel=8,
        texture_channel=2048,
        blur_kernel=(1, 3, 3, 1),
        gray=False
    ):
        super().__init__()
        self.gray = gray

        ch_multiplier = (4, 8, 12, 16, 16, 16, 8, 4)
        upsample = (False, False, False, False, True, True, True, True)

        self.layers = nn.ModuleList()
        in_ch = structure_channel
        for ch_mul, up in zip(ch_multiplier, upsample):
            self.layers.append(
                StyledResBlock(
                    in_ch, channel * ch_mul, texture_channel, up, blur_kernel
                )
            )
            in_ch = channel * ch_mul

        if gray:
            self.to_img = ConvLayer(in_ch, 1, 1, activate=False)
        else:
            self.to_rgb = ConvLayer(in_ch, 3, 1, activate=False)

    def forward(self, structure, texture, noises=None):
        if noises is None:
            noises = [None] * len(self.layers)

        out = structure
        for layer, noise in zip(self.layers, noises):
            out = layer(out, texture, noise)

        if self.gray:
            out = self.to_img(out)
        else:
            out = self.to_rgb(out)

        return out

class CifarGenerator(nn.Module):
    def __init__(
        self,
        channel,
        structure_channel=8,
        texture_channel=128,
        blur_kernel=(1, 3, 3, 1),
        gray=False
    ):
        super().__init__()
        self.gray = gray

        ch_multiplier = (2, 4, 6, 8, 8, 4, 2)
        upsample = (False, False, False, False, True, True, True)

        self.layers = nn.ModuleList()
        in_ch = structure_channel
        for ch_mul, up in zip(ch_multiplier, upsample):
            self.layers.append(
                StyledResBlock(
                    in_ch, channel * ch_mul, texture_channel, up, blur_kernel
                )
            )
            in_ch = channel * ch_mul

        if gray:
            self.to_img = ConvLayer(in_ch, 1, 1, activate=False)
        else:
            self.to_rgb = ConvLayer(in_ch, 3, 1, activate=False)

    def forward(self, structure, texture, noises=None):
        if noises is None:
            noises = [None] * len(self.layers)

        out = structure
        for layer, noise in zip(self.layers, noises):
            out = layer(out, texture, noise)

        if self.gray:
            out = self.to_img(out)
        else:
            out = self.to_rgb(out)

        return out

class OldGenerator(nn.Module):
    def __init__(
        self,
        channel,
        structure_channel=8,
        texture_channel=2048,
        blur_kernel=(1, 3, 3, 1),
        gray=False
    ):
        super().__init__()

        ch_multiplier = (4, 8, 12, 16, 16, 16, 8, 4)
        upsample = (False, False, False, False, True, True, True, True)

        self.layers = nn.ModuleList()
        in_ch = structure_channel
        for ch_mul, up in zip(ch_multiplier, upsample):
            self.layers.append(
                StyledResBlock(
                    in_ch, channel * ch_mul, texture_channel, up, blur_kernel
                )
            )
            in_ch = channel * ch_mul

        if gray:
            self.to_rgb = ConvLayer(in_ch, 1, 1, activate=False)
        else:
            self.to_rgb = ConvLayer(in_ch, 3, 1, activate=False)

    def forward(self, structure, texture, noises=None):
        if noises is None:
            noises = [None] * len(self.layers)

        out = structure
        for layer, noise in zip(self.layers, noises):
            out = layer(out, texture, noise)

        out = self.to_rgb(out)

        return out

class AugGenerator(nn.Module):
    def __init__(
        self,
        channel,
        structure_channel=8,
        texture_channel=1024,
        blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        ch_multiplier = (4, 8, 12, 16, 16, 16, 8, 4)
        upsample = (False, False, False, False, True, True, True, True)

        self.blank_style = nn.Parameter(torch.empty(1, texture_channel))
        nn.init.normal_(self.blank_style)

        self.layers = nn.ModuleList()
        in_ch = structure_channel
        for ch_mul, up in zip(ch_multiplier, upsample):
            self.layers.append(
                StyledResBlock(
                    in_ch, channel * ch_mul, texture_channel, up, blur_kernel
                )
            )
            in_ch = channel * ch_mul

        self.to_rgb = ConvLayer(in_ch, 3, 1, activate=False)

    def forward(self, structure, texture=None, noises=None):
        if noises is None:
            noises = [None] * len(self.layers)
        
        batch_size = structure.size(0)
        if texture is None:
            final_style = self.blank_style.repeat(batch_size, 1)
        else:
            final_style = self.blank_style + texture

        out = structure
        for layer, noise in zip(self.layers, noises):
            out = layer(out, final_style, noise)

        out = self.to_rgb(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=(1, 3, 3, 1), gray=False):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        if gray:
            convs = [ConvLayer(1, channels[size], 1)]
        else:
            convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, downsample=True))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)

        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)

        return out

class CifarDiscriminator(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=(1, 3, 3, 1), gray=False):
        super().__init__()

        channels = {
            4: 128,
            8: 64,
            16: 64,
            32: 32 * channel_multiplier,
            64: 32 * channel_multiplier,
            128: 32 * channel_multiplier,
        }

        if gray:
            convs = [ConvLayer(1, channels[size], 1)]
        else:
            convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, downsample=True))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)

        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)

        return out

class CooccurDiscriminator(nn.Module):
    def __init__(self, channel, size=256):
        super().__init__()

        encoder = [ConvLayer(3, channel, 1)]

        if size >= 32:
            ch_multiplier = (2, 4, 8, 12, 12, 24)
            downsample = (True, True, True, True, True, False)
        elif size == 16:
            ch_multiplier = (2, 4, 8, 12, 24)
            downsample = (True, True, True, True, False)
        elif size == 8:
            ch_multiplier = (2, 4, 8, 12)
            downsample = (True, True, True, False)
        in_ch = channel
        for ch_mul, down in zip(ch_multiplier, downsample):
            encoder.append(ResBlock(in_ch, channel * ch_mul, down))
            in_ch = channel * ch_mul

        if size > 511:
            k_size = 3
            feat_size = 2 * 2

        else:
            k_size = 2
            feat_size = 1 * 1

        encoder.append(ConvLayer(in_ch, channel * 12, k_size, padding="valid"))

        self.encoder = nn.Sequential(*encoder)

        self.linear = nn.Sequential(
            EqualLinear(
                channel * 12 * 2 * feat_size, channel * 32, activation="fused_lrelu"
            ),
            EqualLinear(channel * 32, channel * 32, activation="fused_lrelu"),
            EqualLinear(channel * 32, channel * 16, activation="fused_lrelu"),
            EqualLinear(channel * 16, 1),
        )

    def forward(self, input, reference=None, ref_batch=None, ref_input=None):
        # print(input.shape)
        out_input = self.encoder(input)

        if ref_input is None:
            ref_input = self.encoder(reference)
            _, channel, height, width = ref_input.shape
            ref_input = ref_input.view(-1, ref_batch, channel, height, width)
            ref_input = ref_input.mean(1)

        out = torch.cat((out_input, ref_input), 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out, ref_input


class Cooccurv2Discriminator(nn.Module):
    """
    This new cooccur discriminator is to modify some details
    First, it applies max-pooling on the n_crop
    Second, it applied average-pooling on the spatial dimension
    """
    def __init__(self, channel, size=256, sup=False, gray=False):
        super().__init__()
        self.sup = sup
        if gray:
            encoder = [ConvLayer(1, channel, 1)]
        else:
            encoder = [ConvLayer(3, channel, 1)]

        if size >= 32:
            ch_multiplier = (2, 4, 8, 12, 24)
            downsample = (True, True, True, True, False)
        elif size == 16:
            ch_multiplier = (2, 4, 8, 12)
            downsample = (True, True, True, False)
        elif size == 8:
            ch_multiplier = (2, 4, 8)
            downsample = (True, True, False)
        else:
            raise ValueError(f"Unsupported input size {size} for Cooccurv2Discriminator")
        
        in_ch = channel
        for ch_mul, down in zip(ch_multiplier, downsample):
            encoder.append(ResBlock(in_ch, channel * ch_mul, down))
            in_ch = channel * ch_mul

        # last conv layer
        k_size = 3 if size >= 256 else 1
        encoder.append(ConvLayer(in_ch, channel * 12, k_size, padding="valid"))

        # Average pool over spatial dimension
        encoder.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.encoder = nn.Sequential(*encoder)

        self.linear = nn.Sequential(
            EqualLinear(
                channel * 12 * 2, channel * 32, activation="fused_lrelu"
            ),
            EqualLinear(channel * 32, channel * 32, activation="fused_lrelu"),
            EqualLinear(channel * 32, channel * 16, activation="fused_lrelu"),
            EqualLinear(channel * 16, 1),
        )

        if sup:
            self.cls_fc = nn.Sequential(
                EqualLinear(channel * 12, channel * 16, activation="fused_lrelu"),
                EqualLinear(channel * 16, 14),
            )

    def forward(self, input, n_crop, reference=None, ref_batch=None, ref_input=None):
        ref_pred = None
        # [batch*n_crop, channel, h, w]
        out_input = self.encoder(input)
        _, channel, height, width = out_input.shape
        # [batch, channel, h, w]
        out_input = out_input.view(-1, n_crop, channel, height, width).max(1)[0]

        # [batch, channel, h, w]
        if ref_input is None:
            ref_input = self.encoder(reference)
            _, channel, height, width = ref_input.shape
            ref_input = ref_input.view(-1, n_crop*ref_batch, channel, height, width)
            ref_input = ref_input.max(1)[0]
            if self.sup:
                ref_pred = self.cls_fc(torch.flatten(ref_input, 1))

        out = torch.cat((out_input, ref_input), 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        if ref_pred is not None:
            return out, ref_input, ref_pred
        else:
            return out, ref_input

class PatchDiscriminator(nn.Module):
    """
    This discriminator applies to feature vectors.
    First, it applies max-pooling on all the vectors after some fc layers.
    Lastly, it concats two pooled vectors.
    """
    def __init__(self, channels, weights=1, code_dim=512):
        super().__init__()
        if not isinstance(weights, (tuple, list)):
            weights = [weights] * len(channels)
        assert(len(weights) == len(channels))
        self.weights = weights

        self.encoder = nn.ModuleList()
        for channel in channels:
            enc = nn.Sequential(
                EqualLinear(channel, code_dim, activation="fused_lrelu"),
                EqualLinear(code_dim, code_dim, activation="fused_lrelu"),
            )
            self.encoder.append(enc)

        self.linear = nn.Sequential(
            EqualLinear(code_dim * len(channels) * 2, code_dim, activation="fused_lrelu"),
            EqualLinear(code_dim, code_dim, activation="fused_lrelu"),
            EqualLinear(code_dim, 1),
        )

    def forward(self, inputs, n_crop, references=None, ref_inputs=None):
        # encode inputs
        out_inputs = []
        for i, input in enumerate(inputs):
            out_inputs.append(self.weights[i] * self.encoder[i](input))
        # [batch*n_crop, 1536]
        out_inputs = torch.cat(out_inputs, 1)
        _, channel = out_inputs.shape
        # [batch, channel]
        out_inputs = out_inputs.view(-1, n_crop, channel).max(1)[0]

        # [batch*n_crop, channel]
        if ref_inputs is None:
            ref_inputs = []
            for i, ref in enumerate(references):
                ref_inputs.append(self.weights[i] * self.encoder[i](ref))
            ref_inputs = torch.cat(ref_inputs, 1)
            _, channel = ref_inputs.shape
            ref_inputs = ref_inputs.view(-1, n_crop, channel).max(1)[0]

        out = torch.cat((out_inputs, ref_inputs), 1)
        out = self.linear(out)

        return out, ref_inputs

class MultiProjectors(nn.Module):
    def __init__(self, channels, use_mlp=True, norm=True):
        super().__init__()
        self.use_mlp = use_mlp
        self.norm = norm

        self.projectors = nn.ModuleList()
        for channel in channels:
            proj = nn.Sequential(
                EqualLinear(channel, channel // 2, activation="fused_lrelu"),
                EqualLinear(channel // 2, channel)
            )
            self.projectors.append(proj)

    def forward(self, feats):
        if self.use_mlp:
            projected = []
            for i, feat in enumerate(feats):
                projected.append(self.projectors[i](feat))
        else:
            projected = feats

        if self.norm:
            normed_projected = []
            for feat in projected:
                # l2 norm after projection
                norm = feat.pow(2).sum(1, keepdim=True).pow(1. / 2)
                normed_projected.append(feat.div(norm + 1e-7))
            return normed_projected
        else:
            return projected

class PatchNCELoss(nn.Module):
    def __init__(self, nce_T, batch):
        super().__init__()
        self.nce_T = nce_T
        self.batch = batch
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit
        batch_dim_for_bmm = self.batch
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        # pdb.set_trace()

        return loss
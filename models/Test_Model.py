import numpy as np
import h5py
import os, sys
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

from models.BasicBlock import ResNet, InceptionResNet

import time

class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, channels):
        print("the channels number is :",channels)
        nn.Module.__init__(self)
        in_nchannels = 4
        ch = [32, channels]

        self.block0_en = nn.Sequential(
            ME.MinkowskiConvolution(
                in_nchannels,ch[0],kernel_size=2,stride=2,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
        )
        # tensor 的shape

        self.block1_en = nn.Sequential(
            ME.MinkowskiConvolution(
                ch[0],ch[1],kernel_size=2,stride=2,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )


    # def make_layer(self, block, block_layers, channels):
    #     layers = []
    #     for i in range(block_layers):
    #         layers.append(block(channels=channels))
    #
    #     return nn.Sequential(*layers)
    def forward(self,x):
        print("原始数据结构是：",x.F.shape)
        print("原始数据是：",x.F)
        out_en0 = self.block0_en(x)
        print("第一次压缩之后数据结构是：",out_en0.F.shape)
        print("第一次压缩之后的数据是：",out_en0.F)
        out_en1 = self.block1_en(out_en0)
        print("第二次压缩之后数据结构是：",out_en1.F.shape)


        return [out_en1,out_en0]

class Decoder(nn.Module):
    """
        one level model : for test points num decoder side
    """

    def __init__(self, channels):
        nn.Module.__init__(self)
        out_nchannels = 4
        ch = [channels,32]

        self.block0_de = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=ch[0],
                out_channels=ch[1],
                kernel_size=2,
                stride=2,
                bias=True,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )

        self.block0_cls = ME.MinkowskiConvolution(
            in_channels=ch[1],
            out_channels=out_nchannels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )
        self.block1_de=nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[1],out_nchannels,kernel_size=2,stride=2,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(out_nchannels),
            ME.MinkowskiELU(),)
        self.block1_cls = ME.MinkowskiConvolution(
            in_channels=out_nchannels,
            out_channels=out_nchannels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )

    def get_target_by_sp_tensor(self,out,target_sp_tensor):
        with torch.no_grad():
            def ravel_multi_index(coords, step):
                coords = coords.long()
                step = step.long()

                coords_sum = coords[:, 0] \
                             + coords[:, 1] * step \
                             + coords[:, 2] * step * step \
                             + coords[:, 3] * step * step * step
                return coords_sum
            step = max(out.C.max(), target_sp_tensor.C.max()) + 1
            out_sp_tensor_coords_1d = ravel_multi_index(out.C, step)
            in_sp_tensor_coords_1d = ravel_multi_index(target_sp_tensor.C,step)

            target = np.in1d(out_sp_tensor_coords_1d.cpu().numpy(),
                             in_sp_tensor_coords_1d.cpu().numpy())
        return torch.Tensor(target).bool()

    def forward(self, x, target_label, adaptive, rhos = [1.0,1.0,1.0], training = True):
        targets = []
        outcls = []
        out_geo = []
        keeps = []

        out_de0 = self.block0_de(x)
        print("第一次解压缩之后的数据结构：",out_de0.F.shape)
        print("第一次解压缩之后的数据是：",out_de0.F)
        #out_de = self.res_block0_de(out_de)
        out_cls0 = self.block0_cls(out_de0)
        target0 = self.get_target_by_sp_tensor(out_de0,target_label[0])
        keep0 = (out_cls0.F[:,[1]] > 0).cpu().squeeze()
        keeps.append(keep0)
        targets.append(target0)
        outcls.append(out_cls0)

        if training:
            keep0 += target0

        #out0_pruned = self.pruning(out_de0,keep0.to(out_de0.device))
        out_geo.append(out_de0)
        out_de1 = self.block1_de(out_de0)
        print("最后一次解压缩之后的数据结果是:",out_de1.F.shape)
        print("最后一次解压缩之后的数据是：",out_de1.F)
        out_cls1 = out_de1
        target1 = self.get_target_by_sp_tensor(out_de1,target_label[1])
        keep1 = (out_cls1.F[:,[1]] > 0).cpu().squeeze()
        keeps.append(keep1)
        targets.append(target1)
        outcls.append(out_cls1)

        if training:
            keep1 += target1
        out_geo.append(out_de1)

        return out_de1,outcls,out_geo,targets,keeps

    # def make_layer(self, block, block_layers, channels):
    #     layers = []
    #     for i in range(block_layers):
    #         layers.append(block(channels=channels))
    #
    #     return nn.Sequential(*layers)

if __name__ == '__main__':
    encoder = Encoder(channels=16)
    print(encoder)
    decoder = Decoder(channels=16)
    print(decoder)







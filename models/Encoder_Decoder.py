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
    def __init__(self, channels, block_layers, block):
        print("the channels number is :",channels)
        nn.Module.__init__(self)
        in_nchannels = 4
        ch = [16, 32, 64, 32, channels]
        if block == 'ResNet':
            self.block = ResNet
        elif block == 'InceptionResNet':
            self.block = InceptionResNet

        self.block0_en = nn.Sequential(
            ME.MinkowskiConvolution(
                in_nchannels,ch[0],kernel_size=3,stride=1,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                ch[0],ch[1],kernel_size=2,stride=2,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )
        # tensor 的shape
        self.res_block0_en = self.make_layer(
            self.block,block_layers,ch[1])

        self.block1_en = nn.Sequential(
            ME.MinkowskiConvolution(
                ch[1],ch[1],kernel_size=3,stride=1,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                ch[1],ch[2],kernel_size=2,stride=2,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )
        self.res_block1_en = self.make_layer(
            self.block,block_layers,ch[2])

        self.block2_en = nn.Sequential(
            ME.MinkowskiConvolution(
                ch[2],ch[2],kernel_size=3,stride=1,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                ch[2],ch[3],kernel_size=2,stride=2,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )
        self.res_block2_en = self.make_layer(
            self.block,block_layers,ch[3])

        self.conv3_en = ME.MinkowskiConvolution(
            in_channels=ch[3],
            out_channels=ch[4],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,)


    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        out0_en = self.block0_en(x)
        out0_en = self.res_block0_en(out0_en)
        out1_en = self.block1_en(out0_en)
        out1_en = self.res_block1_en(out1_en)
        out2_en = self.block2_en(out1_en)
        out2_en = self.res_block2_en(out2_en)
        out2_en = self.conv3_en(out2_en)

        return [out2_en, out1_en, out0_en]

class Decoder(nn.Module):
    """
    decoder test
    """
    def __init__(self,channels,block_layers,block):
        nn.Module.__init__(self)
        out_nchannel = 4
        ch = [channels,64,32,16]
        if block == 'ResNet':
            self.block = ResNet
        elif block == 'InceptionResNet':
            self.block = InceptionResNet

        self.block0_de = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels = ch[0],
                out_channels = ch[1],
                kernel_size = 2,
                stride = 2,
                bias = True,
                dimension = 3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                in_channels = ch[1],
                out_channels = ch[1],
                kernel_size = 3,
                stride = 1,
                bias = True,
                dimension = 3,
            ),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )
        self.res_block0_de = self.make_layer(
            self.block,block_layers,ch[1])
        self.block0_cls = ME.MinkowskiConvolution(
            in_channels = ch[1],
            out_channels = out_nchannel,
            kernel_size = 3,
            stride = 1,
            bias = True,
            dimension = 3,
        )

        self.block1_de = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels = ch[1],
                out_channels = ch[2],
                kernel_size = 2,
                stride = 2,
                bias = True,
                dimension = 3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                in_channels = ch[2],
                out_channels = ch[2],
                kernel_size = 3,
                stride = 1,
                bias = True,
                dimension = 3,
            ),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )
        self.res_block1_de = self.make_layer(
            self.block,block_layers,ch[2])
        self.block1_cls = ME.MinkowskiConvolution(
            ch[2],
            out_nchannel,
            kernel_size = 3,
            stride = 1,
            bias = True,
            dimension = 3)

        self.block2_de = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[2],
                ch[3],
                kernel_size = 2,
                stride = 2,
                bias = True,
                dimension = 3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                ch[3],
                ch[3],
                kernel_size = 3,
                stride = 1,
                bias = True,
                dimension = 3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )
        self.res_block2_de = self.make_layer(
            self.block,block_layers,ch[3]
        )
        self.block2_cls = ME.MinkowskiConvolution(
            ch[3],
            out_nchannel,
            kernel_size = 3,
            stride = 1,
            bias = True,
            dimension = 3)

        # 最后一次卷积，卷成初始的四维坐标
        self.conv3_de = ME.MinkowskiConvolutionTranspose(
            ch[3],
            out_nchannel,
            kernel_size = 3,
            stride = 1,
            bias = True,
            dimension = 3)

        self.pruning = ME.MinkowskiPruning()
        # 往下接着写get target by tensor 和 forward函数

    def get_target_by_sp_tensor(self, out, target_sp_tensor):
        # with torch.no_grad() 指的是后面的操作不进行计算图的生成，计算图用于后续的bp算法
        with torch.no_grad():
            def ravel_multi_index(coords, step):
                # .long() 将数字转换成长整形
                coords = coords.long()
                step = step.long()

                coords_sum = coords[:, 0] \
                             + coords[:, 1] * step \
                             + coords[:, 2] * step * step \
                             + coords[:, 3] * step * step * step
                return coords_sum

            step = max(out.C.max(), target_sp_tensor.C.max()) + 1
            out_sp_tensor_coords_1d = ravel_multi_index(out.C, step)
            in_sp_tensor_coords_1d = ravel_multi_index(target_sp_tensor.C, step)

            # test whether each element of a 1-D array is also present in a second array.
            # np.in1d(A,B) 在序列B中寻找与序列A相同的值，并返回一逻辑值（True,False）或逻辑值构成的向量
            target = np.in1d(out_sp_tensor_coords_1d.cpu().numpy(),
                             in_sp_tensor_coords_1d.cpu().numpy())
            # print("the new target is :",target)

        return torch.Tensor(target).bool()

    def forward(self, x, target_label, adaptive, rhos=[1.0, 1.0, 1.0], training=True):
        targets = []
        out_cls = []
        # return the geo-feats info of the output tensor
        out_geo = []
        keeps = []

        print("decoder side the original shape is :",x.F.shape)

        # Block0  Decode0
        out0_de = self.block0_de(x)
        out0_de = self.res_block0_de(out0_de)

        print("the shape of the point cloud after the first upsamle is :",out0_de.F.shape)
        # 此处的out0_cls 要进行一定的变换，比如取出第一列的特征进行判断
        out0_cls = self.block0_cls(out0_de)

        target0 = self.get_target_by_sp_tensor(out0_de,target_label[0])
        targets.append(target0)
        out_cls.append(out0_cls)
        out_geo.append(out0_de)

        # 这个地方要注意是取得第一列的占用信息来进行pruning
        keep0 = (out0_cls.F[:,[1]] > 0).cpu().squeeze()
        keeps.append(keep0)

        if training:
            keep0 += target0

        # 进行pruning
        out0_pruned = self.pruning(out0_de,keep0.to(out0_de.device))

        # Block1 Decode1
        out1_de = self.block1_de(out0_pruned)
        out1_de = self.res_block1_de(out1_de)
        print("the shape of the point cloud after the second upsamle is :",out1_de.F.shape)

        out1_cls = self.block1_cls(out1_de)
        target1 = self.get_target_by_sp_tensor(out1_de,target_label[1])

        targets.append(target1)
        out_cls.append(out1_cls)
        out_geo.append(out1_de)

        keep1 = (out1_cls.F[:,[1]] > 0).cpu().squeeze()
        keeps.append(keep1)

        if training:
            keep1 += target1

        # 进行pruning
        out1_pruned = self.pruning(out1_de,keep1.to(out1_de.device))

        # Block2 Decode1
        out2_de = self.block2_de(out1_pruned)
        out2_de = self.res_block2_de(out2_de)

        print("the shape of the point cloud after the third upsamle is :",out2_de.F.shape)

        out2_cls = self.block2_cls(out2_de)

        target2 = self.get_target_by_sp_tensor(out2_de,target_label[2])
        targets.append(target2)
        out_cls.append(out2_cls)
        out_geo.append(out2_de)

        keep2 = (out2_cls.F[:,[1]]> 0).cpu().squeeze()
        # keeps.append(keep2)

        if training:
            keep2 += target2

        # 进行pruning
        out2_pruned = self.pruning(out2_de,keep2.to(out2_de.device))


        out_final = self.conv3_de(out2_pruned)

        return  out_final,out_cls,out_geo,targets,keeps

    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))

        return nn.Sequential(*layers)

if __name__ == '__main__':
    encoder = Encoder(channels=16,block_layers=3,block='InceptionResNet')
    print(encoder)
    decoder = Decoder(channels=16,block_layers=3,block='InceptionResNet')
    print(decoder)


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
        nn.Module.__init__(self)
        in_nchannels = 4
        ch = [16, 32, 64, 32, channels]
        if block == 'ResNet':
            self.block = ResNet
        elif block == 'InceptionResNet':
            self.block = InceptionResNet

        self.block_test_en = nn.Sequential(
            ME.MinkowskiConvolution(
                4,8,kernel_size = 2,stride=2,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(8),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                8,8,kernel_size=3,stride=1,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(8),
            ME.MinkowskiELU(),)

        self.res_test_en = self.make_layer(
            self.block, block_layers, 8)

        self.block_test_en1 = nn.Sequential(
            ME.MinkowskiConvolution(
                8,16,kernel_size = 2,stride=2,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                16,16,kernel_size=3,stride=1,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiELU(),)

        self.res_test_en1 = self.make_layer(
            self.block, block_layers, 16)

        self.block_test_de = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                16,8,kernel_size = 2,stride=2,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(8),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                8,8,kernel_size = 3,stride=1,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(8),
            ME.MinkowskiELU(),)

        self.res_test_de = self.make_layer(
            self.block, block_layers, 8)


        self.block_test_de1 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                8,4,kernel_size = 2,stride=2,bias=True,dimension = 3,),
            ME.MinkowskiBatchNorm(4),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                4,4,kernel_size=3,stride=1,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(4),
            ME.MinkowskiELU(),
        )

        self.res_test_de1 = self.make_layer(
            self.block, block_layers, 4)

        self.block0_en = nn.Sequential(
            ME.MinkowskiConvolution(
                in_nchannels,ch[0],kernel_size=2,stride=2,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                ch[0],ch[1],kernel_size=3,stride=1,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )
        # tensor 的shape
        self.res_block0_en = self.make_layer(
            self.block,block_layers,ch[1])

        self.block1_en = nn.Sequential(
            ME.MinkowskiConvolution(
                ch[1], ch[2], kernel_size=2, stride=2, bias=True, dimension=3, ),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                ch[2],ch[2],kernel_size=3,stride=1,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )
        self.res_block1_en = self.make_layer(
            self.block,block_layers,ch[2])

        self.block2_en = nn.Sequential(
            ME.MinkowskiConvolution(
                # ch[2] = 64, ch[3] = 32
                ch[2],ch[3],kernel_size=2,stride=2,bias=True,dimension=3,),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                ch[3], ch[3], kernel_size=3, stride=1, bias=True, dimension=3, ),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )
        self.res_block2_en = self.make_layer(
            self.block,block_layers,ch[3])

        self.conv3_en = ME.MinkowskiConvolution(
            ch[3],ch[4],kernel_size=3,stride=1,bias=True,dimension=3)

        ######################################################
        self.pre_deconv = ME.MinkowskiConvolutionTranspose(
            ch[0], ch[1], kernel_size=3, stride=1, bias=True, dimension=3, )

        self.block0_de = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                # ch[1]=32, ch[2]=64
                ch[1], ch[2], kernel_size=2, stride=2, bias=True, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                ch[2], ch[2], kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )
        self.res_block0_de = self.make_layer(
            self.block, block_layers, ch[2])
        self.block0_cls = ME.MinkowskiConvolution(
            ch[2], 4, kernel_size=3, stride=1, bias=True, dimension=3,
        )


    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("原始数据 :",x)
        # print("test原始数据格式为 :",x.F.shape)
        # data_test = self.block_test_en(x)
        # data_test = self.res_test_en(data_test)
        # # print("第一次压缩之后的数据为 :",data_test.F)
        # print("test第一次压缩之后的数据格式为 :",data_test.F.shape)
        # data_test_1 = self.block_test_en1(data_test)
        # data_test_1 = self.res_test_en1(data_test_1)
        # # print("第二次压缩之后的数据为 :",data_test_1.F)
        # print("test第二次压缩之后的数据格式为 :",data_test_1.F.shape)
        # data_test_de = self.block_test_de(data_test_1)
        # data_test_de = self.res_test_de(data_test_de)
        # # print("第一次解压缩之后的数据为 :",data_test_de.F)
        # print("test第一次解压缩之后的数据格式为 :",data_test_de.F.shape)
        # data_test_de_1 = self.block_test_de1(data_test_de)
        # data_test_de_1=self.res_test_de1(data_test_de_1)
        # # print("第二次解压缩之后的数据为 ：",data_test_de_1.F)
        # print("test第二次解压缩之后的数据格式为 :",data_test_de_1.F.shape)

        out0_en = self.block0_en(x)
        out0_en = self.res_block0_en(out0_en)
        out1_en = self.block1_en(out0_en)
        out1_en = self.res_block1_en(out1_en)
        out2_en = self.block2_en(out1_en)
        out2_en = self.res_block2_en(out2_en)
        out2_en = self.conv3_en(out2_en)

        # print("********************************解码测试********************************")
        # test_de = self.pre_deconv(out2_en)
        # print("初始情况",test_de.F.shape)
        # test_de_1 = self.block0_de(test_de)
        # test_de_11 = self.res_block0_de(test_de_1)
        # print("模拟解码一次之后的数据格式",test_de_11.F.shape)
        # print("模拟解码之后的数据为：",test_de_11.F)
        # print("*********************************测试结束××××××××××××××××××××××××××××××××")

        return [out2_en, out1_en, out0_en]

class Decoder(nn.Module):
    """
    decoder test
    """
    def __init__(self,channels,block_layers,block):
        nn.Module.__init__(self)
        out_nchannel = 4
        ch = [channels,32,64,32,16]
        if block == 'ResNet':
            self.block = ResNet
        elif block == 'InceptionResNet':
            self.block = InceptionResNet

        self.pre_deconv = ME.MinkowskiConvolutionTranspose(
            ch[0],ch[1],kernel_size = 3,stride = 1,bias = True,dimension = 3,)

        self.block0_de = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                # ch[1]=32, ch[2]=64
                ch[1],ch[2],kernel_size = 2,stride = 2,bias=True,dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                ch[2],ch[2],kernel_size = 3,stride = 1,bias = True,dimension = 3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )
        self.res_block0_de = self.make_layer(
            self.block,block_layers,ch[2])
        self.block0_cls = ME.MinkowskiConvolution(
            ch[2],out_nchannel,kernel_size = 3,stride = 1,bias = True,dimension = 3,
        )

        self.block1_de = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[2],ch[3],kernel_size = 2,stride = 2,bias = True,dimension = 3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                ch[3],ch[3],kernel_size = 3,stride = 1,bias = True,dimension = 3,),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )
        self.res_block1_de = self.make_layer(
            self.block,block_layers,ch[3])
        self.block1_cls = ME.MinkowskiConvolution(
            ch[3],out_nchannel,kernel_size = 3,stride = 1,bias = True,dimension = 3)

        self.block2_de = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[3],ch[4],kernel_size = 2,stride = 2,bias = True,dimension = 3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                ch[4],ch[4],kernel_size = 3,stride = 1,bias = True,dimension = 3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
        )
        self.res_block2_de = self.make_layer(
            self.block,block_layers,ch[4]
        )
        self.block2_cls = ME.MinkowskiConvolution(
            ch[4],out_nchannel,kernel_size = 3,stride = 1,bias = True,dimension = 3)

        # 最后一次卷积，卷成初始的四维坐标
        self.conv3_de = ME.MinkowskiConvolutionTranspose(
            ch[4],out_nchannel,kernel_size = 3,stride = 1,bias = True,dimension = 3)

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

        return torch.Tensor(target).bool()

    def forward(self, x, target_label, adaptive, rhos=[1.0, 1.0, 1.0], training=True):
        targets = []
        out_cls = []
        # return the geo-feats info of the output tensor
        out_geo = []
        keeps = []

        # print("解码器端的初始数据格式 ：",x.F.shape)

        x = self.pre_deconv(x)
        # print("预处理之后的数据格式是：",x.F.shape)
        # Block0  Decode0
        out0_de = self.block0_de(x)
        # print("第一次解码之后的数据格式 :",out0_de.F.shape)
        out0_de = self.res_block0_de(out0_de)

        # 此处的out0_cls 要进行一定的变换，比如取出第一列的特征进行判断
        out0_cls = self.block0_cls(out0_de)

        target0 = self.get_target_by_sp_tensor(out0_de,target_label[0])
        targets.append(target0)
        out_cls.append(out0_cls)


        #print("get first feature :",out0_cls.F[:,[1]])
        # 这个地方要注意是取得第一列的占用信息来进行pruning
        keep0 = (out0_cls.F[:,[1]] > 0).cpu().squeeze()
        keeps.append(keep0)

        if training:
            keep0 += target0

        # 进行pruning
        out0_pruned = self.pruning(out0_de,keep0.to(out0_de.device))
        out_geo.append(out0_pruned)

        # Block1 Decode1
        out1_de = self.block1_de(out0_pruned)
        out1_de = self.res_block1_de(out1_de)

        # print("两次解压缩之后的点云数据格式为 ：",out1_de.F.shape)

        out1_cls = self.block1_cls(out1_de)
        target1 = self.get_target_by_sp_tensor(out1_de,target_label[1])

        targets.append(target1)
        out_cls.append(out1_cls)

        keep1 = (out1_cls.F[:,[1]] > 0).cpu().squeeze()
        keeps.append(keep1)

        if training:
            keep1 += target1

        # 进行pruning
        out1_pruned = self.pruning(out1_de,keep1.to(out1_de.device))
        out_geo.append(out1_pruned)

        # Block2 Decode1
        out2_de = self.block2_de(out1_pruned)
        out2_de = self.res_block2_de(out2_de)

        # print("三次解压缩之后的点云数据格式是：", out2_de.F.shape)

        out2_cls = self.block2_cls(out2_de)

        target2 = self.get_target_by_sp_tensor(out2_de,target_label[2])
        targets.append(target2)
        out_cls.append(out2_cls)

        keep2 = (out2_cls.F[:,[1]]> 0).cpu().squeeze()
        keeps.append(keep2)

        if training:
            keep2 += target2

        # 进行pruning
        out2_pruned = self.pruning(out2_de,keep2.to(out2_de.device))
        out_final = self.conv3_de(out2_pruned)

        out_geo.append(out_final)

        return out_final,out_cls,out_geo,targets,keeps

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


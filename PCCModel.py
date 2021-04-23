# 2020.02.13 compression network
import sys
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

from models.exp_model import Encoder, Decoder
from models.EntropyModel import EntropyBottleneck


class PCC(nn.Module):
	"""
	Encoder
	"""

	def __init__(self, channels=16):
		nn.Module.__init__(self)
		# self.nchannels=channels
		# self.encoder = Encoder(channels=channels)
		# self.decoder = Decoder(channels=channels)

		self.encoder = Encoder(channels=channels, block_layers=3, block='InceptionResNet')
		self.decoder = Decoder(channels=channels, block_layers=3, block='InceptionResNet')
		self.entropy_bottleneck = EntropyBottleneck(channels)

	def forward(self, x, target_format, adaptive, training, device):
		ys = self.encoder(x)
		y = ME.SparseTensor(ys[0].F, coordinates=ys[0].C, tensor_stride=8, device=device)
		# add noise to feature for quantizition
		feats_tilde, likelihood = self.entropy_bottleneck(y.F, training, device)
		# 此处声明SparseTensor时没有用coordinates只用的features和coordinate_map_key
		y_tilde = ME.SparseTensor(feats_tilde, coordinate_map_key=y.coordinate_map_key, coordinate_manager=y.coordinate_manager, device=device)

		cm = y_tilde.coordinate_manager

		# TODO from v0.4 to v0.5
		target_map_key = cm.insert_and_map(
			x.C,
			tensor_stride=1)

		# 此处ys[1:]+[x] ys中一共包含3个元素，ys[1:]意味从第二个元素开始取，ys[1:]+[x]的意思是将[x]作为
		# list中的第三个元素放到target_label中，这样target_label中其实包含三个元素，前两个是压缩之后的，最后一个是原始的x

		out, out_cls,out_geo, targets, keeps = self.decoder(y_tilde, target_label=ys[1:]+[x], adaptive=adaptive, training=training)
		comp_exp = ys[1:] + [x]

		#print("the len of the keeps is :",len(keeps))
		#print("the len of the targets is :",len(targets))

		return ys, likelihood, out, out_cls, out_geo, targets, comp_exp, keeps


if __name__ == '__main__':
	pcc = PCC(channels=16)
	print(pcc)


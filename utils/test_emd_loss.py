import torch
import numpy as np
import time
from utils.emd import  earth_mover_distance
from utils.chamfer_distance import ChamferDistance

# gt
p1 = torch.from_numpy(np.array([[[1.7, -0.1, 0.1], [0.1, 1.2, 0.3]]], dtype=np.float32)).cuda()
p1 = p1.repeat(3, 1, 1)
p2 = torch.from_numpy(np.array([[[0.3, 1.8, 0.2], [1.2, -0.2, 0.3]]], dtype=np.float32)).cuda()
p2 = p2.repeat(3, 1, 1)
print(p1)
print(p2)
p1.requires_grad = True
p2.requires_grad = True

gt_dist = (((p1[0, 0] - p2[0, 1])**2).sum() + ((p1[0, 1] - p2[0, 0])**2).sum()) / 2 +  \
         (((p1[1, 0] - p2[1, 1])**2).sum() + ((p1[1, 1] - p2[1, 0])**2).sum()) * 2 + \
         (((p1[2, 0] - p2[2, 1])**2).sum() + ((p1[2, 1] - p2[2, 0])**2).sum()) / 3
print('gt_dist: ', gt_dist)

gt_dist.backward()
print(p1.grad)
print(p2.grad)

# cd
p1 = torch.from_numpy(np.array([[[1.7, -0.1, 0.1], [0.1, 1.2, 0.3]]], dtype=np.float32)).cuda()
#p1 = p1.repeat(3, 1, 1)
p2 = torch.from_numpy(np.array([[[0.3, 1.8, 0.2], [1.2, -0.2, 0.3]]], dtype=np.float32)).cuda()
#p2 = p2.repeat(3, 1, 1)

print("the p1 is :",p1)
print("the p2 is :",p2)
p1.requires_grad = True
p2.requires_grad = True

chamfer_dist = ChamferDistance()

#...
# points and points_reconstructed are n_points x 3 matrices

dist1, dist2 = chamfer_dist(p1, p2)
loss = (torch.mean(dist1)) + (torch.mean(dist2))


loss1 = earth_mover_distance(p1[0], p2[0], transpose=False)
print("the cd distance is ",loss.item())
print("the emd loss is :",loss1.item())
# loss = d[0] / 2 + d[1] * 2 + d[2] / 3
# print("the loss is :",loss)

loss.backward()
points1 = torch.rand(32, 1000, 3).cuda()

print("the points1 is :",points1)
print("the p1 grad is :",p1.grad)
print("the p2 grad is :",p2.grad)


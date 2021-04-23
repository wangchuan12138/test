# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Zhan Ma (mazhan@nju.edu.cn); Nanjing University, Vision Lab.
# Last update: 2020.06.06
import os
import open3d as o3d
import time, os, sys, glob, argparse
import numpy as np
import torch
import MinkowskiEngine as ME

from sklearn.preprocessing import StandardScaler

import importlib
from dataprocess.data_loader import PCDataset, make_data_loader
from utils.loss import get_metrics
from utils.loss import get_emd_loss
from utils.loss import get_cd_loss
from utils.pc_error_wrapper import pc_error

# from myutils.logger import Logger
from tensorboardX import SummaryWriter
import logging

def getlogger(logdir):
  logger = logging.getLogger(__name__)
  logger.setLevel(level = logging.INFO)

  handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
  handler.setFormatter(formatter)

  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  console.setFormatter(formatter)

  logger.addHandler(handler)
  logger.addHandler(console)

  return logger


def parse_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--model", default="PCCModel")
  parser.add_argument("--channels", type=int, default=16)

  parser.add_argument("--dataset", default='/home/zhanggai/PCGC/PCGCv2/training_dataset/')
  parser.add_argument("--num_test", type=int, default=1024)
  parser.add_argument("--dataset_8i", default='testdata/8iVFB/')

  parser.add_argument("--alpha", type=float, default=20., help="weights for distoration.")

  parser.add_argument("--init_ckpt", default='ckpts/tp/tp74000.pth')
  parser.add_argument("--reset", default=False, action='store_true', help='reset training')

  parser.add_argument("--lr", type=float, default=8e-4)
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--global_step", type=int, default=int(164600))
  parser.add_argument("--base_step", type=int, default=int(100),  help='frequency for recording state.')
  parser.add_argument("--test_step", type=int, default=int(2000),  help='frequency for test and save.')
  # parser.add_argument("--random_seed", type=int, default=4, help='random_seed.')

  parser.add_argument("--max_norm", type=float, default=1.,  help='max norm for gradient clip, close if 0')
  parser.add_argument("--clip_value", type=float, default=0,  help='max value for gradient clip, close if 0')

  parser.add_argument("--adaptive", default=True, action='store_true', help='using adaptive threshold')
  parser.add_argument("--target_format", type=str, default='sp_tensor', help='key or sp_tensor')
  
  parser.add_argument("--logdir", type=str, default='logs', help="logger direction.")
  parser.add_argument("--ckptdir", type=str, default='ckpts', help="ckpts direction.")
  parser.add_argument("--prefix", type=str, default='tp', help="prefix of checkpoints/logger, etc.")
  parser.add_argument("--lr_gamma", type=float, default=0.5, help="gamma for lr_scheduler.")
  parser.add_argument("--lr_step", type=int, default=6e3, help="step for adjusting lr_scheduler.")  


  args = parser.parse_args()
  return args


def test(pcc, test_dataloader, logger, writer, writername, step, test_pc_error, args, device):

  start_time = time.time()

  # data.
  test_iter = iter(test_dataloader)

  # loss & metrics.
  all_loss = 0.
  all_bpp = 0.
  all_sum_loss = 0.
  all_losses = np.zeros(3)
  all_metrics = np.zeros((3,3))
  all_pc_errors = np.zeros(3)

  # model & crit.
  pcc.to(device)# to cpu.
  crit = torch.nn.BCEWithLogitsLoss()

  # loop per batch.
  for i in range(len(test_iter)):
    coords, feats = test_iter.next()
    x = ME.SparseTensor(feats, coordinates=coords, device=device)
    
    # Forward.
    _, likelihood, out, out_cls, out_geo, targets, comp_exp_encoder, keeps = pcc(x,
                                                    target_format=args.target_format, 
                                                    adaptive=args.adaptive, 
                                                    training=False, 
                                                    device=device)

    # get loss.
    loss = 0
    loss_G = 0
    losses_G = []
    losses = []
    num_losses = len(out_cls)
    num_losses_G = len()
    # 三次下采样逐层计算loss，最后求和
    # 此处计算loss时对占有率和几何features信息分别计算最后加和
    for out_cl, target in zip(out_cls, targets):
      curr_loss = crit(out_cl.F[:,[1]].squeeze(),
                       target.type(out_cl.F.dtype).to(device))
      losses.append(curr_loss.item())
      loss += curr_loss / float(num_losses)


    bpp = torch.sum(torch.log(likelihood)) / -(torch.log(torch.Tensor([2.0]).to(device))) / float(x.__len__())

    sum_loss = args.alpha * loss + 1. * bpp

    # get metrics.
    metrics = []
    for keep, target in zip(keeps, targets):
      curr_metric = get_metrics(keep, target.bool())
      metrics.append(curr_metric)

    # get pc_error.
    if test_pc_error:
      ori_pcd = o3d.geometry.PointCloud()
      ori_pcd.points = o3d.utility.Vector3dVector(x.decomposed_coordinates[0])
      # ori_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=20))
      
      orifile = args.prefix+'ori.ply'
      o3d.io.write_point_cloud(orifile, ori_pcd, write_ascii=True)

      rec_pcd = o3d.geometry.PointCloud()
      rec_pcd.points = o3d.utility.Vector3dVector(out.decomposed_coordinates[0])

      recfile = args.prefix+'rec.ply'
      o3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)

      pc_error_metrics = pc_error(infile1=orifile, infile2=recfile, res=1024)
      pc_errors = [pc_error_metrics['mse1,PSNR (p2point)'][0], 
                  pc_error_metrics['mse2,PSNR (p2point)'][0], 
                  pc_error_metrics['mseF,PSNR (p2point)'][0]]

    # record.
    with torch.no_grad():
      all_loss += loss
      all_bpp += bpp 
      all_sum_loss += sum_loss
      all_losses += np.array(losses)
      all_metrics += np.array(metrics)
      if test_pc_error:
        all_pc_errors += np.array(pc_errors)

  print('======testing time:', round(time.time() - start_time, 4), 's')

  all_loss /= len(test_iter)
  all_bpp /= len(test_iter)
  all_sum_loss /= len(test_iter)
  all_losses /= len(test_iter)
  all_metrics /= len(test_iter)
  if test_pc_error:
    all_pc_errors /= len(test_iter)


  # logger.
  logger.info(f'\nIteration: {step}')
  logger.info(f'Loss: {all_loss.item():.4f}')
  logger.info(f'bpp: {all_bpp.item():.4f}')
  logger.info(f'Sum Loss: {all_sum_loss.item():.4f}')
  logger.info(f'Loss (s-m-l): {np.round(all_losses, 4).tolist()}')
  logger.info(f'Metrics (s-m-l): {np.round(all_metrics, 4).tolist()}')
  if test_pc_error:
    logger.info(f'all_pc_errors: {np.round(all_pc_errors, 4).tolist()}')

  # writer.
  writer.add_scalars(main_tag=writername+'/losses', 
                    tag_scalar_dict={'loss' :all_loss.item(), 
                                     'bpp' : all_bpp.item(),
                                     'sum_loss': all_sum_loss.item()}, 
                    global_step=step)

  writer.add_scalars(main_tag=writername+'/losses/details', 
                    tag_scalar_dict={'loss1' :all_losses[0], 
                                     'loss2' : all_losses[1],
                                     'loss3': all_losses[2]}, 
                    global_step=step)

  writer.add_scalars(main_tag=writername+'/metrics', 
                    tag_scalar_dict={'IoU1': all_metrics[0,2], 
                                     'IoU2': all_metrics[1,2], 
                                     'IoU3': all_metrics[2,2]}, 
                    global_step=step)
  if test_pc_error:
    writer.add_scalars(main_tag=writername+'/pc_errors', 
                      tag_scalar_dict={'p2point1': all_pc_errors[0],
                                        'p2point2': all_pc_errors[1],
                                        'p2pointF': all_pc_errors[2],},
                      global_step=step)
 
  # return all_losses, all_bpp, all_sum_loss, all_losses, all_metrics
  return


def train(pcc, train_dataloader, test_dataloader, test_dataloader2, logger, writer, args, device):
  # Optimizer.
  optimizer = torch.optim.Adam([{"params":pcc.encoder.parameters(), 'lr':args.lr},
                                {"params":pcc.decoder.parameters(), 'lr':args.lr},
                                {"params":pcc.entropy_bottleneck.parameters(), 'lr':args.lr}], 
                                betas=(0.9, 0.999), weight_decay=1e-4)
  # adjust lr.
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
  # criterion.
  crit = torch.nn.BCEWithLogitsLoss()

  # define checkpoints direction.
  ckptdir = os.path.join(args.ckptdir, args.prefix)
  if not os.path.exists(ckptdir):
    logger.info(f'Make direction for saving checkpoints: {ckptdir}')
    os.makedirs(ckptdir)

  # Load checkpoints.
  start_step = 1
  if args.init_ckpt == '':
    logger.info(f'Random initialization.')
  else:
    # load params from checkpoints.
    logger.info(f'Load checkpoint from {args.init_ckpt}')
    ckpt = torch.load(args.init_ckpt)
    pcc.encoder.load_state_dict(ckpt['encoder'])
    pcc.decoder.load_state_dict(ckpt['decoder'])
    pcc.entropy_bottleneck.load_state_dict(ckpt['entropy_bottleneck'])
    # load start step & optimizer.
    if not args.reset:
      # optimizer.load_state_dict(ckpt['optimizer'])
      start_step = ckpt['step'] + 1

  # start step.
  logger.info(f'LR: {scheduler.get_lr()}')
  print('==============', start_step)
  # 获取数据
  train_iter = iter(train_dataloader)

  start_time = time.time()

  #
  all_loss_R = 0.
  all_loss_G = 0.
  all_bpp = 0.
  all_sum_loss = 0.

  all_losses = np.zeros(3)
  all_metrics = np.zeros((3,3))
  # 32000个的训练
  for i in range(start_step, args.global_step+1):

    print("*******************************************iteration " +str(i)+"***************************************")

    optimizer.zero_grad()

    s = time.time()
    coords, feats = train_iter.next()
    dataloader_time = time.time() - s# 

    x = ME.SparseTensor(features=feats, coordinates=coords, device=device)

    
    # if x.__len__() >= 4e5:
    #   logger.info(f'\n\n\n======= larger than 4e5 ======: {x.__len__()}\n\n\n')
    #   continue

    # Forward.
    # 此处要重新获得特征信息中的x，y，z信息
    # 此处做坐标数据的标准化过程
    ori_coords = x.features[:,1:]
    print("原始点云坐标是 :",ori_coords)
    ss = StandardScaler()
    feats_ss = ss.fit_transform(x.features[:,1:].cpu())
    x.features[:,1:] = torch.tensor(feats_ss)
    _, likelihood, _, out_cls, out_geo, targets, comp_exps, keeps = pcc(x,
                                                    target_format=args.target_format, 
                                                    adaptive=args.adaptive, 
                                                    training=True, 
                                                    device=device)


    # 坐标恢复
    rec_coords = ss.inverse_transform(out_geo[2].features[:,1:].cpu().detach().numpy())
    rec_coords = np.floor(rec_coords)
    rec_coords = torch.from_numpy(rec_coords).cuda()
    print("重建后的点云坐标是：",rec_coords)

    ori_coords = torch.unsqueeze(torch.tensor(ori_coords),1)
    rec_coords = torch.unsqueeze(torch.tensor(rec_coords),1)

    cd_G = get_cd_loss(ori_coords, rec_coords)
    cd_G = cd_G.item()
    print("the cd distance between the ori and process is :",cd_G)




    loss_R = 0.
    loss_G = 0.
    losses_R = []
    num_losses_R = len(out_cls)


    for out_cl, target in zip(out_cls, targets):

      # out_cl.F.dtype 返回输出数据的特征的类型, out_cl.F.dtype的类型是torch.float32
      # target.type is torch.BoolTensor
      # .squeeze()用于去除维度为1的目标[[1,2,3]]-->[1,2,3]
      # .unsqueeze()用于增加目标的维度 [1,2,3]-->[[1,2,3]]

      curr_loss_R = crit(out_cl.F[:,[1]].squeeze(),
                        target.type(out_cl.F.dtype).to(device))

      losses_R.append(curr_loss_R.item())
      loss_R += curr_loss_R / float(num_losses_R)

    # 计算feature中x,y,z 的loss
    # F[:,1]取第一列元素
    ori = comp_exps[2].F[:,1:]
    ori = torch.unsqueeze(ori,1)
    aft = out_geo[2].F[:,1:]
    aft = torch.unsqueeze(aft,1)
    # print("计算loss时的压缩后点云数据结构是：",ori)
    # print("计算loss时的原始点云数据结构是：",aft)
    # print("点数为：",len(out_geo[2].F[:,1:]))
    curr_loss_G = get_cd_loss(ori,aft)
    loss_G = curr_loss_G.item()


    # for out_ge,comp_exp in zip(out_geo,comp_exps):
    #   # print("the len of the out_ge is :",len(out_ge.F))
    #   # print("the len of the comp_exp is :",len(comp_exp.F))
    #
    #   # problem: 点数对不起来
    #   curr_loss_G = crit(out_ge.F,comp_exp.F)
    #
    #   losses_G.append(curr_loss_G.item())
    #   loss_G += curr_loss_G / float(num_losses_G)

    metrics = []
    for keep, target in zip(keeps, targets):
      # print("the keep is :",keep)
      # print("the target is :",target)
      curr_metric = get_metrics(keep, target.bool())
      # print("the curr_metric is :",curr_metric)
      metrics.append(curr_metric)

    bpp = torch.sum(torch.log(likelihood)) / -(torch.log(torch.Tensor([2.0]).to(device))) / float(x.__len__())

    sum_loss = args.alpha * loss_R + 1. * bpp + loss_G
    if i % 1 == 0:
      print("输出的坐标的数据是：",out_geo[2].F[:,1:])
      print("计算对比的几何坐标的数据是:",comp_exps[2].F[:,1:])
      print("the curr loss_R is",loss_R.item())
      print("the curr_loss_G is :", curr_loss_G.item())
      print("the sum_loss is :",sum_loss.item())

    # Backward.
    sum_loss.backward()

    # # reconstruct the point cloud
    # ss = StandardScaler()
    # re_pc = ss.inverse_transform(out_geo[2].F[:,1:])
    # print("the reconstruct point cloud is :",re_pc)

    # Optional clip gradient. 
    # if args.max_norm != 0:
    #   # clip by norm
    #   max_grad_before = max(p.grad.data.abs().max() for p in pcc.parameters())
    #   total_norm = torch.nn.utils.clip_grad_norm_(pcc.parameters(), args.max_norm)
    #
    #   if total_norm > args.max_norm:
    #
    #     def get_total_norm(parameters, norm_type=2):
    #       total_norm = 0.
    #       for p in parameters:
    #         param_norm = p.grad.data.norm(norm_type)
    #         total_norm += param_norm.item() ** norm_type
    #       total_norm = total_norm ** (1. / norm_type)
    #       return total_norm
    #
    #     print('total_norm:',
    #       '\nBefore: total_norm:,', total_norm,
    #       'max grad:', max_grad_before,
    #       '\nthreshold:', args.max_norm,
    #       '\nAfter:', get_total_norm(pcc.parameters()),
    #       'max grad:', max(p.grad.data.abs().max() for p in pcc.parameters()))
    #
    # if args.clip_value != 0:
    #   torch.nn.utils.clip_grad_value_(pcc.parameters(), args.clip_value)
    #   print('after gradient clip',  max(p.grad.data.abs().max() for p in pcc.parameters()))

    optimizer.step()

    # record.
    with torch.no_grad():
      all_bpp += bpp 
      all_loss_R = all_loss_R + loss_R
      all_loss_G = all_loss_G + loss_G
      all_sum_loss += sum_loss

      # 未知作用
      all_losses = all_losses + np.array(losses_R)
      all_metrics += np.array(metrics)

    # Display.
    if i % args.base_step == 0:
      # average.
      with torch.no_grad():
        all_bpp /= args.base_step
        all_loss_R /= args.base_step
        all_loss_G /= args.base_step
        all_sum_loss /= args.base_step

        all_losses /= args.base_step
        all_metrics /= args.base_step

      if np.isinf(all_loss_G):
        logger.info('inf error!')
        sys.exit(0)

      # logger.
      logger.info(f'\nIteration: {i}')
      logger.info(f'Running time: {((time.time()-start_time)/60):.2f} min')
      logger.info(f'Data Loading time: {dataloader_time:.5f} s')
      logger.info(f'Loss R: {all_loss_R.item():.4f}')
      logger.info(f'Loss G: {all_loss_G:.4f}')
      logger.info(f'bpp: {all_bpp.item():.4f}')
      logger.info(f'Sum Loss: {all_sum_loss.item():.4f}')

      logger.info(f'Loss (s-m-l): {np.round(all_losses, 4).tolist()}')
      logger.info(f'Metrics (s-m-l): {np.round(all_metrics, 4).tolist()}')


      # tensorboard writer.
      # parameters instructions
      # main_tag means the name of the logout results
      # tag_scalar_dict means the specific values of the y axis
      # global_step means the specific valus of the x axis
      writer.add_scalars(main_tag='train/losses', 
                        tag_scalar_dict={'loss_R' :all_loss_R.item(),
                                         'loss_G' :all_loss_G,
                                         'bpp' : all_bpp.item(),
                                         'sum_loss': all_sum_loss.item()},
                        global_step=i)
      # 分层输出对占有率的损失计算
      writer.add_scalars(main_tag='train/losses/details', 
                        tag_scalar_dict={'loss1' :all_losses[0], 
                                         'loss2' : all_losses[1],
                                         'loss3': all_losses[2]}, 
                        global_step=i)
      # 输出评价标准
      writer.add_scalars(main_tag='train/metrics', 
                        tag_scalar_dict={'IoU1': all_metrics[0,2], 
                                         'IoU2': all_metrics[1,2], 
                                         'IoU3': all_metrics[2,2]}, 
                        global_step=i)


      # return 0.
      all_bpp = 0.
      all_loss_G = 0.
      all_loss_R = 0.
      all_sum_loss = 0.
      all_losses = np.zeros(3)
      all_metrics = np.zeros((3,3))

      # empty cache.
      torch.cuda.empty_cache()

    if i % args.test_step == 0:
      logger.info(f'\n==========Evaluation: iter {i}==========')
      # save.
      logger.info(f'save checkpoints: {os.path.join(ckptdir, args.prefix + str(i))}')
      torch.save({'step': i,
                  'encoder': pcc.encoder.state_dict(), 
                  'decoder': pcc.decoder.state_dict(),
                  'entropy_bottleneck': pcc.entropy_bottleneck.state_dict()
                  }, os.path.join(ckptdir, args.prefix + str(i) + '.pth'))

      # Evaluation.
      # logger.info(f'\n=====Evaluation: iter {i} =====')
      # with torch.no_grad():
      #   test(pcc=pcc, test_dataloader=test_dataloader,
      #     logger=logger, writer=writer, writername='eval', step=i, test_pc_error=False, args=args, device=device)

      torch.cuda.empty_cache()

    if i % (args.test_step*2) == 0:
      # Evaluation 8i.
      # logger.info(f'\n=====Evaluation: iter {i} 8i =====')
      # with torch.no_grad():
      #   test(pcc=pcc, test_dataloader=test_dataloader2,
      #     logger=logger, writer=writer, writername='eval_8i', step=i, test_pc_error=True, args=args, device='cpu')

      torch.cuda.empty_cache()

      pcc.to(device)

    if i % int(args.lr_step) == 0:
      scheduler.step()
      logger.info(f'LR: {scheduler.get_lr()}')
      

if __name__ == '__main__':
  args = parse_args()

  logdir = os.path.join(args.logdir, args.prefix)
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  logger = getlogger(logdir)
  logger.info(args)
  writer = SummaryWriter(log_dir=logdir)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info(f'Device:{device}')

  # Load data.
  filedirs = glob.glob(args.dataset+'*.h5')
  filedirs = sorted(filedirs)
  logger.info(f'Files length: {len(filedirs)}')

  # 加载训练用的数据集
  train_dataset = PCDataset(filedirs[int(args.num_test):], feature_format='geometry')
  train_dataloader = make_data_loader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      shuffle=True, 
                                      num_workers=1,
                                      repeat=True)
  
  test_dataset = PCDataset(filedirs[:int(args.num_test)], feature_format='geometry')
  test_dataloader = make_data_loader(dataset=test_dataset, 
                                      batch_size=args.batch_size, 
                                      shuffle=False, 
                                      num_workers=1,
                                      repeat=False)

  # 8i dataset
  eighti_filedirs = glob.glob(args.dataset_8i+'*.ply')
  eighti_filedirs = sorted(eighti_filedirs)
  logger.info(f'Files length: {len(eighti_filedirs)}')

  eighti_dataset = PCDataset(eighti_filedirs, feature_format='geometry')
  eighti_dataloader = make_data_loader(dataset=eighti_dataset, 
                                        batch_size=1, 
                                        shuffle=False, 
                                        num_workers=1,
                                        repeat=False)

  # Network.
  model = importlib.import_module(args.model)
  pcc = model.PCC(channels=args.channels).to(device)
  logger.info(pcc)

  train(pcc, train_dataloader, test_dataloader, eighti_dataloader, logger, writer, args, device)



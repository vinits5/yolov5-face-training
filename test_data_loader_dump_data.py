import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from threading import Thread
from warnings import warn

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.face_datasets_with_face_crop import create_dataloader, LoadFaceImagesAndLabels
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    print_mutation, set_logging
from utils.google_utils import attempt_download
from utils.loss import compute_loss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first

def showlabels(img, targets, idx, output_dir):
    boxs = targets[:, 2:6]
    landmarks = targets[:, 6:]
    import cv2
    import matplotlib.pyplot as plt
    img = img.permute(1,2,0).numpy()
    plt.imsave('img.png', img)
    img = cv2.imread('img.png')
    boxs = boxs.numpy()
    landmarks = landmarks.numpy()
    file = open(f'{output_dir}/image_{idx}_bbox.txt', 'w')
    for box in boxs:
        x,y,w,h = box[0] * img.shape[1], box[1] * img.shape[0], box[2] * img.shape[1], box[3] * img.shape[0]
        file.write(str(int(x - w/2))+' '+str(int(y - h/2))+' '+str(int(x + w/2))+' '+str(int(y + h/2))+'\n')
        #cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.rectangle(img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
    file.close()

    file = open(f'{output_dir}/image_{idx}_lmark.txt','w')
    for landmark in landmarks:
        #cv2.circle(img,(60,60),30,(0,0,255))
        lm_txt = ''
        for i in range(5):
            cv2.circle(img, (int(landmark[2*i] * img.shape[1]), int(landmark[2*i+1]*img.shape[0])), 3 ,(0,0,255), -1)
            lm_txt += str(int(landmark[2*i] * img.shape[1])) + ' '
            lm_txt += str(int(landmark[2*i+1] * img.shape[0])) + ' '
        lm_txt += '\n'
        file.write(lm_txt)
    file.close()
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    # plt.imshow(img[:,:,::-1])
    # plt.show()
    # plt.imsave(f'{output_dir}/img_{idx}.png', img[:,:,::-1])
    cv2.imwrite(f'{output_dir}/image_{idx}.png', img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5n.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/widerface.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[800, 800], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', default=False, help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size
    
    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')

    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
    
    train_path = '/home/oem/vinit/FaceDetection/yolov5/yolov5-face/data/widerface/train_awing_spoof'
    imgsz = 800
    gs = 32

    dataset = LoadFaceImagesAndLabels(train_path, imgsz, batch_size,
                                      augment=True,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=opt.rect,  # rectangular training
                                      cache_images=opt.cache_images,
                                      single_cls=opt.single_cls,
                                      stride=int(gs),
                                      pad=0.0,
                                      image_weights=opt.image_weights,
                                    )

    output_dir = 'data_dump_crop_face_v4'
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    # for i in range(len(dataset)):
    from tqdm import tqdm
    for i in tqdm(range(len(dataset.img_files))):
        # if 'FaceDetect_' in dataset.img_files[i]:
        data = dataset[i]
        if data[-1] == True:
            showlabels(data[0], data[1], i, output_dir)
            # import pdb; pdb.set_trace()

    # data = dataset[0]
    # showlabels(data[0], data[1])

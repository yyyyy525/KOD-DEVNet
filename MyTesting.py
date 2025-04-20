import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from lib.pvt import Hitnet
from utils.dataloader import My_test_dataset

from PIL import Image
import matplotlib.cm as cm

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size default 352')
parser.add_argument('--pth_path', type=str,
                    default='')  
opt = parser.parse_args()

for _data_name in [ 'COD10K', 'CAMO', 'CHAMELEON', 'NC4K' ]:
    
    data_path = './{}/'.format(_data_name)
    cod_save_path = './res/{}/pred_C/'.format(_data_name)
    sod_save_path = './res/{}/pred_S/'.format(_data_name)
    all_save_path = './res/{}/pred_A/'.format(_data_name)
    color_save_path = './res/{}/pred_score/'.format(_data_name)
    model = Hitnet()
    model.cuda()
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.pth_path))
    model.eval()                

    os.makedirs(cod_save_path, exist_ok=True)
    os.makedirs(sod_save_path, exist_ok=True)
    os.makedirs(all_save_path, exist_ok=True)
    os.makedirs(color_save_path, exist_ok=True)
    image_root = '{}/Image/'.format(data_path)
    gtc_root = '{}/GT/'.format(data_path)
    gts_root = '{}/GT/'.format(data_path)
    gta_root = '{}/GT/'.format(data_path)
    print('root', image_root, gtc_root, gts_root, gta_root)
    test_loader = My_test_dataset(image_root, gtc_root, gts_root, gta_root, opt.testsize)
    print('****', test_loader.size)
    for i in range(test_loader.size):
        image, gt_c, gt_s, gt_a, name = test_loader.load_data()
        print('***name', name)
        gt_c = np.asarray(gt_c, np.float32)
        gt_c /= (gt_c.max() + 1e-8)

        gt_s = np.asarray(gt_s, np.float32)
        gt_s /= (gt_s.max() + 1e-8)

        gt_a = np.asarray(gt_a, np.float32)
        gt_a /= (gt_a.max() + 1e-8)
        image = image.cuda()

        res_c, res_a, res_s, res1_c, res1_a, res1_s, Ln, Pn = model(image)

        res_c = F.interpolate(res_c[-1] + res1_c, size=gt_c.shape, mode='bilinear', align_corners=False)
        res_c = res_c.sigmoid().data.cpu().numpy().squeeze()
        pred_c = (res_c - res_c.min()) / (res_c.max() - res_c.min() + 1e-8)

        res_a = F.interpolate(res_a[-1] + res1_a, size=gt_a.shape, mode='bilinear', align_corners=False)
        res = res_a
        res_a = res_a.sigmoid().data.cpu().numpy().squeeze()
        pred_a = (res_a - res_a.min()) / (res_a.max() - res_a.min() + 1e-8)
        cv2.imwrite(all_save_path + name, pred_a * 255)

        res_s = F.interpolate(res_s[-1] + res1_s, size=gt_s.shape, mode='bilinear', align_corners=False)
        res_s = res_s.sigmoid().data.cpu().numpy().squeeze()
        pred_s = (res_s - res_s.min()) / (res_s.max() - res_s.min() + 1e-8)

        res_n = F.interpolate(Ln[-1] + Pn, size=gt_a.shape, mode='bilinear', align_corners=False)

        sm = nn.Softmax(dim=1)
        res_n = sm(res_n)
        P = np.zeros((gt_a.shape[0], gt_a.shape[1]))
        res = res.detach().cpu().numpy()
        res = res.reshape(res.shape[2], res.shape[3])
        so = res_n[0][0].detach().cpu().numpy()
        co = res_n[0][1].detach().cpu().numpy()
        P = so
        print(so.shape)

        # 颜色映射
        cmap = cm.get_cmap("jet") 
        mapped_images = []
        # 存储映射结果
        mapped_image = np.zeros((P.shape[0], P.shape[1], 3), dtype=np.uint8)

        nonzero_pixels = (res > 0)
        mapped_image[nonzero_pixels] = (cmap(P[nonzero_pixels])[:, :3] * 255).astype(np.uint8)

        result = Image.fromarray(mapped_image)
        result.save(color_save_path + name)

        print('> {} - {}'.format(_data_name, name))
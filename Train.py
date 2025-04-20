
import torch
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import random
from datetime import datetime
from lib.pvt import Hitnet
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator

####
def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')

evaluator = Evaluator(2)
# weight=torch.from_numpy(np.array([0.1, 2.0, 0.5])).float()
criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode='ce')
global train_loss

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def val(model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        test_loader = test_dataset(image_root=opt.test_path + '/Imgs/',
                            gtc_root=opt.test_path + '/GTC/',
                            gts_root=opt.test_path + '/GTS/',
                            gta_root=opt.test_path + '/GT/',
                            gtn_root=opt.test_path + '/GT-falsecolor8/',
                            testsize=opt.trainsize)

        for i in range(test_loader.size):
            image, gt_c, gt_s, gt_a, gt_n, name = test_loader.load_data()
            gt_c = np.asarray(gt_c, np.float32)
            gt_c /= (gt_c.max() + 1e-8)
            gt_s = np.asarray(gt_s, np.float32)
            gt_s /= (gt_s.max() + 1e-8)
            gt_a = np.asarray(gt_a, np.float32)
            gt_a /= (gt_a.max() + 1e-8)
            image = image.cuda()
            res_c, res_a, res_s, res1_c,  res1_a,  res1_s,  res_n, res1_n = model(image)
            
            res_c = F.interpolate(res_c[-1] + res1_c, size=gt_c.shape, mode='bilinear', align_corners=False)
            res_c = res_c.sigmoid().data.cpu().numpy().squeeze()
            res_c = (res_c - res_c.min()) / (res_c.max() - res_c.min() + 1e-8)

            res_a = F.interpolate(res_a[-1] + res1_a, size=gt_a.shape, mode='bilinear', align_corners=False)
            res_a = res_a.sigmoid().data.cpu().numpy().squeeze()
            res_a = (res_a - res_a.min()) / (res_a.max() - res_a.min() + 1e-8)

            res_s = F.interpolate(res_s[-1] + res1_s, size=gt_s.shape, mode='bilinear', align_corners=False)
            res_s = res_s.sigmoid().data.cpu().numpy().squeeze()
            res_s = (res_s - res_s.min()) / (res_s.max() - res_s.min() + 1e-8)

            find_sod1 = "sun"
            find_sod2 = "ILSVR"
            if (find_sod1 in name) or (find_sod2 in name):
                mae_sum += np.sum(np.abs(res_s - gt_s)) * 1.0 / (gt_s.shape[0] * gt_s.shape[1])
            else:
                mae_sum += np.sum(np.abs(res_c - gt_c)) * 1.0 / (gt_c.shape[0] * gt_c.shape[1])

            # gt_n = gt_n.cpu().numpy()
            # pred = F.upsample(res_n[-1] + res1_n, size=(gt_n.shape[1], gt_n.shape[2]), mode='bilinear', align_corners=False)
            # pred = pred.data.cpu().numpy()
            # pred = np.argmax(pred, axis=1)
            # evaluator.add_batch(gt_n, pred)

            mae_sum += np.sum(np.abs(res_a - gt_a)) * 1.0 / (gt_a.shape[0] * gt_a.shape[1])

        mae = mae_sum / test_loader.size
        
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


def train(cotrain_loader, unitrain_loader, sotrain_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    
    global loss_P2_record
    loss_P2_record = AvgMeter()

    dataloader_cod = iter(cotrain_loader)
    dataloader_unify = iter(unitrain_loader)

    train_loss = 0

    for i, pack in enumerate(sotrain_loader):

        ### 更新 encoder 的参数
        for param in model.module.backbone.parameters():
            param.requires_grad = True

        Simages, Sgts_c, Sgts_s, Sgts_a, Sgts_n, Sname = pack
        # print(Sname)
        train_loss += iter_train(Simages, Sgts_c, Sgts_s, Sgts_a, Sgts_n, 2)
        try:
            Cimages, Cgts_c, Cgts_s, Cgts_a, Cgts_n, Cname = next(dataloader_cod)
            # print(Cname)
            train_loss += iter_train(Cimages, Cgts_c, Cgts_s, Cgts_a, Cgts_n, 1)
        except StopIteration:
            dataloader_cod = iter(cotrain_loader)
            Cimages, Cgts_c, Cgts_s, Cgts_a, Cgts_n, Cname = next(dataloader_cod)
            # print(Cname)
            train_loss += iter_train(Cimages, Cgts_c, Cgts_s, Cgts_a, Cgts_n, 1)

        if i % 60 == 0:  
            ### 不更新 encoder 的参数
            for param in model.module.backbone.parameters():
                param.requires_grad = False
            try:
                Uimages, Ugts_c, Ugts_s, Ugts_a, Ugts_n, Uname = next(dataloader_unify)
                # print(Uname)
                train_loss += iter_train(Uimages, Ugts_c, Ugts_s, Ugts_a, Ugts_n, 0)
            except StopIteration:
                dataloader_unify = iter(unitrain_loader)
                Uimages, Ugts_c, Ugts_s, Ugts_a, Ugts_n, Uname = next(dataloader_unify)
                # print(Uname)
                train_loss += iter_train(Uimages, Ugts_c, Ugts_s, Ugts_a, Ugts_n, 0)


        # ---- train visualization ----
        if i % 100 == 0 or i == so_total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, so_total_step,
                         loss_P2_record.show())) ##loss_P2_record.show()
            logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, so_total_step,
                         loss_P2_record.show()))
    # save model
    print("Loss:{}" ,train_loss)
    save_path = opt.save_path
    if epoch % opt.epoch_save == 0:
        torch.save(model.state_dict(), save_path + str(epoch) + 'Hitnet-PVT.pth')

def iter_train(images, gts_c, gts_s, gts_a, gts_n, flag):
    optimizer.zero_grad()
    # ---- data prepare ----
    images = Variable(images).cuda()
    gts_c = Variable(gts_c).cuda()
    gts_s = Variable(gts_s).cuda()
    gts_a = Variable(gts_a).cuda()
    gts_n = Variable(gts_n).cuda().squeeze(dim=1)
    # ---- rescale ----
    trainsize = int(round(opt.trainsize * 1 / 32) * 32)
    # ---- forward ----
    Plc, Pla, Pls, Pc, Pa, Ps, Ln, Pn = model(images)

    # ---- loss function ----
    losses_c = [structure_loss(out, gts_c) for out in Plc]
    losses_a = [structure_loss(out, gts_a) for out in Pla]
    losses_s = [structure_loss(out, gts_s) for out in Pls]
    losses_N = [criterion(out, gts_n) for out in Ln]

    loss_c = 0
    loss_a = 0
    loss_s = 0
    loss_ln = 0
    gamma = 0.2
    # print('iteration num',len(P1))
    for it in range(len(Plc)):
        loss_c += (gamma * it) * losses_c[it]
        loss_a += (gamma * it) * losses_a[it]
        loss_s += (gamma * it) * losses_s[it]
        loss_ln += (gamma * it) * losses_N[it]

    loss_Pc = structure_loss(Pc, gts_c)
    loss_Pa = structure_loss(Pa, gts_a)
    loss_Ps = structure_loss(Ps, gts_s)
    loss_Pn = criterion(Pn, gts_n)
    if flag == 0:
        loss = loss_c + loss_s + loss_a + loss_Pc + loss_Ps + loss_Pa + 5 * loss_ln + 5 * loss_Pn 
    elif flag == 1:
        loss = loss_c + loss_a + loss_Pc + loss_Pa + 5 * loss_ln + 5 * loss_Pn 
    elif flag == 2:
        loss = loss_s + loss_a + loss_Ps + loss_Pa + 5 * loss_ln + 5 * loss_Pn  
    # ---- backward ----
    loss.backward()
    clip_gradient(optimizer, opt.clip)
    optimizer.step()
    
    # ---- recording loss ----
    loss_P2_record.update(loss.data, opt.batchsize)
    return loss


if __name__ == '__main__':

    ##################model_name#############################
    # model_name = 'Hitnet_pvt_wo_pretrained_fusion'
    model_name = 'Hitnet_pvt_wo_pretrained_fusion_debug'

    ###############################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,default=60, help='epoch number')
    parser.add_argument('--lr', type=float,default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str,default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation',default=False, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int,default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,default=512, help='training dataset size,candidate=352,704,1056')
    parser.add_argument('--clip', type=float,default=0.5, help='gradient clipping margin')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--decay_rate', type=float,default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,default='./Train3decoder/',help='path to train dataset')
    parser.add_argument('--test_path', type=str,default='./Train3decoder/val',help='path to testing dataset')
    parser.add_argument('--save_path', type=str,default='./checkpoints/'+model_name+'/')
    parser.add_argument('--epoch_save', type=int,default=1, help='every n epochs to save model')
    opt = parser.parse_args()


    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    logging.basicConfig(filename=opt.save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")


    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = Hitnet().cuda()
    model = torch.nn.DataParallel(model, device_ids=[0,1])


    if opt.load is not None:
        pretrained_dict=torch.load(opt.load)
        print('!!!!!!Succefully load model from!!!!!! ', opt.load)
        load_matched_state_dict(model, pretrained_dict)

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('model paramters',sum(p.numel() for p in model.parameters() if p.requires_grad))

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, params), opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, params), opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    cimage_root = '{}/COData/Imgs/'.format(opt.train_path)
    cgtc_root = '{}/COData/GTC/'.format(opt.train_path)
    cgts_root = '{}/COData/GTS/'.format(opt.train_path)
    cgta_root = '{}/COData/GT/'.format(opt.train_path)
    cgtn_root = '{}/COData/GT-falsecolor8/'.format(opt.train_path)

    aimage_root = '{}/COSOData/Imgs/'.format(opt.train_path)
    agtc_root = '{}/COSOData/GTC/'.format(opt.train_path)
    agts_root = '{}/COSOData/GTS/'.format(opt.train_path)
    agta_root = '{}/COSOData/GT/'.format(opt.train_path)
    agtn_root = '{}/COSOData/GT-falsecolor8/'.format(opt.train_path)

    simage_root = '{}/SOData/Imgs/'.format(opt.train_path)
    sgtc_root = '{}/SOData/GTC/'.format(opt.train_path)
    sgts_root = '{}/SOData/GTS/'.format(opt.train_path)
    sgta_root = '{}/SOData/GT/'.format(opt.train_path)
    sgtn_root = '{}/SOData/GT-falsecolor8/'.format(opt.train_path)


    cotrain_loader = get_loader(cimage_root, cgtc_root, cgts_root, cgta_root, cgtn_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    unitrain_loader = get_loader(aimage_root, agtc_root, agts_root, agta_root, agtn_root, batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    sotrain_loader = get_loader(simage_root, sgtc_root, sgts_root, sgta_root, sgtn_root, batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              augmentation=opt.augmentation)

    co_total_step = len(cotrain_loader)
    uni_total_step = len(unitrain_loader)
    so_total_step = len(sotrain_loader)

    writer = SummaryWriter(opt.save_path + 'summary')

    print("#" * 20, "Start Training", "#" * 20)
    best_mae = 1
    best_epoch = 0
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(cotrain_loader, unitrain_loader, sotrain_loader, model, optimizer, epoch, opt.save_path)
        if epoch % opt.epoch_save==0:
            val( model, epoch, opt.save_path, writer)

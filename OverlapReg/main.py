import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn import DataParallel
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys, os
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_utils import ModelNet40_Reg
from model import OverlapNet
from evaluate_funcs import calculate_R_msemae, calculate_t_msemae, evaluate_mask


use_cuda = torch.cuda.is_available()
gpu_id = 0
torch.cuda.set_device(gpu_id)
if not os.path.isdir("./logs"):
    os.mkdir("./logs")
writer = SummaryWriter('./logs')
batch_size = 16  # 16
epochs = 500
lr = 1e-3
partial_overlap = 2
subsampled_rate_src = 0.8
subsampled_rate_tgt = 0.8
unseen = False
noise = False
file_type = 'modelnet40'
# file_type = 'Kitti_odo'
# file_type = 'bunny'


# set seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(1234)


def test_one_epoch(net, test_loader):
    net.eval()
    total_loss = 0
    reg_total_loss = 0

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():

        accs_src = []
        preciss_src = []
        recalls_src = []
        f1s_src = []

        accs_tgt = []
        preciss_tgt = []
        recalls_tgt = []
        f1s_tgt = []

        Rs_gt = []
        ts_gt = []
        Rs1_pred = []
        ts1_pred = []
        for src, target, R_gt, t_gt, euler, gt_mask_src, gt_mask_tgt in tqdm(test_loader):
            batch_size = src.shape[0]
            if use_cuda:
                src = src.cuda()
                target = target.cuda()
                R_gt = R_gt.cuda()
                t_gt = t_gt.cuda()

            init_R, init_t, mask_src, mask_tgt, mask_src_idx, mask_tgt_idx, \
                R1_pred, t1_pred = net(src, target)

            # compute mask loss and registration loss
            loss_mask = loss_fn(mask_src, gt_mask_src.long().cuda()) + loss_fn(mask_tgt, gt_mask_tgt.long().cuda())
            E = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
            loss_init = F.mse_loss(torch.matmul(init_R.transpose(2, 1), R_gt), E) \
                        + F.mse_loss(init_t, t_gt)
            loss_reg1 = F.mse_loss(torch.matmul(R1_pred.transpose(2, 1), R_gt), E) \
                   + F.mse_loss(t1_pred, t_gt)
            loss_reg = loss_reg1
            loss = loss_mask + loss_reg + loss_init
            total_loss += loss.item()
            reg_total_loss += loss_reg.item()

            Rs_gt.append(R_gt.detach().cpu().numpy())
            ts_gt.append(t_gt.detach().cpu().numpy())
            Rs1_pred.append(R1_pred.detach().cpu().numpy())
            ts1_pred.append(t1_pred.detach().cpu().numpy())
            # 评估
            acc_src, precis_src, recall_src, _ = evaluate_mask(torch.max(mask_src, dim=1)[1], gt_mask_src)
            _, _, _, f1_src = evaluate_mask(mask_src_idx, gt_mask_src)
            accs_src.append(acc_src)
            preciss_src.append(precis_src)
            recalls_src.append(recall_src)
            f1s_src.append(f1_src)

            acc_tgt, precis_tgt, recall_tgt, _ = evaluate_mask(torch.max(mask_tgt, dim=1)[1], gt_mask_tgt)
            _, _, _, f1_tgt = evaluate_mask(mask_tgt_idx, gt_mask_tgt)
            accs_tgt.append(acc_tgt)
            preciss_tgt.append(precis_tgt)
            recalls_tgt.append(recall_tgt)
            f1s_tgt.append(f1_tgt)

        Rs_gt = np.concatenate(Rs_gt, axis=0)
        ts_gt = np.concatenate(ts_gt, axis=0)
        Rs1_pred = np.concatenate(Rs1_pred, axis=0)
        ts1_pred = np.concatenate(ts1_pred, axis=0)

        acc_src = np.mean(accs_src)
        precis_src = np.mean(preciss_src)
        recall_src = np.mean(recalls_src)
        f1_src = np.mean(f1s_src)

        acc_tgt = np.mean(accs_tgt)
        precis_tgt = np.mean(preciss_tgt)
        recall_tgt = np.mean(recalls_tgt)
        f1_tgt = np.mean(f1s_tgt)

        f1 = (f1_src + f1_tgt) / 2
        acc = (acc_src + acc_tgt) / 2
        precis = (precis_src + precis_tgt) / 2
        recall = (recall_src + recall_tgt) / 2

    return total_loss, f1, acc, precis, recall, Rs_gt, ts_gt, Rs1_pred, ts1_pred, reg_total_loss


def train_one_epoch(net, opt, train_loader):
    net.train()
    total_loss = 0
    reg_total_loss = 0
    accs_src = []
    preciss_src = []
    recalls_src = []
    f1s_src = []

    accs_tgt = []
    preciss_tgt = []
    recalls_tgt = []
    f1s_tgt = []

    Rs_gt = []
    ts_gt = []
    Rs1_pred = []
    ts1_pred = []

    loss_fn = nn.CrossEntropyLoss()

    for src, target, R_gt, t_gt, euler, gt_mask_src, gt_mask_tgt in tqdm(train_loader):
        batch_size = src.shape[0]
        # print(src.shape, target.shape)
        if use_cuda:
            src = src.cuda()
            target = target.cuda()
            R_gt = R_gt.cuda()
            t_gt = t_gt.cuda()

        init_R, init_t, mask_src, mask_tgt, mask_src_idx, mask_tgt_idx, \
            R1_pred, t1_pred = net(src, target)

        opt.zero_grad()
        # compute mask loss and registration loss
        loss_mask = loss_fn(mask_src, gt_mask_src.long().cuda()) + loss_fn(mask_tgt, gt_mask_tgt.long().cuda())
        E = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        loss_init = F.mse_loss(torch.matmul(init_R.transpose(2, 1), R_gt), E) \
                    + F.mse_loss(init_t, t_gt)
        loss_reg1 = F.mse_loss(torch.matmul(R1_pred.transpose(2, 1), R_gt), E) \
               + F.mse_loss(t1_pred, t_gt)
        loss_reg = loss_reg1
        # loss = loss_mask/loss_mask.detach() + loss_reg/loss_reg.detach()
        loss = loss_mask + loss_reg + loss_init
        # print((loss_mask/loss_mask.detach()).item(), loss_reg.item())
        total_loss += loss_mask.item() + loss_reg.item()
        reg_total_loss += loss_reg.item()
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), 5, norm_type=2)
        opt.step()

        Rs_gt.append(R_gt.detach().cpu().numpy())
        ts_gt.append(t_gt.detach().cpu().numpy())
        Rs1_pred.append(R1_pred.detach().cpu().numpy())
        ts1_pred.append(t1_pred.detach().cpu().numpy())
        # 评估
        acc_src, precis_src, recall_src, _ = evaluate_mask(torch.max(mask_src, dim=1)[1], gt_mask_src)
        _, _, _, f1_src = evaluate_mask(mask_src_idx, gt_mask_src)
        accs_src.append(acc_src)
        preciss_src.append(precis_src)
        recalls_src.append(recall_src)
        f1s_src.append(f1_src)

        acc_tgt, precis_tgt, recall_tgt, _ = evaluate_mask(torch.max(mask_tgt, dim=1)[1], gt_mask_tgt)
        _, _, _, f1_tgt = evaluate_mask(mask_tgt_idx, gt_mask_tgt)
        accs_tgt.append(acc_tgt)
        preciss_tgt.append(precis_tgt)
        recalls_tgt.append(recall_tgt)
        f1s_tgt.append(f1_tgt)

    Rs_gt = np.concatenate(Rs_gt, axis=0)
    ts_gt = np.concatenate(ts_gt, axis=0)
    Rs1_pred = np.concatenate(Rs1_pred, axis=0)
    ts1_pred = np.concatenate(ts1_pred, axis=0)
    acc_src = np.mean(accs_src)
    precis_src = np.mean(preciss_src)
    recall_src = np.mean(recalls_src)
    f1_src = np.mean(f1s_src)

    acc_tgt = np.mean(accs_tgt)
    precis_tgt = np.mean(preciss_tgt)
    recall_tgt = np.mean(recalls_tgt)
    f1_tgt = np.mean(f1s_tgt)

    f1 = (f1_src + f1_tgt) / 2
    acc = (acc_src + acc_tgt) / 2
    precis = (precis_src + precis_tgt) / 2
    recall = (recall_src + recall_tgt) / 2

    return total_loss, f1, acc, precis, recall, Rs_gt, ts_gt, Rs1_pred, ts1_pred, reg_total_loss


if __name__ == '__main__':

    best_loss = np.inf
    best_f1 = 0
    best_precis = 0
    best_recall = 0
    best_acc = 0

    best_R_mse = np.inf
    best_R_rmse = np.inf
    best_R_mae = np.inf
    best_t_mse = np.inf
    best_t_rmse = np.inf
    best_t_mae = np.inf

    best_t_error = np.inf
    best_angle_error = np.inf
    best_angle_mat_error = np.inf

    if file_type == 'modelnet40':
        all_points = 1024
        src_subsampled_points = int(subsampled_rate_src * all_points)
        tgt_subsampled_points = int(subsampled_rate_tgt * all_points)
    elif file_type in ['bunny', 'Kitti_odo']:
        all_points = 2048
        src_subsampled_points = int(subsampled_rate_src * all_points)
        tgt_subsampled_points = int(subsampled_rate_tgt * all_points)
    train_loader = DataLoader(
        dataset=ModelNet40_Reg(all_points, partition='train', max_angle=45, max_t=0.5, unseen=unseen,
                               file_type=file_type,
                               subsampled_rate_src=subsampled_rate_src, subsampled_rate_tgt=subsampled_rate_tgt,
                               partial_overlap=partial_overlap, noise=noise),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        dataset=ModelNet40_Reg(all_points, partition='test', max_angle=45, max_t=0.5, unseen=unseen,
                               file_type=file_type,
                               subsampled_rate_src=subsampled_rate_src, subsampled_rate_tgt=subsampled_rate_tgt,
                               partial_overlap=partial_overlap, noise=noise),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=2
    )

    net = OverlapNet(all_points=all_points, src_subsampled_points=src_subsampled_points, tgt_subsampled_points=tgt_subsampled_points)
    opt = optim.RAdam(params=net.parameters(), lr=lr)

    if use_cuda:
        net = net.cuda()
        # net = DataParallel(net, device_ids=[0, 1])

    start_epoch = -1
    RESUME = False  # 是否加载模型继续上次训练
    if RESUME:
        path_checkpoint = "./checkpoint/ckpt%s.pth"%(str(file_type)+str(subsampled_rate_src)+str(subsampled_rate_tgt))  # 断点路径
        checkpoint = torch.load(path_checkpoint, map_location=lambda storage, loc: storage.cuda(gpu_id))  # 加载断点
        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        # scheduler.load_state_dict(checkpoint["lr_step"])
        opt.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        # 加载上次best结果
        best_loss = checkpoint['best_loss']
        best_precis = checkpoint['best_Precis']
        best_recall = checkpoint['best_Recall']
        best_acc = checkpoint['best_Acc']
        best_f1 = checkpoint['best_f1']

        best_R_mse = checkpoint['best_MSE(R)']
        best_R_rmse = checkpoint['best_RMSE(R)']
        best_R_mae = checkpoint['best_MAE(R)']
        best_t_mse = checkpoint['best_MSE(t)']
        best_t_rmse = checkpoint['best_RMSE(t)']
        best_t_mae = checkpoint['best_MAE(t)']

    for epoch in range(start_epoch + 1, epochs):
        train_total_loss, train_f1, train_acc, train_precis, train_recall, train_Rs_gt, train_ts_gt, \
        train_Rs1_pred, train_ts1_pred, train_reg_loss = train_one_epoch(net, opt, train_loader)

        test_total_loss, test_f1, test_acc, test_precis, test_recall, test_Rs_gt, test_ts_gt, \
        test_Rs1_pred, test_ts1_pred, test_reg_loss = test_one_epoch(net, test_loader)

        train_R_mse, train_R_mae = calculate_R_msemae(train_Rs_gt, train_Rs1_pred)
        train_R_rmse = np.sqrt(train_R_mse)
        train_t_mse, train_t_mae = calculate_t_msemae(train_ts_gt, train_ts1_pred)
        train_t_rmse = np.sqrt(train_t_mse)

        test_R_mse, test_R_mae = calculate_R_msemae(test_Rs_gt, test_Rs1_pred)
        test_R_rmse = np.sqrt(test_R_mse)
        test_t_mse, test_t_mae = calculate_t_msemae(test_ts_gt, test_ts1_pred)
        test_t_rmse = np.sqrt(test_t_mse)

        if test_reg_loss <= best_loss:
            best_loss = test_reg_loss
            best_precis = test_precis
            best_recall = test_recall
            best_f1 = test_f1
            best_acc = test_acc

            best_R_mse = test_R_mse
            best_R_rmse = test_R_rmse
            best_R_mae = test_R_mae

            best_t_mse = test_t_mse
            best_t_rmse = test_t_rmse
            best_t_mae = test_t_mae

            # 保存最好的checkpoint
            checkpoint_best = {
                "net": net.state_dict(),
            }
            if not os.path.isdir("./checkpoint"):
                os.mkdir("./checkpoint")
            torch.save(checkpoint_best, './checkpoint/ckpt_best%s.pth'%(str(file_type)+str(subsampled_rate_src)+str(subsampled_rate_tgt)))

        print('---------Epoch: %d---------' % (epoch+1))
        print('Train: Loss: %f, F1: %f, Acc: %f, Precis: %f, Recall: %f, MSE(R): %f, RMSE(R): %f, '
              'MAE(R): %f, MSE(t): %f, RMSE(t): %f, MAE(t): %f'
              % (train_total_loss, train_f1, train_acc, train_precis, train_recall, train_R_mse, train_R_rmse,
                 train_R_mae, train_t_mse, train_t_rmse, train_t_mae))

        print('Test: Loss: %f, F1: %f, Acc: %f, Precis: %f, Recall: %f, MSE(R): %f, RMSE(R): %f, '
              'MAE(R): %f, MSE(t): %f, RMSE(t): %f, MAE(t): %f,'
              % (test_total_loss, test_f1, test_acc, test_precis, test_recall, test_R_mse, test_R_rmse,
                 test_R_mae, test_t_mse, test_t_rmse, test_t_mae))

        print('Best: Loss: %f, F1: %f, Acc: %f, Precis: %f, Recall: %f, MSE(R): %f, RMSE(R): %f, '
              'MAE(R): %f, MSE(t): %f, RMSE(t): %f, MAE(t): %f'
              % (best_loss, best_f1, best_acc, best_precis, best_recall, best_R_mse, best_R_rmse,
                 best_R_mae, best_t_mse, best_t_rmse, best_t_mae))
        writer.add_scalar('Train/train_loss', train_total_loss, global_step=epoch)
        writer.add_scalar('Train/train_Precis', train_precis, global_step=epoch)
        writer.add_scalar('Train/train_Recall', train_recall, global_step=epoch)
        writer.add_scalar('Train/train_Acc', train_acc, global_step=epoch)
        writer.add_scalar('Train/train_F1', train_f1, global_step=epoch)
        writer.add_scalar('Train/train_MSER', train_R_mse, global_step=epoch)
        writer.add_scalar('Train/train_RMSER', train_R_rmse, global_step=epoch)
        writer.add_scalar('Train/train_MAER', train_R_mae, global_step=epoch)
        writer.add_scalar('Train/train_MSEt', train_t_mse, global_step=epoch)
        writer.add_scalar('Train/train_RMSEt', train_t_rmse, global_step=epoch)
        writer.add_scalar('Train/train_MAEt', train_t_mae, global_step=epoch)

        writer.add_scalar('Test/test_loss', test_total_loss, global_step=epoch)
        writer.add_scalar('Test/test_Precis', test_precis, global_step=epoch)
        writer.add_scalar('Test/test_Recall', test_recall, global_step=epoch)
        writer.add_scalar('Test/test_Acc', test_acc, global_step=epoch)
        writer.add_scalar('Test/test_F1', test_f1, global_step=epoch)
        writer.add_scalar('Test/test_MSER', test_R_mse, global_step=epoch)
        writer.add_scalar('Test/test_RMSER', test_R_rmse, global_step=epoch)
        writer.add_scalar('Test/test_MAER', test_R_mae, global_step=epoch)
        writer.add_scalar('Test/test_MSEt', test_t_mse, global_step=epoch)
        writer.add_scalar('Test/test_RMSEt', test_t_rmse, global_step=epoch)
        writer.add_scalar('Test/test_MAEt', test_t_mae, global_step=epoch)

        writer.add_scalar('Best/best_loss', best_loss, global_step=epoch)
        writer.add_scalar('Best/best_Precis', best_precis, global_step=epoch)
        writer.add_scalar('Best/best_Recall', best_recall, global_step=epoch)
        writer.add_scalar('Best/best_Acc', best_acc, global_step=epoch)
        writer.add_scalar('Best/best_F1', best_f1, global_step=epoch)
        writer.add_scalar('Best/best_MSER', best_R_mse, global_step=epoch)
        writer.add_scalar('Best/best_RMSER', best_R_rmse, global_step=epoch)
        writer.add_scalar('Best/best_MAER', best_R_mae, global_step=epoch)
        writer.add_scalar('Best/best_MSEt', best_t_mse, global_step=epoch)
        writer.add_scalar('Best/best_RMSEt', best_t_rmse, global_step=epoch)
        writer.add_scalar('Best/best_MAEt', best_t_mae, global_step=epoch)

        # 保存checkpoint
        checkpoint = {
            "net": net.state_dict(),
            'optimizer': opt.state_dict(),
            "epoch": epoch,
            # "lr_step": scheduler.state_dict(),
            "best_loss":best_loss,
            'best_Precis': best_precis,
            'best_Recall': best_recall,
            'best_Acc': best_acc,
            'best_f1': best_f1,

            'best_MSE(R)': best_R_mse,
            'best_RMSE(R)': best_R_rmse,
            'best_MAE(R)': best_R_mae,
            'best_MSE(t)': best_t_mse,
            'best_RMSE(t)': best_t_rmse,
            'best_MAE(t)': best_t_mae,
        }
        if not os.path.isdir("./checkpoint"):
            os.mkdir("./checkpoint")
        torch.save(checkpoint, './checkpoint/ckpt%s.pth'%(str(file_type)+str(subsampled_rate_src)+str(subsampled_rate_tgt)))
    writer.close()





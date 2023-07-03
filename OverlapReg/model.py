import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math, sys, random
sys.path.append("..")
from feature_extract import PointNet, DGCNN
from util import batch_transform, batch_quat2mat


def mask_point(mask_idx, points):
    # masks: [b, n] : Tensor, 包含0和1
    # points: [b, 3, n] : Tensor
    # return: [b, 3, n2] : Tensor
    batch_size = points.shape[0]
    points = points.permute(0, 2, 1)
    mask_idx = mask_idx.reshape(batch_size, -1, 1)
    new_pcs = points * mask_idx
    new_points = []

    for new_pc in new_pcs:
        # 删除被屏蔽的0点
        temp = new_pc[:, ...] == 0
        temp = temp.cpu()
        idx = np.argwhere(temp.all(axis=1))
        new_point = np.delete(new_pc.cpu().detach().numpy(), idx, axis=0)
        new_points.append(new_point)

    new_points = np.array(new_points)
    new_points = torch.from_numpy(new_points)
    if torch.cuda.is_available():
        new_points = new_points.cuda()
    return new_points.permute(0, 2, 1)


def gather_points(points, inds):
    '''
    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]


def feature_interaction(src_embedding, tar_embedding):
    # embedding: (batch, emb_dims, num_points)
    num_points1 = src_embedding.shape[2]

    simi1 = cos_simi(src_embedding, tar_embedding)  # (num_points1, num_points2)

    src_embedding = src_embedding.permute(0, 2, 1)
    tar_embedding = tar_embedding.permute(0, 2, 1)

    simi_src = nn.Softmax(dim=2)(simi1)  # 转化为概率
    glob_tar = torch.matmul(simi_src, tar_embedding)  # 加权平均tar的全局特征
    glob_src = torch.max(src_embedding, dim=1, keepdim=True)[0]
    glob_src = glob_src.repeat(1, num_points1, 1)
    # print(glob_src.shape, glob_tar.shape,src_embedding.shape)
    inter_src_feature = torch.cat((src_embedding, glob_tar, glob_src, glob_tar-glob_src), dim=2)  # 交互特征
    inter_src_feature = inter_src_feature.permute(0, 2, 1)

    return inter_src_feature


def cos_simi(src_embedding, tgt_embedding):
    # (batch, emb_dims, num_points)
    batch_size, num_dims, num_points1 = src_embedding.size()
    batch_size, num_dims, num_points2 = tgt_embedding.size()

    # src_norm = src_embedding / (src_embedding.norm(dim=1).reshape(batch_size, 1, num_points1))
    # tar_norm = tgt_embedding / (tgt_embedding.norm(dim=1).reshape(batch_size, 1, num_points2))
    src_norm = F.normalize(src_embedding, p=2, dim=1)
    tar_norm = F.normalize(tgt_embedding, p=2, dim=1)
    simi = torch.matmul(src_norm.transpose(2, 1).contiguous(), tar_norm)  # (batch, num_points1, num_points2)
    return simi


class MLPs(nn.Module):
    def __init__(self, in_dim, mlps):
        super(MLPs, self).__init__()
        self.mlps = nn.Sequential()
        l = len(mlps)
        for i, out_dim in enumerate(mlps):
            self.mlps.add_module(f'fc_{i}', nn.Linear(in_dim, out_dim))
            if i != l - 1:
                self.mlps.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        x = self.mlps(x)
        return x


class InitReg(nn.Module):
    def __init__(self):
        super(InitReg, self).__init__()
        self.num_dims = 512
        self.encoder = PointNet(n_emb_dims=self.num_dims)
        self.decoder = MLPs(in_dim=self.num_dims*2, mlps=[512, 512, 256, 7])

    def forward(self, src, tgt):
        # (batch, 3, n)
        src_emb = self.encoder(src)
        tgt_emb = self.encoder(tgt)
        src_glob, _ = torch.max(src_emb, dim=2)
        tgt_glob, _ = torch.max(tgt_emb, dim=2)
        pose7d = self.decoder(torch.cat((src_glob, tgt_glob), dim=1))
        batch_t, batch_quat = pose7d[:, :3], pose7d[:, 3:] / (
                torch.norm(pose7d[:, 3:], dim=1, keepdim=True) + 1e-8)
        batch_R = batch_quat2mat(batch_quat)

        return batch_R, batch_t


class RegNet(nn.Module):
    def __init__(self, n_emb_dims=1024):
        super(RegNet, self).__init__()
        self.emb_dims = n_emb_dims
        self.emb_dims1 = int(self.emb_dims / 2)
        self.emb_nn1 = DGCNN(self.emb_dims1, k=32)
        self.init_reg = InitReg()
        self.emb_nn2_src = nn.Sequential(
            nn.Conv1d(self.emb_dims1 * 4, self.emb_dims1 * 2, 1), nn.BatchNorm1d(self.emb_dims1 * 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(self.emb_dims1 * 2, self.emb_dims, 1), nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.emb_nn2_tgt = nn.Sequential(
            nn.Conv1d(self.emb_dims1 * 4, self.emb_dims1 * 2, 1), nn.BatchNorm1d(self.emb_dims1 * 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(self.emb_dims1 * 2, self.emb_dims, 1), nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.my_iter = torch.ones(1)
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def generate_keypoints(self, src, tgt, src_embedding, tgt_embedding):
        # src, tgt: (batch, n, 3)
        # embedding: (batch, emb_dims, num_points)
        # return: (batch, 3, n*3/4)
        num_points1 = src.shape[1]
        num_points2 = tgt.shape[1]
        simi1 = cos_simi(src_embedding, tgt_embedding)  # (batch, num_points1, num_points2)
        simi1 = torch.max(simi1, dim=2)[0]  # (batch, num_points1)
        values, indices = torch.topk(simi1, k=int(num_points1 * 0.9), dim=1, sorted=False)
        src_keypoints = gather_points(src, indices)
        src_embedding_key = gather_points(src_embedding.permute(0, 2, 1), indices)

        simi2 = cos_simi(tgt_embedding, src_embedding)  # (batch, num_points2, num_points1)
        simi2 = torch.max(simi2, dim=2)[0]  # (batch, num_points1)
        values, indices = torch.topk(simi2, k=int(num_points2 * 0.9), dim=1, sorted=False)
        tgt_keypoints = gather_points(tgt, indices)
        tgt_embedding_key = gather_points(tgt_embedding.permute(0, 2, 1), indices)

        return src_keypoints.permute(0, 2, 1), tgt_keypoints.permute(0, 2, 1), src_embedding_key.permute(0, 2, 1), tgt_embedding_key.permute(0, 2, 1)

    def generate_corr(self,src, tgt, src_embedding, tar_embedding):
        # src, tgt: (batch, n, 3)
        # embedding: (batch, emb_dims, num_points)
        simi1 = cos_simi(src_embedding, tar_embedding)  # (num_points1, num_points2)
        simi2 = cos_simi(tar_embedding, src_embedding)

        simi_src = nn.Softmax(dim=2)(simi1)  # 转化为概率
        src_corr = torch.matmul(simi_src, tgt)  # 加权平均tar的全局特征作为对应点(n1, 3)

        simi_tar = nn.Softmax(dim=2)(simi2)  # 转化为概率
        tgt_corr = torch.matmul(simi_tar, src)  # 加权平均src的全局特征作为对应点(n2, 3)

        return src_corr.permute(0, 2, 1), tgt_corr.permute(0, 2, 1)

    def SVD(self, src, src_corr):
        # (batch, 3, n)
        batch_size = src.shape[0]
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous()).cpu()
        R = []
        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0)).contiguous()
            r_det = torch.det(r).item()
            diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                              [0, 1.0, 0],
                                              [0, 0, r_det]]).astype('float32')).to(v.device)
            r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
            R.append(r)

        R = torch.stack(R, dim=0).cuda()
        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        if self.training:
            self.my_iter += 1
        return R, t.view(batch_size, 3)

    def forward(self, src, tgt):
        # src, tgt: (batch, 3, n)
        # embedding: (batch, emb_dims, num_points)
        # trans_src = self.stn(src)
        # trans_tgt = self.stn(tgt)
        # src = torch.bmm(src.transpose(2, 1), trans_src).transpose(2, 1)
        # tgt = torch.bmm(tgt.transpose(2, 1), trans_tgt).transpose(2, 1)
        src_embedding = self.emb_nn1(src)
        tgt_embedding = self.emb_nn1(tgt)
        # 特征融合
        inter_src_feature = feature_interaction(src_embedding, tgt_embedding)
        inter_tar_feature = feature_interaction(tgt_embedding, src_embedding)
        # 进一步提取特征
        src_embedding = self.emb_nn2_src(inter_src_feature)
        tgt_embedding = self.emb_nn2_tgt(inter_tar_feature)
        src, tgt, src_embedding, tgt_embedding = self.generate_keypoints(src.permute(0, 2, 1), tgt.permute(0, 2, 1), src_embedding, tgt_embedding)
        src_embedding = torch.where(torch.isnan(src_embedding), torch.full_like(src_embedding, random.random()),
                                    src_embedding)
        tgt_embedding = torch.where(torch.isnan(tgt_embedding), torch.full_like(tgt_embedding, random.random()),
                                    tgt_embedding)
        src_corr, tgt_corr = self.generate_corr(src.permute(0, 2, 1), tgt.permute(0, 2, 1), src_embedding, tgt_embedding)
        R1, t1 = self.SVD(src, src_corr)

        return R1, t1


class OverlapNet(nn.Module):
    def __init__(self, n_emb_dims=1024, all_points=1024, src_subsampled_points=768, tgt_subsampled_points=768):
        super(OverlapNet, self).__init__()
        self.emb_dims = n_emb_dims
        self.all_points = all_points
        self.emb_dims1 = int(self.emb_dims / 2)
        self.src_subsampled_points = src_subsampled_points
        self.tgt_subsampled_points = tgt_subsampled_points
        self.reg_net = RegNet(self.emb_dims)
        self.emb_nn1 = DGCNN(self.emb_dims1, k=32)
        self.init_reg = InitReg()
        self.emb_nn2_src = nn.Sequential(
            nn.Conv1d(self.emb_dims1 * 4, self.emb_dims1 * 2, 1), nn.BatchNorm1d(self.emb_dims1 * 2),nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(self.emb_dims1 * 2, self.emb_dims, 1), nn.BatchNorm1d(self.emb_dims),nn.LeakyReLU(negative_slope=0.01),
        )
        self.emb_nn2_tgt = nn.Sequential(
            nn.Conv1d(self.emb_dims1 * 4, self.emb_dims1 * 2, 1), nn.BatchNorm1d(self.emb_dims1 * 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(self.emb_dims1 * 2, self.emb_dims, 1), nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.score_nn_src = nn.Sequential(
            nn.Conv1d(self.emb_dims, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(512, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 1, 1), nn.Sigmoid()
        )
        self.score_nn_tgt = nn.Sequential(
            nn.Conv1d(self.emb_dims, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(512, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 1, 1), nn.Sigmoid()
        )
        self.mask_src_nn = nn.Sequential(
            nn.Conv1d(self.tgt_subsampled_points, 1024, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(512, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 2, 1)
        )
        self.mask_tgt_nn = nn.Sequential(
            nn.Conv1d(self.src_subsampled_points, 1024, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(512, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 2, 1)
        )

    def forward(self, *input):
        src = input[0]  # 1024
        tgt = input[1]  # 768
        batch_size = src.shape[0]
        init_R, init_t = self.init_reg(src, tgt)
        src = torch.matmul(init_R, src) + init_t.reshape(batch_size, 3, 1)

        src_embedding = self.emb_nn1(src)
        tgt_embedding = self.emb_nn1(tgt)
        # 特征融合
        inter_src_feature = feature_interaction(src_embedding, tgt_embedding)
        inter_tar_feature = feature_interaction(tgt_embedding, src_embedding)
        # 进一步提取特征
        src_embedding = self.emb_nn2_src(inter_src_feature)
        tgt_embedding = self.emb_nn2_tgt(inter_tar_feature)
        # 计算打分
        src_score = self.score_nn_src(src_embedding).reshape(batch_size, 1, -1)
        tgt_score = self.score_nn_tgt(tgt_embedding).reshape(batch_size, 1, -1)

        src_score = nn.Softmax(dim=2)(src_score)
        tgt_score = nn.Softmax(dim=2)(tgt_score)

        simi1 = cos_simi(src_embedding, tgt_embedding)
        simi2 = cos_simi(tgt_embedding, src_embedding)

        # 结合打分计算相似度
        simi_src = simi1 * tgt_score
        simi_tgt = simi2 * src_score

        mask_src = self.mask_src_nn(simi_src.permute(0, 2, 1))
        mask_tgt = self.mask_tgt_nn(simi_tgt.permute(0, 2, 1))
        overlap_points = self.all_points - (self.all_points - self.src_subsampled_points) \
                         - (self.all_points - self.tgt_subsampled_points)

        mask_src_score = torch.softmax(mask_src, dim=1)[:, 1, :].detach()  # (B, N)
        mask_tgt_score = torch.softmax(mask_tgt, dim=1)[:, 1, :].detach()
        # 取前overlap_points个点作为重叠点
        mask_src_idx = torch.zeros(mask_src_score.shape).cuda()
        values, indices = torch.topk(mask_src_score, k=overlap_points, dim=1)
        mask_src_idx.scatter_(1, indices, 1)  # (dim, 索引, 根据索引赋的值)
        # mask_src_idx = torch.where(mask_src > values[:, -1].reshape(batch_size, -1), 1, 0)

        mask_tgt_idx = torch.zeros(mask_tgt_score.shape).cuda()
        values, indices = torch.topk(mask_tgt_score, k=overlap_points, dim=1)
        mask_tgt_idx.scatter_(1, indices, 1)  # (dim, 索引, 根据索引赋的值)
        # mask_tgt_idx = torch.where(mask_tgt > values[:, -1].reshape(batch_size, -1), 1, 0)

        src = mask_point(mask_src_idx, src)
        tgt = mask_point(mask_tgt_idx, tgt)

        R1, t1 = self.reg_net(src, tgt)

        return init_R, init_t, mask_src, mask_tgt, mask_src_idx, mask_tgt_idx, R1, t1


# # src,tar:[batchsize, 3, num_points]
# src = torch.rand([4, 3, 800])
# tar = torch.rand([4, 3, 768])
# model = OverlapNet()
# mask_src, mask_tgt, mask_src_idx, mask_tgt_idx = model(src, tar)



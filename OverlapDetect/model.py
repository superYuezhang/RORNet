import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math, sys
sys.path.append("..")
from feature_extract import PointNet, DGCNN


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
    # src_norm = src_embedding / (src_embedding.norm(dim=1).reshape(batch_size, 1, num_points1))
    # tar_norm = tgt_embedding / (tgt_embedding.norm(dim=1).reshape(batch_size, 1, num_points2))
    src_norm = F.normalize(src_embedding, p=2, dim=1)
    tar_norm = F.normalize(tgt_embedding, p=2, dim=1)
    simi = torch.matmul(src_norm.transpose(2, 1).contiguous(), tar_norm)  # (batch, num_points1, num_points2)
    return simi


class OverlapNet(nn.Module):
    def __init__(self, n_emb_dims=1024, all_points=1024, src_subsampled_points=768, tgt_subsampled_points=768):
        super(OverlapNet, self).__init__()
        self.emb_dims = n_emb_dims
        self.all_points = all_points
        self.emb_dims1 = int(self.emb_dims / 2)
        self.src_subsampled_points = src_subsampled_points
        self.tgt_subsampled_points = tgt_subsampled_points
        self.emb_nn1 = DGCNN(self.emb_dims1, k=32)
        self.emb_nn2_src = nn.Sequential(
            nn.Conv1d(self.emb_dims1 * 4, self.emb_dims1 * 2, 1), nn.BatchNorm1d(self.emb_dims1 * 2),nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(self.emb_dims1 * 2, self.emb_dims, 1), nn.BatchNorm1d(self.emb_dims),nn.LeakyReLU(negative_slope=0.01),
        )
        self.emb_nn2_tgt = nn.Sequential(
            nn.Conv1d(self.emb_dims1 * 4, self.emb_dims1 * 2, 1), nn.BatchNorm1d(self.emb_dims1 * 2),nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(self.emb_dims1 * 2, self.emb_dims, 1), nn.BatchNorm1d(self.emb_dims),nn.LeakyReLU(negative_slope=0.01),
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
        overlap_points = self.all_points - (self.all_points - self.src_subsampled_points)\
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

        return mask_src, mask_tgt, mask_src_idx, mask_tgt_idx


# # src,tar:[batchsize, 3, num_points]
# src = torch.rand([4, 3, 800])
# tar = torch.rand([4, 3, 768])
# model = OverlapNet()
# mask_src, mask_tgt, mask_src_idx, mask_tgt_idx = model(src, tar)



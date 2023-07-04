import os, sys, glob, h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import minkowski
import open3d as o3d

# (9840, 2048, 3), (9840, 1)
# download in：https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
DATA_DIR = '/home/zy/dataset'
def load_data(partition, file_type='modelnet40'):
    # 读取训练集or测试集
    if file_type == 'Kitti_odo':
        file_name = 'KITTI_odometry/sequences/'
        DATA_FILES = {
            'train': ['00', '01', '02', '03', '04', '05'],  # 0,1,2,3,4,5
            'test': ['08', '09', '10'],  # 8,9,10
        }
        all_data = []
        all_label = []
        for idx in DATA_FILES[partition]:
            for fname in glob.glob(os.path.join(DATA_DIR, file_name, idx, "velodyne", "*.bin")):
                template_data = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
                # print(template_data.shape)
                points = template_data[:4096, :3]
                # points_idx = np.arange(points.shape[0])
                # np.random.shuffle(points_idx)
                # points = points[points_idx[:2048], :]
                all_data.append(points)
                all_label.append(0)
        return np.array(all_data), np.array(all_label)
    elif file_type == 'modelnet40':
        file_name = 'modelnet40_ply_hdf5_2048'
    elif file_type == 'bunny':
        file_name = 'bunny/data/'
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, file_name, '*.ply')):
            pc = o3d.io.read_point_cloud(h5_name)
            points = normalize_pc(np.array(pc.points))
            # 采样10000个点
            points_idx = np.arange(points.shape[0])
            np.random.shuffle(points_idx)
            points = points[points_idx[:4096], :]
            all_data.append(points)

        return np.array(all_data), np.array(all_label)
    else:
        print('Error file name!')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, file_name, 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        if file_name == 'S3DIS_hdf5':
            data = data[:, :, 0:3]
        label = f['label'][:].astype('int64')
        f.close()
        # 取1024个点
        # points_idx = np.arange(data.shape[1])
        # np.random.shuffle(points_idx)
        # data = data[:, points_idx[:1024], :]
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label  # (9840, 2048, 3), (9840, 1)


def normalize_pc(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(point_cloud) ** 2, axis=-1)))
    point_cloud /= furthest_distance
    return point_cloud


def add_outliers(pointcloud, gt_mask):
    # pointcloud: 			Point Cloud (ndarray) [NxC]
    # output: 				Corrupted Point Cloud (ndarray) [(N+300)xC]
    if isinstance(pointcloud, np.ndarray):
        pointcloud = torch.from_numpy(pointcloud)

    num_outliers = 20
    N, C = pointcloud.shape
    outliers = 2*torch.rand(num_outliers, C)-1 					# Sample points in a cube [-0.5, 0.5]
    pointcloud = torch.cat([pointcloud, outliers], dim=0)
    gt_mask = torch.cat([gt_mask, torch.zeros(num_outliers)])

    idx = torch.randperm(pointcloud.shape[0])
    pointcloud, gt_mask = pointcloud[idx], gt_mask[idx]
    return pointcloud.numpy(), gt_mask


# 加入高斯噪声
def jitter_pointcloud(pointcloud, sigma=0.2, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    # pointcloud += sigma * np.random.randn(N, C)
    return pointcloud


def Farthest_Point_Sampling(pointcloud1, src_subsampled_points, tgt_subsampled_points=None):
    # (num_points, 3)
    pointcloud1 = pointcloud1
    num_points = pointcloud1.shape[0]

    if tgt_subsampled_points is None:
        nbrs1 = NearestNeighbors(n_neighbors=src_subsampled_points, algorithm='auto',
                                 metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
        random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]])
        idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((src_subsampled_points,))
        gt_mask_src = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)

        return pointcloud1[idx1, :], gt_mask_src

    else:
        nbrs_src = NearestNeighbors(n_neighbors=src_subsampled_points, algorithm='auto',
                                    metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])

        # 打乱点的顺序
        nbrs_tgt = NearestNeighbors(n_neighbors=tgt_subsampled_points, algorithm='auto',
                                    metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
        random = np.random.random(size=(1, 3))

        random_p1 = random + np.array([[500, 500, 500]])
        src = nbrs_src.kneighbors(random_p1, return_distance=False).reshape((src_subsampled_points,))
        mask_src = torch.zeros(num_points).scatter_(0, torch.tensor(src), 1)  # (src_subsampled_points)
        src = torch.sort(torch.tensor(src))[0]

        random_p2 = random - np.array([[500, 500, 500]])
        tgt = nbrs_tgt.kneighbors(random_p2, return_distance=False).reshape((tgt_subsampled_points,))
        mask_tgt = torch.zeros(num_points).scatter_(0, torch.tensor(tgt), 1)  # (tgt_subsampled_points)
        tgt = torch.sort(torch.tensor(tgt))[0]

        return pointcloud1[src, :], mask_src, pointcloud1[tgt, :], mask_tgt


class ModelNet40_Reg(Dataset):
    def __init__(self, num_points, subsampled_rate_src, subsampled_rate_tgt, partition='train', max_angle=45, max_t=0.5,
                 noise=False, partial_overlap=2, unseen=False, file_type='modelnet40'):
        self.partial_overlap = partial_overlap  # 部分重叠的点云个数：0，1，2
        self.data, self.label = load_data(partition, file_type=file_type)
        self.num_points = num_points
        self.file_type = file_type
        self.partition = partition
        self.label = self.label.squeeze()  # 去掉维度为1的条目
        self.max_angle = np.pi / 180 * max_angle
        self.max_t = max_t
        self.noise = noise
        self.unseen =unseen
        self.subsampled_rate_src = subsampled_rate_src
        self.subsampled_rate_tgt = subsampled_rate_tgt

        if file_type == 'modelnet40' and self.unseen:
            # simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]

        # pointcloud = self.data[item]
        anglex = np.random.uniform(-self.max_angle, self.max_angle)
        angley = np.random.uniform(-self.max_angle, self.max_angle)
        anglez = np.random.uniform(-self.max_angle, self.max_angle)
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        euler_ab = np.asarray([anglez, angley, anglex])
        # 平移矩阵t
        translation_ab = np.array([np.random.uniform(-self.max_t, self.max_t), np.random.uniform(-self.max_t, self.max_t),
                                   np.random.uniform(-self.max_t, self.max_t)])
        # 第item个物体 点云1 [Nx3]
        pointcloud1 = pointcloud

        # 部分重叠
        if self.partial_overlap == 2:
            # (num_points, 3)
            src_subsampled_points = int(self.subsampled_rate_src * pointcloud1.shape[0])
            tgt_subsampled_points = int(self.subsampled_rate_tgt * pointcloud1.shape[0])
            # (num_points, 3)
            pointcloud1, mask_src, pointcloud2, mask_tgt = Farthest_Point_Sampling(
                pointcloud1, src_subsampled_points, tgt_subsampled_points)
            # print("src",torch.unique(mask_src, return_counts=True), pointcloud1.shape)
            # print("tgt",torch.unique(mask_tgt, return_counts=True), pointcloud2.shape)

            gt_mask_src = []
            gt_mask_tgt = []
            for i in range(pointcloud.shape[0]):
                if mask_src[i] == 1 and mask_tgt[i] == 1:
                    gt_mask_src.append(1)
                    gt_mask_tgt.append(1)
                elif mask_src[i] == 1 and mask_tgt[i] == 0:
                    gt_mask_src.append(0)
                elif mask_src[i] == 0 and mask_tgt[i] == 1:
                    gt_mask_tgt.append(0)

            gt_mask_src = torch.Tensor(gt_mask_src)
            gt_mask_tgt = torch.Tensor(gt_mask_tgt)

            pointcloud2 = rotation_ab.apply(pointcloud2).T + np.expand_dims(translation_ab, axis=1)
            # 打乱点的顺序
            state = np.random.get_state()
            pointcloud1 = np.random.permutation(pointcloud1).T
            np.random.set_state(state)
            gt_mask_src = np.random.permutation(gt_mask_src).T

            if self.noise:
                # ---加入噪声---
                # (num_points, 3)
                pointcloud2 = jitter_pointcloud(pointcloud2.T)
                pointcloud2 = pointcloud2.T

            return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
                   translation_ab.astype('float32'), euler_ab.astype('float32'), gt_mask_src, gt_mask_tgt

        else:
            raise ValueError('partial_overlap must be 2!')

    def __len__(self):
        return self.data.shape[0]


# data, label = load_data('train', file_type='Kitti_odo')
# print(data.shape)


# RORNet: Partial-to-Partial Registration Network With Reliable Overlapping Representations

by Yue Wu , Yue Zhang , Wenping Ma , Maoguo Gong , Xiaolong Fan , Mingyang Zhang , A. K. Qin , and Qiguang Miao, and details are in [paper](https://ieeexplore.ieee.org/document/10168979).

## Usage

1. Clone the repository.

2. Change the "DATA_DIR" parameter in the "data_utils.py" file to its own data set folder path.

3. Run the "main.py" file.

## Requirement

​	h5py=3.7.0

​	open3d=0.15.2

​	pytorch=1.11.0

​	scikit-learn=1.1.1

​	transforms3d=0.4.1

​	tensorboardX=1.15.0

​	tqdm

​	numpy

## Dataset

​		(1) [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)

​		(2) [KITTI_odo](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

​		(3) [Stanford Bunny](http://graphics.stanford.edu/data/3Dscanrep/)

## Citation

If you find the code or trained models useful, please consider citing:

```
@article{2023rornet,
  title={RORNet: Partial-to-Partial Registration Network With Reliable Overlapping Representations},
  author={Wu, Yue and Zhang, Yue and Ma, Wenping and Gong, Maoguo and Fan, Xiaolong and Zhang, Mingyang and Qin, AK and Miao, Qiguang},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgement

Our code refers to [PointNet](https://github.com/fxia22/pointnet.pytorch), [DCP](https://github.com/WangYueFt/dcp) and [MaskNet](https://github.com/vinits5/masknet). We want to thank the above open-source projects.

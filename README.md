# RORNet: Partial-to-Partial Registration Network With Reliable Overlapping Representations

by Yue Wu , Yue Zhang , Wenping Ma , Maoguo Gong , Xiaolong Fan , Mingyang Zhang , A. K. Qin , and Qiguang Miao, and details are in [paper](https://ieeexplore.ieee.org/document/10168979).

## Usage

1. Clone the repository.

2. Change the "DATA_DIR" parameter in the "data_utils.py" file to its own data set folder path.

3. Run the "main.py" in OverlapDetect file and save the pkl file; load pkl file trained by OverlapDetect file and run the OverlapReg file. 
**Note**: you need to make the "OverlapNet" model consistent for the OverlapDetect file and the OverlapReg file.

4. For convenience, We provide end-to-end training "running OverlapReg/main.py directly", but there may be a loss of accuracy.

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
@article{{rornet,
	author={Wu, Yue and Zhang, Yue and Ma, Wenping and Gong, Maoguo and Fan, Xiaolong and Zhang, Mingyang and Qin, A. K. and Miao, Qiguang},
	journal={IEEE Transactions on Neural Networks and Learning Systems}, 
	title={RORNet: Partial-to-Partial Registration Network With Reliable Overlapping Representations}, 
	year={2024},
	volume={35},
	number={11},
	pages={15453-15466},
	doi={10.1109/TNNLS.2023.3286943}
}
```

## Acknowledgement

Our code refers to [PointNet](https://github.com/fxia22/pointnet.pytorch), [DCP](https://github.com/WangYueFt/dcp) and [MaskNet](https://github.com/vinits5/masknet). We want to thank the above open-source projects.

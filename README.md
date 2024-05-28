# Guidance to Use the Codes of LFC-Siamese-Matching
## 1.Recommended Environment: 
```
Python 3.7
Pytorch 1.7.0
torchvision 0.8.0
numpy 1.21.5
h5py 2.9.0
```
## 2.Data preparation: 
Download the YFCC100M dataset and the SUN3D dataset from the [OANet](https://github.com/zjhthu/OANet) repository. Create a folder called "data" and place the obtained dataset here

## 3.Pretrained model:  
Pretrained models have been placed in the folder of this project, you can easily find them

## 4.Training:
If you want to retrain the model, please delete the folder named "log" first, and then execute the following commands(Taking msa_lfc_yfcc as an example):
```
cd msa_lfc_yfcc
python main.py --data_tr='../data/yfcc-sift-2000-train.hdf5' --data_va='../data/yfcc-sift-2000-val.hdf5' --run_mode='train'
```

## 5.Testing:
You can find the pretrained models in a folder called "log" in this project , test the model according to the following commands(Taking msa_lfc_yfcc as an example):
```
cd msa_lfc_yfcc
python main.py --use_ransac=True --data_te='../data/yfcc-sift-2000-val.hdf5' --run_mode='test'  --gpu_id=0
python main.py --use_ransac=False --data_te='../data/yfcc-sift-2000-val.hdf5' --run_mode='test' --gpu_id=0
mkdir ./log/main.py/test/known
mv ./log/main.py/test/*txt ./log/main.py/test/known

python main.py --use_ransac=True --data_te='../data/yfcc-sift-2000-test.hdf5' --run_mode='test' --gpu_id=0
python main.py --use_ransac=False --data_te='../data/yfcc-sift-2000-test.hdf5' --run_mode='test' --gpu_id=0
mkdir ./log/main.py/test/unknown
mv ./log/main.py/test/*txt ./log/main.py/test/unknown
```
## 6.Reference:
```
@inproceedings{wang2023local,
  title={Local Consensus Enhanced Siamese Network with Reciprocal Loss for Two-view Correspondence Learning},
  author={Wang, Linbo and Wu, Jing and Fang, Xianyong and Liu, Zhengyi and Cao, Chenjie and Fu, Yanwei},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={5235--5243},
  year={2023}
}
```

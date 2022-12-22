PyTorch code for the paper "RT3D: Achieving Real-Time Execution of 3D Convolutional Neural Networks on Mobile Devices" accepted by AAAI 2021.

- Pruning sparisity types include "vanilla" and "kernel group sparsity" in weight blocks.
- Networks include C3D, R(2+1)D (from torchvision), and S3D.

---

- Install Anaconda
- Install dependencies:
    Build a file named "conda_requirements.txt" in your work directory, then write the following dependencies in your file:
    ```sh
    numpy
    pytorch>=1.2.0
    ffmpeg==4.0
    torchvision
    pillow==6.2.1
    pyyaml
    python-lmdb
    pyarrow==0.11.1
    tqdm
    ```
    Build a file named "pip_requirements.txt" with:
    ```sh
    sk-video
    ```
    Then, run  `pip install -r pip_requirements.txt`, and `conda install --file conda_requirements.txt`
- Download datasets: UCF101. Converting into LMDB files is recommended.
- Pretrained should be in `checkpoint` folder, and pruned models will be saved into this folder.
- Run pruning code by running scripts in `scripts` folder: e.g., `bash scripts/blk-kgs_c3d.sh`

# Baseline Models

Download the unpruned baseline models and put them in `./checkpoint` folder in your work directory.
- C3D:
  - UCF101: [OneDrive](https://1drv.ms/u/s!Ak42T1bb_EnDccNcAatzOCx7W6s), [Baidu Netdisk](https://pan.baidu.com/s/1GQcIbxAp-W3KP3PQ77IgRQ) (Password: qjzz)
  - HMDB51: [OneDrive](https://1drv.ms/u/s!Ak42T1bb_EnDcKH7lmo8u20HC1Q), [Baidu Netdisk](https://pan.baidu.com/s/1X3y-JLFJiOAe1-5Dor_Mbw) (Password: 0ijw)
- R2Plus1D:
  - UCF101: [OneDrive](https://1drv.ms/u/s!Ak42T1bb_EnDbV0RVjlROM14jKE), [Baidu Netdisk](https://pan.baidu.com/s/1OFunz8QRAoPG62vY9guigg) (Password: agm1)
  - HMDB51: [OneDrive](https://1drv.ms/u/s!Ak42T1bb_EnDbH3aMVAW5m--GX0), [Baidu Netdisk](https://pan.baidu.com/s/1m2yssTWf5mVf2h4eErrZ1g) (Password: gvlz)
- S3D:
  - UCF101: [OneDrive](https://1drv.ms/u/s!Ak42T1bb_EnDb4TwsqFPcK9sgbg), [Baidu Netdist](https://pan.baidu.com/s/14rdPLK5j3rQ2y9Ms1QvO1A) (Password: f49l)
  - HMDB51: [OneDrive](https://1drv.ms/u/s!Ak42T1bb_EnDbrXgI3WmQWHXxgs), [Baidu Netdisk](https://pan.baidu.com/s/1E_ybo-38lCpBHTz5hGvFNQ) (Password: 4ggq)


# Datasets

Download and process the dataset with the following commands in `./dataset` in your work directory.
<!-- - (only for reference) https://github.com/chaoyuaw/pytorch-coviar/blob/master/GETTING_STARTED.md -->

## UCF101

- Download UCF101 original video dataset from https://www.crcv.ucf.edu/data/UCF101/UCF101.rar and unpack to `ucf101` folder. Download the train/test splits for action recognition from https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip and unpack.
  - To unpack `rar` files, install `unrar` by running `sudo apt install unrar`.
```sh
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
mkdir ucf101 && unrar e UCF101.rar ucf101
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate && unzip UCF101TrainTestSplits-RecognitionTask.zip
```
- Extract frames from videos by running `python extract_videos.py -d ucf101`.
- Convert frames into LMDB files with `create_lmdb.py`.
```sh
python create_lmdb.py -d ucf101_frame -s train -vr 0 10000
python create_lmdb.py -d ucf101_frame -s val -vr 0 4000
```
```sh
rt3d-pruning/dataset/ucf101_frame/
├── v_ApplyEyeMakeup_g01_c01
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   ├── 00004.jpg
│   ├── 00005.jpg
```

The corresponding python file `extract_videos.py` and `create_lmdb.py` can be downloaded at
https://drive.google.com/drive/folders/1nPWDAvlpOOpNGbVMJUGM-e9uJlbIO4ey?usp=sharing

#  Training with pruning 

Training with ADMM:
```sh
python opt_main.py --data_location /data/UCF101/ --admm  --epoch 50  --arch r2+1d  --config-file  r2+1d-pretrained_3.20x
```
You can find other arguments in options.py file. The above command specifies important arguments such as architecture (--arch r2+1d) and admm training (--admm). 

Finetuning after ADMM pruning:
```sh
python opt_main.py --data_location /data/UCF101/  --masked-retrain  --epoch 50  --arch r2+1d  --config-file  r2+1d-pretrained_3.20x --combine-progressive  --warmup
```
You can find other arguments in options.py file. The above command specifies important arguments such as architecture (--arch r2+1d) and mask retraining (--masked-retrain). 




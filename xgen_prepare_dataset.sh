#!/bin/bash

set -e
data_path="/data/video-classification-s2-1d"
mkdir ${data_path} && cd ${data_path}

wget https://xgen.oss-cn-hongkong.aliyuncs.com/data/video-classification-s2-1d/UCF101.rar
wget https://xgen.oss-cn-hongkong.aliyuncs.com/data/video-classification-s2-1d/UCF101TrainTestSplits-RecognitionTask.zip

mkdir ucf101 && unrar e UCF101.rar ucf101
unzip UCF101TrainTestSplits-RecognitionTask.zip

python /root/Projects/video-classification-s2-1d/extract_videos.py -d ucf101
python /root/Projects/video-classification-s2-1d/create_lmdb.py -d ucf101_frame -s train -vr 0 10000
python /root/Projects/video-classification-s2-1d/create_lmdb.py -d ucf101_frame -s val -vr 0 4000

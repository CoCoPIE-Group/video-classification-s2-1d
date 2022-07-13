#!/bin/bash

set -e
data_path="/data/video-classification-s2-1d"
mkdir ${data_path} && cd ${data_path}

wget https://xgen.oss-cn-hongkong.aliyuncs.com/data/video-classification-s2-1d/UCF101.rar
wget https://xgen.oss-cn-hongkong.aliyuncs.com/data/video-classification-s2-1d/UCF101TrainTestSplits-RecognitionTask.zip

mkdir ucf101 && unrar e UCF101.rar ucf101
unzip UCF101TrainTestSplits-RecognitionTask.zip


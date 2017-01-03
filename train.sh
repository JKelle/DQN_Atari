#!/bin/bash

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/vision/vision_users/jkelle/tensorflow/cudatoolkit/lib64:/vision/vision_users/jkelle/tensorflow/cudatoolkit/extras/CUPTI/lib64:/vision/vision_users/jkelle/tensorflow/cudnndownload/cuda/lib64/"
export CUDA_HOME=/opt/cuda-8.0

/vision/vision_users/jkelle/tensorflow/bin/python train.py

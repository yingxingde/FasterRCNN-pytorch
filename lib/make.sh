#!/usr/bin/env bash

cd roi_pooling/src/cuda
echo "Compiling roi_pooling kernels by nvcc..."
nvcc -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../../
python build.py
cd ../

cd nms/src/cuda
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../../
python build.py
cd ../

CUDA_PATH=/usr/local/cuda/
python setup.py build_ext --inplace
rm -rf build
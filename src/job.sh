#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=MultiGPU
#SBATCH -D .
#SBATCH --output=submit-STREAMS.o%j
#SBATCH --error=submit-STREAMS.e%j
#SBATCH -A cuda
#SBATCH -p cuda
### Se piden 4 GPUs 
#SBATCH --gres=gpu:4

export PATH=/Soft/cuda/12.2.2/bin:$PATH


#./kernel00.exe 10000 Y

#./kernel01.exe 10000 Y

#./kernelSIM.exe

./partitionGPUpinned.exe 



#nsys nvprof --print-gpu-trace ./kernel00.exe 300000 N
#nsys nvprof --print-gpu-trace ./kernel01.exe 300000 N
#nsys nvprof --print-gpu-trace  ./kernelSIM.exe
#nsys nvprof --print-gpu-trace  ./partitionGPU.exe



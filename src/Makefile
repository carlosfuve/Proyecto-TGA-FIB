CUDA_HOME   = /Soft/cuda/12.2.2

NVCC        = $(CUDA_HOME)/bin/nvcc
ARCH        = -gencode arch=compute_86,code=sm_86
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudadevrt -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXEDYN	        = partitionGPUpinned.exe
OBJDYN	        = partitionGPUpinned.o


default: $(EXEDYN)

partitionGPUpinned.o: partitionGPUpinned.cu
	$(NVCC) $(ARCH) -dc partitionGPUpinned.cu -o $@ 

$(EXEDYN): $(OBJDYN)
	$(NVCC)  $(ARCH) -rdc=true $(OBJDYN)  $(LD_FLAGS) -o $(EXEDYN)

all:	 $(EXEDYN) 

clean:
	rm -rf *.o kernel*.exe


GCC = gcc

#OpenCL_SDK=/global/homes/s/sgrauerg/NVIDIA_GPU_Computing_SDK
#INCLUDE=-I/global/homes/s/sgrauerg/lifan/OpenCL
#LIBPATH=-L${OpenCL_SDK}/OpenCL/common/lib -L${OpenCL_SDK}/shared/lib
LIB=-lOpenCL -lm -O3         #make sure OpenCL library is correctly installed

FILES = brdr2d.c 

all: brdr2d_gpu

brdr2d_gpu:
	${GCC}  ${LIBPATH} ${INCLUDE} ${LIB} -o brdr2d.exe ${FILES}

clean:
	rm -f *~ *.exe



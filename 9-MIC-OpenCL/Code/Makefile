GCC = gcc

#OpenCL_SDK=
#INCLUDE=-I/usr/local/stow/cuda_5.0.35/include/
#LIBPATH=-L${OpenCL_SDK}/OpenCL/common/lib -L${OpenCL_SDK}/shared/lib
LIB= -O3 -lOpenCL -lm

FILES = brdr2d.c 

all: brdr2d_gpu

brdr2d_gpu:
	${GCC}  ${LIB} -o brdr2d.exe ${FILES}

clean:
	rm -f *~ *.exe


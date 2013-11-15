GCC = nvcc
INCLUDE = -I./inc/ -I./


#LIBS = -lcutil
FLAGS = -arch sm_20 -O3  
FILES = brdr2d.cu 
all: brdr2d_gpu

brdr2d_gpu:
	${GCC} ${FLAGS} ${INCLUDE} -o brdr2d.exe ${FILES}

clean:
	rm -f *~ *.exe



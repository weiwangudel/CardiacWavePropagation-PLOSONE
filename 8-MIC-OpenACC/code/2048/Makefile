all:
	gcc -O2 brdr2d.c -lm  -o brdr2d_seq.exe
acc_opencl:
	rm *hmp*  &
	hmpp --openacc-target=OPENCL gcc -O3 -lOpenCL -o brdr2d_acccl.exe brdr2d.c 
acc_cuda:
	rm *hmp* &
	hmpp --openacc-target=CUDA gcc -O3 -o brdr2d_acccuda.exe brdr2d.c
run:
	./brdr2d.exe
clean:
	rm brdr2d.exe

#include <assert.h>
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/
/*********************************************/
// datarr columns:  
// datarr[][][0][] is Vm       (mV)
// datarr[][][1][] is dVmdt    (mV/msec)
// datarr[][][2][] is IK1      (uA/cm^2)
// datarr[][][3][] is Ix1      (uA/cm^2)
// datarr[][][4][] is x1       (unitless)
// datarr[][][5][] is INa      (uA/cm^2)
// datarr[][][6][] is m        (unitless)
// datarr[][][7][] is h        (unitless)
// datarr[][][8][] is Is       (uA/cm^2)
// datarr[][][9][] is d       (unitless)
// datarr[][][10][] is f       (unitless)
// datarr[][][11][] is Cai     (mole/L)
// datarr[][][12][] is Isum    (uA/cm^2)
// datarr[][][13][] is Diff    (mV/msec) 
// datarr[][][14][] is Istim   (uA/cm^2)  Istim should always be the last variable in datarr
/*********************************************/
// derivarr columns: 
// derivarr[0] is current time  (msec)
// derivarr[1] is dx1dt         (unitless)
// derivarr[2] is dmdt          (unitless)
// derivarr[3] is dhdt          (unitless)
// derivarr[4] is dddt          (unitless)
// derivarr[5] is dfdt          (unitless)
// derivarr[6] is dCaidt        (mole/L)
/*********************************************/
// Constants: 
// constarr[0] is gK1   (mmho/cm^2)
// constarr[1] is gNa   (mmho/cm^2)
// constarr[2] is ENa   (mV)  
// constarr[3] is gx1   (mmho/cm^2)
// constarr[4] is gs    (mmho/cm^2)
// constarr[5] is Cm    (uF/cm^2)
// constarr[6] is kCa   (msec^-1)
// constarr[7] is gNaC  (mmho/cm^2)     /* should be set to zero in brdr2dtask.dat */
// constarr[8] is Dpara   (cm^2/msec)
// constarr[9] is Dperpen (cm^2/msec)
// constarr[10] is theta  (degrees)
// constarr[11] is sigma  (unitless)
// constarr[12] is A      (unitless)
/*********************************************/
// Diffusion Tensor:  note-> D12=D21
// D[0][0] is D11     (cm^2/msec)
// D[0][1] is D12     (cm^2/msec)
// D[1][0] is D21     (cm^2/msec)
// D[1][1] is D22     (cm^2/msec)
/*********************************************/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
//#include <sys/dir.h>  use to check for and/or create data directory?
#include "brdr2d.h"
#include "brdr2dinout.c"

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void invokeGPU(int argc, char** argv);
void GPU_Mem_init(void);
void opencl_init();

//extern "C"
//void computeGold( float* reference, float* idata, const unsigned int len);
void initialize();
void openfiles();
void buildedgestim();
void buildptstim();
void build2ptstims();
void buildbarstim1();
void buildbarstim2();
void buildcrossstim();
void stimulate();
void blockonoff();
void brgates();
void brcurrents();
void bcs();
void output();
void closefiles();
void brfc();
void readbrfc();
void stability();
double rtclock();


const int VAR_N = 1;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	printf("Initializing ... \n");
	initialize();
	if (stimnum > 0)
	{
		printf("Building stimulus matrix ... \n");
		buildptstim();  
	}
	printf("Opening files ... \n");
	openfiles();

	step=0;		/* output function calls step */
  	printf("Writing initial conditions ... \n");
 	output();


	time_t ttime=time(0);            // Get current time
	
  	char *stime=ctime(&ttime);
	//printf("%s",stime);
	
	invokeGPU(argc, argv);

	printf("Saving final conditions...\n\n");

	brfc();
 	printf("         tfinal: %5.3f msec\n",tfinal);
  	printf("     Final time: %5.3f msec\n",derivarr[0]);
  	printf("         Nsteps: %10.2f\n",Nsteps);
  	printf("Number of steps: %d\n",step);
  	printf("             Nx: %d\n",Nx);
  	printf("             Ny: %d\n",Ny);

  	//ttime=time(0);                 // Get current time
  	//stime=ctime(&ttime);
  	//printf("%s\n",stime);
    
  	closefiles();
}

void
invokeGPU(int argc, char *argv[])
{
	int i,j,k,l;

	//initialize OpenCL platform and kernels
	opencl_init();

	//allocate GPU memory, copy data from host to device	
	GPU_Mem_init();	
	
	//unsigned int timer = 0;
  	printf("Entering time loop ... \n");
	step = 1;
	derivarr[0] += dt;
	deriv3darr[0][0][0] += dt;  // update time (msec)

	errcode = clEnqueueWriteBuffer(clCommandQue, d_derivarr, CL_TRUE, 0, sizeof(double), deriv3darr[0][0], 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
 
	//cutilSafeCall(cudaMemcpy(d_derivarr,deriv3darr[0][0],sizeof(double), cudaMemcpyHostToDevice));			

	// setup execution parameters
	THREAD_DIMX = atoi(argv[1]);
	THREAD_DIMY = atoi(argv[2]);
	if(Nx%THREAD_DIMX != 0){
		printf("Nx is %d, Thread_Dimx is %d, Nx % Thread_Dimx != 0 return\n",Nx,THREAD_DIMX); 
		return;
	}
	else BLOCK_DIMX = Nx/THREAD_DIMX;
	if(Ny%THREAD_DIMY != 0){
		printf("Ny is %d, Thread_Dimy is %d, Ny % Thread_Dimy != 0 return\n",Ny,THREAD_DIMY);
		return;
	}
	else BLOCK_DIMY = Ny/THREAD_DIMY;
	//dim3 dimGrid(BLOCK_DIMX,BLOCK_DIMY,1);
	//dim3 dimBlock(THREAD_DIMX,THREAD_DIMY,1);

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = THREAD_DIMX;
	localWorkSize[1] = THREAD_DIMY;
	globalWorkSize[0] = Nx;
	globalWorkSize[1] = Ny;

	//cudaPrintfInit();
	double gpu_start = rtclock();
	double stim_time=0;
	double block_time=0;
	double cur_time=0;
	double gate_time=0;
	double bcs_time=0;
	double atomic_time=0;
	double mem_time=0;
	double time_temp;
	while (derivarr[0] <= tfinal+dt && step <= Nsteps + 1)
	{
		// from (1 to Nx) instead of (0 to Nx+1)
		// do not loop through ghost points */
		//GPU Kernel Execution
		if(stimnum>0)
		{	time_temp = rtclock();
			// Set the arguments of the kernel
			errcode =  clSetKernelArg(clKernel_stimulate, 0, sizeof(int), (void *)&stimnum);
			errcode |= clSetKernelArg(clKernel_stimulate, 1, sizeof(cl_mem), (void *)&d_datarr);
			errcode |= clSetKernelArg(clKernel_stimulate, 2, sizeof(cl_mem), (void *)&d_stimarr);
			errcode |= clSetKernelArg(clKernel_stimulate, 3, sizeof(cl_mem), (void *)&d_derivarr);
			errcode |= clSetKernelArg(clKernel_stimulate, 4, sizeof(int), (void *)&varnum);
			errcode |= clSetKernelArg(clKernel_stimulate, 5, sizeof(int), (void *)&step);
			errcode |= clSetKernelArg(clKernel_stimulate, 6, sizeof(double), (void *)&Istimamp);
			errcode |= clSetKernelArg(clKernel_stimulate, 7, sizeof(int), (void *)&Nx);
			errcode |= clSetKernelArg(clKernel_stimulate, 8, sizeof(int), (void *)&Ny);
			errcode |= clSetKernelArg(clKernel_stimulate, 9, sizeof(double), (void *)&stimint);
			if(errcode != CL_SUCCESS) printf("Error in seting arguments for kernel stimulate\n");
			// Execute the OpenCL kernel
			errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_stimulate, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
			if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
			errcode = clFinish(clCommandQue);
			stim_time += (double)(rtclock()-time_temp);
		}
		
		if(blocktimenum>0)
		{
			time_temp = rtclock();
			errcode =  clSetKernelArg(clKernel_blockonoff, 0, sizeof(int), (void *)&blocktimenum);
			errcode |= clSetKernelArg(clKernel_blockonoff, 1, sizeof(cl_mem), (void *)&d_derivarr);
			errcode |= clSetKernelArg(clKernel_blockonoff, 2, sizeof(cl_mem), (void *)&d_blocktimes);
			errcode |= clSetKernelArg(clKernel_blockonoff, 3, sizeof(cl_mem), (void *)&d_block);
			errcode |= clSetKernelArg(clKernel_blockonoff, 4, sizeof(int), (void *)&Nx);
			errcode |= clSetKernelArg(clKernel_blockonoff, 5, sizeof(int), (void *)&Ny);
			if(errcode != CL_SUCCESS) printf("Error in seting arguments for kernel blockonoff\n");
			// Execute the OpenCL kernel
			errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_blockonoff, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
			if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
			errcode = clFinish(clCommandQue);
			block_time += (double)(rtclock()-time_temp);
		}
		
		time_temp = rtclock();
		errcode =  clSetKernelArg(clKernel_brgates, 0, sizeof(int), (void *)&varnum);
		errcode |= clSetKernelArg(clKernel_brgates, 1, sizeof(cl_mem), (void *)&d_datarr);
		errcode |= clSetKernelArg(clKernel_brgates, 2, sizeof(cl_mem), (void *)&d_derivarr);
		errcode |= clSetKernelArg(clKernel_brgates, 3, sizeof(cl_mem), (void *)&d_constarr);
		errcode |= clSetKernelArg(clKernel_brgates, 4, sizeof(int), (void *)&step);
		errcode |= clSetKernelArg(clKernel_brgates, 5, sizeof(int), (void *)&Nx);
		errcode |= clSetKernelArg(clKernel_brgates, 6, sizeof(int), (void *)&Ny);
		if(errcode != CL_SUCCESS) printf("Error in seting arguments for kernel brgates\n");
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_brgates, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
		errcode = clFinish(clCommandQue);
		gate_time += (double)(rtclock()-time_temp);
		

		time_temp = rtclock();
		errcode =  clSetKernelArg(clKernel_brcurrents, 0, sizeof(int), (void *)&stimnum);
		errcode |= clSetKernelArg(clKernel_brcurrents, 1, sizeof(cl_mem), (void *)&d_datarr);
		errcode |= clSetKernelArg(clKernel_brcurrents, 2, sizeof(cl_mem), (void *)&d_derivarr);
		errcode |= clSetKernelArg(clKernel_brcurrents, 3, sizeof(int), (void *)&step);
		errcode |= clSetKernelArg(clKernel_brcurrents, 4, sizeof(double), (void *)&Istimamp);
		errcode |= clSetKernelArg(clKernel_brcurrents, 5, sizeof(int), (void *)&Nx);
		errcode |= clSetKernelArg(clKernel_brcurrents, 6, sizeof(int), (void *)&Ny);
		errcode |= clSetKernelArg(clKernel_brcurrents, 7, sizeof(int), (void *)&varnum);
		errcode |= clSetKernelArg(clKernel_brcurrents, 8, sizeof(cl_mem), (void *)&d_constarr);
		errcode |= clSetKernelArg(clKernel_brcurrents, 9, sizeof(cl_mem), (void *)&d_Afield);
		errcode |= clSetKernelArg(clKernel_brcurrents, 10, sizeof(cl_mem), (void *)&d_block);
		errcode |= clSetKernelArg(clKernel_brcurrents, 11, sizeof(cl_mem), (void *)&d_Dp);
		errcode |= clSetKernelArg(clKernel_brcurrents, 12, sizeof(double), (void *)&dt);
		if(errcode != CL_SUCCESS) printf("Error in seting arguments for kernel brcurrents\n");
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_brcurrents, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
		errcode = clFinish(clCommandQue);
		cur_time += (double)(rtclock()-time_temp);
		
		
		size_t globalItemSize = 1;
		size_t localItemSize = 1;
		time_temp = rtclock();
		errcode =  clSetKernelArg(clKernel_bcs, 0, sizeof(double), (void *)&dx);
		errcode |= clSetKernelArg(clKernel_bcs, 1, sizeof(double), (void *)&dy);
		errcode |= clSetKernelArg(clKernel_bcs, 2, sizeof(cl_mem), (void *)&d_D);
		errcode |= clSetKernelArg(clKernel_bcs, 3, sizeof(int), (void *)&BC);
		errcode |= clSetKernelArg(clKernel_bcs, 4, sizeof(int), (void *)&step);
		errcode |= clSetKernelArg(clKernel_bcs, 5, sizeof(int), (void *)&Nx);
		errcode |= clSetKernelArg(clKernel_bcs, 6, sizeof(int), (void *)&Ny);
		errcode |= clSetKernelArg(clKernel_bcs, 7, sizeof(int), (void *)&varnum);
		errcode |= clSetKernelArg(clKernel_bcs, 8, sizeof(cl_mem), (void *)&d_Dp);
		errcode |= clSetKernelArg(clKernel_bcs, 9, sizeof(cl_mem), (void *)&d_datarr);
		errcode |= clSetKernelArg(clKernel_bcs, 10, sizeof(cl_mem), (void *)&d_derivarr);
		errcode |= clSetKernelArg(clKernel_bcs, 11, sizeof(double), (void *)&dt);
		if(errcode != CL_SUCCESS) printf("Error in seting arguments for kernel bcs\n");
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_bcs, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
		errcode = clFinish(clCommandQue);
		bcs_time += (double)(rtclock()-time_temp);

        time_temp = rtclock();
		errcode = clSetKernelArg(clKernel_update, 0, sizeof(int), (void *)&step);
		errcode = clSetKernelArg(clKernel_update, 1, sizeof(int), (void *)&varnum);
		errcode = clSetKernelArg(clKernel_update, 2, sizeof(int), (void *)&Nx);
		errcode = clSetKernelArg(clKernel_update, 3, sizeof(int), (void *)&Ny);
		errcode = clSetKernelArg(clKernel_update, 4, sizeof(double), (void *)&Dp[0][0]);
		errcode = clSetKernelArg(clKernel_update, 5, sizeof(double), (void *)&Dp[0][1]);
		errcode = clSetKernelArg(clKernel_update, 6, sizeof(double), (void *)&Dp[1][0]);
		errcode = clSetKernelArg(clKernel_update, 7, sizeof(double), (void *)&Dp[1][1]);
		errcode = clSetKernelArg(clKernel_update, 8, sizeof(cl_mem), (void *)&d_datarr);
		if(errcode != CL_SUCCESS){ printf("Error in seting arguments for kernel update\n"); exit(1);}
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_update, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS){ printf("Error in launching kernel update\n");	exit(1);}	
		errcode = clFinish(clCommandQue);		
		atomic_time += (double)(rtclock()-time_temp);
		
		time_temp = rtclock();
		if (step % rpN == 0) {
			// Coalescing cudaMemcpy
			//cutilSafeCall(cudaMemcpy(linear_datarr,d_datarr,(Nx+2)*(Ny+2)*varnum*2*sizeof(double),cudaMemcpyDeviceToHost));
			errcode = clEnqueueReadBuffer(clCommandQue, d_datarr, CL_TRUE, 0, (Nx+2)*(Ny+2)*varnum*2*sizeof(double), linear_datarr, 0, NULL, NULL);

			// copy host memory to device
			for (l = 0; l < 2; l++)
			{
				for (k = 0; k < varnum; k++)
				{
					for (i = 0; i < (Nx+2); i++)
					{
						for (j = 0; j < (Ny+2); j++)
						{
							datarr[l][k][i][j] = 
						     	*(linear_datarr+
							l*(Nx+2)*(Ny+2)*varnum+
							k*(Nx+2)*(Ny+2)+
							i*(Ny+2)+
							j);
						}
					}
				}
			}

		       output();       
	  
			printf("%4.4e msec, Vm(%d,%d): %3.2f mV GPU\n",
				derivarr[0], mNx, mNy, datarr[step%2][0][mNx][mNy]);
		}
		mem_time += (double)(rtclock()-time_temp);
		step++;
		
		derivarr[0] += dt;
		deriv3darr[0][0][0] += dt;  // update time (msec) 
		
     	
	}
	double gpu_end = rtclock();    
	
	printf("total         time is %.2lf\n",(double)(gpu_end-gpu_start));	
	printf("Kernel stim   time is %.2lf\n",stim_time);
	printf("Kernel block  time is %.2lf\n",block_time);
	printf("Kernel gate   time is %.2lf\n",gate_time);
	printf("Kernel cur    time is %.2lf\n",cur_time);
	printf("Kernel atomic time is %.2lf\n",atomic_time);
	printf("Kernel bcs    time is %.2lf\n",bcs_time);
	printf("memory copy   time is %.2lf\n",mem_time);
	printf("GPU           time is %.2lf\n",stim_time+block_time+gate_time+cur_time+atomic_time+bcs_time);				   
	
}

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void buildptstim(void){
/* point stimulus  */
  int i,j,k;
  int Nxx,Nyy;
  double stimsizeir;
  double radius;

  Nxx=(unsigned int)(floor(Nx/2));
  Nyy=(unsigned int)(floor(Ny/2));
  printf("Point stimulus centered at %d,%d\n",Nxx,Nyy);
  stimsizeir=floor(stimsize1/dx);
  printf("Point stimulus radius: %4.3f cm, %4.3f pixels\n",stimsize1,stimsizeir);
  for (k=0;k<stimnum;++k){
    for (i=0;i<=Nx;++i){
      for (j=0;j<=Ny;++j){
        radius=sqrt(((double)(Nxx-i))*((double)(Nxx-i)) + ((double)(Nyy-j))*((double)(Nyy-j)));
    if (radius<=stimsizeir){
          stimarr[i][j][k][0]=stimes[k];
    }
      }
    }
  }
}

void opencl_init(void)
{
	// Load the kernel source code into the array source_str
	int i;
	fp = fopen("brdr2d.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );

	// Get platform and device information
	errcode = clGetPlatformIDs(0, NULL, &num_platforms);
	if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);

	
	if (0 < num_platforms) 
	{
		platform_id = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
		errcode = clGetPlatformIDs(num_platforms, platform_id, NULL);
		if(errcode != CL_SUCCESS) printf("error in getting platform id\n");

		for (i = 0; i < num_platforms; ++i) 
		{

			errcode = clGetPlatformInfo(platform_id[i],CL_PLATFORM_VENDOR,sizeof(str_temp), str_temp, NULL);

			if(errcode != CL_SUCCESS) printf("error in getting platform name\n");
			if(errcode == CL_SUCCESS) printf("platform %d vendor is %s\n",i,str_temp);
			


			if (strcmp(str_temp, "NVIDIA Corporation")!=0) 
			  {
			    printf("Platform is not nVidia, check next platform\n");
			  }
			else{
			  errcode = clGetPlatformInfo(platform_id[i],CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
			  if(errcode == CL_SUCCESS) printf("platform %d name is %s\n",i,str_temp);
			  errcode = clGetDeviceIDs( platform_id[i], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
			  if(errcode == CL_SUCCESS) printf("device id is %d\n",device_id);
			  break;
			}
		}


        free(platform_id);
	}

	errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("device name is %s\n",str_temp);
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context %d\n",errcode);
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");

	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	printf("errorcode=%d\n", errcode);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the OpenCL kernel
	clKernel_stimulate = clCreateKernel(clProgram, "d_stimulate_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	// Create the OpenCL kernel
	clKernel_blockonoff = clCreateKernel(clProgram, "d_blockonoff_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	// Create the OpenCL kernel
	clKernel_brgates = clCreateKernel(clProgram, "d_brgates_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	// Create the OpenCL kernel
	clKernel_brcurrents = clCreateKernel(clProgram, "d_brcurrents_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	// Create the OpenCL kernel
	clKernel_bcs = clCreateKernel(clProgram, "kernel_call_device_bcs", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	// Create the OpenCL kernel
	clKernel_update = clCreateKernel(clProgram, "NinePointLaplacian", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");

	
	
}

void GPU_Mem_init()
{
	int i, j, k,l;		// loop index
	long int xyzw_size;
	long int xyzw_stim_size;
	long int xyz_deriv_size;
	
	/*// Use device with highest Gflops/s
	cudaSetDevice( cutGetMaxGflopsDeviceId() );
	
	int* d_varnum; 
	cudaMalloc((void **)&d_varnum, size_int);
	cudaMemcpy(d_varnum, &varnum, size_int, cudaMemcpyHostToDevice);

	int* d_step;    
	cudaMalloc((void **)&d_step, size_int);
	cudaMemcpy(d_step, &step, size_int, cudaMemcpyHostToDevice);

	double* d_Istimamp;
	cudaMalloc((void **)&d_Istimamp, size_double);
	cudaMemcpy(d_Istimamp, &Istimamp, size_double, cudaMemcpyHostToDevice);

	int* d_Nx;             
	cudaMalloc((void **)&d_Nx, size_int);
	cudaMemcpy(d_Nx, &Nx, size_int, cudaMemcpyHostToDevice);

	int* d_Ny;             
	cudaMalloc((void **)&d_Ny, size_int);
	cudaMemcpy(d_Ny, &Ny, size_int, cudaMemcpyHostToDevice);

	int* d_blocktimenum;
	cudaMalloc((void **)&d_blocktimenum, size_int);
	cudaMemcpy(d_blocktimenum, &blocktimenum, size_int, cudaMemcpyHostToDevice);

	double* d_stimint;             
	cudaMalloc((void **)&d_stimint, size_double);
	cudaMemcpy(d_stimint, &stimint, size_double, cudaMemcpyHostToDevice);

	double* d_dt;             
	cudaMalloc((void **)&d_dt, size_double);
	cudaMemcpy(d_dt, &dt, size_double, cudaMemcpyHostToDevice);

	int* d_BC; 
	cudaMalloc((void **)&d_BC, size_int);
	cudaMemcpy(d_BC, &BC, size_int, cudaMemcpyHostToDevice);
	
	double* d_dx;
	cudaMalloc((void **)&d_dx, size_double);
	cudaMemcpy(d_dx, &dx, size_double, cudaMemcpyHostToDevice);

	double* d_dy;
	cudaMalloc((void **)&d_dy, size_double);
	cudaMemcpy(d_dy, &dy, size_double, cudaMemcpyHostToDevice);*/

	xyzw_size = (Nx+2) * (Ny+2) * (varnum) * (datarr4dim) * sizeof(double);
	xyzw_stim_size = Nx * Ny * stimnum * 2 * sizeof(double);
	xyz_deriv_size = Nx * Ny * derivnum * sizeof(double);
	
	// allocate host memory
	// should have already allocated host memory 

 	// allocate device memory

	d_datarr = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, xyzw_size, NULL, &errcode);
	if(stimnum>0) d_stimarr = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, xyzw_stim_size, NULL, &errcode);
	d_derivarr = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, xyz_deriv_size, NULL, &errcode);
	if(blocktimenum>0) d_blocktimes = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, blocktimenum*2*sizeof(double), NULL, &errcode);
	d_block = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, (Nx+2)*(Ny+2)*sizeof(int), NULL, &errcode);
	d_constarr = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, constnum*sizeof(double), NULL, &errcode);
	d_Afield = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, Nx*Ny*sizeof(double), NULL, &errcode);
	d_Dp = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, 2*2*sizeof(double), NULL, &errcode);
	d_D = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, 2*2*sizeof(double), NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");
	
	/*cutilSafeCall(cudaMalloc((void**)&d_datarr, 
			xyzw_size));	
	cutilSafeCall(cudaMalloc((void**)&d_stimarr, 
			xyzw_stim_size));	
	cutilSafeCall(cudaMalloc((void**)&d_derivarr, 
			xyz_deriv_size));
	
	cutilSafeCall(cudaMalloc((void**)&d_blocktimes,
			blocktimenum*2*sizeof(double)));
	
	cutilSafeCall(cudaMalloc((void**)&d_block,
			(Nx+2)*(Ny+2)*sizeof(int)));
	
	cutilSafeCall(cudaMalloc((void**)&d_constarr,
			constnum*sizeof(double)));
	
	cutilSafeCall(cudaMalloc((void**)&d_Afield,
			Nx*Ny*sizeof(double)));
	
	cutilSafeCall(cudaMalloc((void**)&d_Dp,
			2*2*sizeof(double)));
	
	cutilSafeCall(cudaMalloc((void**)&d_D,
			2*2*sizeof(double)));*/
    			
	linear_datarr = (double *) malloc ( (unsigned int)
             (sizeof(double)*2*varnum*(Ny+2)*(Nx+2)));

	if (NULL == linear_datarr)
	{
		printf("Malloc Failed\n");
		exit(-1);
	}
                                    
    // copy host memory to device
       	for (l = 0; l < 2; l++)
	{
		for (k = 0; k < varnum; k++)
		{
			for (i = 0; i < (Nx+2); i++)
			{
				for (j = 0; j < (Ny+2); j++)
                		{
		             		*(linear_datarr+
		                	l*(Ny+2)*(Nx+2)*varnum+
		                	k*(Ny+2)*(Nx+2)+
		                	i*(Ny+2)+
		                	j) = datarr[l][k][i][j]; 
                		}
			}
		}
	}
 
	// Coalescing cudaMemcpy
	errcode = clEnqueueWriteBuffer(clCommandQue, d_datarr, CL_TRUE, 0, (Nx+2)*(Ny+2)*varnum*2*sizeof(double), linear_datarr, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");

	//cutilSafeCall(cudaMemcpy(d_datarr,linear_datarr,(Nx+2)*(Ny+2)*varnum*2*sizeof(double),cudaMemcpyHostToDevice));


    
	linear_stimarr = (double*)malloc((unsigned int)
                        (Nx*Ny*stimnum*2*sizeof(double))); 
	if (NULL == linear_stimarr)
	{
		printf("Malloc Linear Stimarr Failed\n");
		exit(-1);
	}
	//stim array
	for (i = 0; i < Nx; ++i)
	{
		for (j = 0; j < Ny; ++j)
		{
			for (k = 0; k < stimnum; ++k)
			{
                		for (l = 0; l < 2; ++l)
				{
				     *(linear_stimarr+
				        i*Ny*stimnum*2+
				        j*stimnum*2+
				        k*2+
				        l) = stimarr[i][j][k][l]; 
				}
			}
		}
	} 

	if(stimnum>0){
		errcode = clEnqueueWriteBuffer(clCommandQue, d_stimarr, CL_TRUE, 0, Nx*Ny*stimnum*2*sizeof(double), linear_stimarr, 0, NULL, NULL);
		if(errcode != CL_SUCCESS)printf("Error in writing buffer stimarr %d\n",errcode);
	}
	//cutilSafeCall( cudaMemcpy(d_stimarr,linear_stimarr,Nx*Ny*stimnum*2*sizeof(double),cudaMemcpyHostToDevice) );


    
	linear_deriv3darr = (double*)malloc((unsigned int)
                        (Nx*Ny*derivnum*sizeof(double))); 
	if (NULL == linear_deriv3darr)
	{
		printf("Malloc Linear Deriv3darr Failed\n");
		exit(-1);
	}
	//derive3d array
	for (i = 0; i < Nx; ++i)
	{
		for (j = 0; j < Ny; ++j)
		{
			for (k = 0; k < derivnum; ++k)
			{
		             *(linear_deriv3darr+
		                i*Ny*derivnum+
		                j*derivnum+
		                k) = deriv3darr[i][j][k]; 
			}
		}
	} 

	errcode = clEnqueueWriteBuffer(clCommandQue, d_derivarr, CL_TRUE, 0, Nx*Ny*derivnum*sizeof(double), linear_deriv3darr, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffer derivarr\n");

	//cutilSafeCall( cudaMemcpy(d_derivarr,linear_deriv3darr,Nx*Ny*derivnum*sizeof(double),cudaMemcpyHostToDevice) );
	
	/* 1D array just Memcpy */
	/* d_blocktimes */
	for (i = 0; i < blocktimenum; i++)
	{
		errcode = clEnqueueWriteBuffer(clCommandQue, d_blocktimes, CL_TRUE, i*2*sizeof(double), 2*sizeof(double), blocktimes[i], 0, NULL, NULL);
		if(errcode != CL_SUCCESS)printf("Error in writing buffer d_blocktimes\n");
		//cutilSafeCall(cudaMemcpy(d_blocktimes+i*2,blocktimes[i],2*sizeof(double), cudaMemcpyHostToDevice) );
	}	
   	/* d_block */
	for (i = 0; i < Nx + 2; i++)
	{	
		errcode = clEnqueueWriteBuffer(clCommandQue, d_block, CL_TRUE, i*(Ny+2)*sizeof(int), (Ny+2)*sizeof(int), block[i], 0, NULL, NULL);
		if(errcode != CL_SUCCESS)printf("Error in writing buffer d_block\n");
		//cutilSafeCall(cudaMemcpy(d_block+i*(Ny+2),block[i],(Ny+2)*sizeof(int), cudaMemcpyHostToDevice) );
	}	
	/* d_constarr */
	errcode = clEnqueueWriteBuffer(clCommandQue, d_constarr, CL_TRUE, 0, constnum*sizeof(double), constarr, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffer constarr\n");
	//cutilSafeCall(cudaMemcpy(d_constarr, constarr, constnum*sizeof(double),cudaMemcpyHostToDevice));
   	
	/* d_Afield */
	for (i = 0; i < Nx; i++)
	{	
		errcode = clEnqueueWriteBuffer(clCommandQue, d_Afield, CL_TRUE, i*Ny*sizeof(double), Ny*sizeof(double), Afield[i], 0, NULL, NULL);
		if(errcode != CL_SUCCESS)printf("Error in writing buffer d_Afield \n");
		//cutilSafeCall(cudaMemcpy(d_Afield+i*Ny,Afield[i],Ny*sizeof(double), cudaMemcpyHostToDevice) );
	}
 	/* d_Dp */		
	for (i = 0; i < 2; i++)
	{	
		errcode = clEnqueueWriteBuffer(clCommandQue, d_Dp, CL_TRUE, i*2*sizeof(double), 2*sizeof(double), Dp[i], 0, NULL, NULL);
		if(errcode != CL_SUCCESS)printf("Error in writing buffer d_Dp\n");
		//cutilSafeCall(cudaMemcpy(d_Dp+i*2,Dp[i],2*sizeof(double), cudaMemcpyHostToDevice) );
	}
	/* d_D */
	for (i = 0; i < 2; i++)
	{	
		errcode = clEnqueueWriteBuffer(clCommandQue, d_D, CL_TRUE, i*2*sizeof(double), 2*sizeof(double), D[i], 0, NULL, NULL);
		if(errcode != CL_SUCCESS)printf("Error in writing buffer d_D\n");
		//cutilSafeCall(cudaMemcpy(d_D+i*2,D[i],2*sizeof(double), cudaMemcpyHostToDevice) );
	}
	return;
}

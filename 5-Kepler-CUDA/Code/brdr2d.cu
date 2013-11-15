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
//#include "brdr2dequations.c"

// includes, project
#include <cutil_inline.h>
#include "cuPrintf.cu"
// includes, kernels
#include <brdr2d_kernel.cu>


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void invokeGPU(int argc, char** argv);
void GPU_Mem_init(void);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len);
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
int	size_int = sizeof(int) * VAR_N;
int size_double = sizeof(double) * VAR_N;


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
  	printf("%s",stime);
	
	invokeGPU(argc, argv);

	printf("Saving final conditions...\n\n");

	brfc();
 	printf("         tfinal: %5.3f msec\n",tfinal);
  	printf("     Final time: %5.3f msec\n",derivarr[0]);
  	printf("         Nsteps: %10.2f\n",Nsteps);
  	printf("Number of steps: %d\n",step);
  	printf("             Nx: %d\n",Nx);
  	printf("             Ny: %d\n",Ny);

  	ttime=time(0);                 // Get current time
  	stime=ctime(&ttime);
  	printf("%s\n",stime);
    
  	closefiles();
}

void
invokeGPU(int argc, char *argv[])
{
	int i,j,k;

	//allocate GPU memory, copy data from host to device	
	GPU_Mem_init();
	
	//unsigned int timer = 0;
  	printf("Entering time loop ... \n");
	//cutilCheckError( cutCreateTimer( &timer));
	//cutilCheckError( cutStartTimer( timer));
	step = 1;
	derivarr[0] += dt;
	deriv3darr[0][0][0] += dt;  // update time (msec) 
	cutilSafeCall(cudaMemcpy(d_derivarr,
				deriv3darr[0][0],
				sizeof(double), cudaMemcpyHostToDevice));			

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
	dim3 dimGrid(BLOCK_DIMX,BLOCK_DIMY,1);
	dim3 dimBlock(THREAD_DIMX,THREAD_DIMY,1);

	cudaPrintfInit();
	
	double gpu_start = rtclock();
	double stim_time=0;
	double block_time=0;
	double cur_time=0;
	double gate_time=0;
	double bcs_time=0;
	double mem_time=0;
	double time_temp;
	double update_time=0;
	while (derivarr[0] <= tfinal+dt && step <= Nsteps + 1)
	{
		// from (1 to Nx) instead of (0 to Nx+1)
		// do not loop through ghost points */
		//GPU Kernel Execution
		time_temp = rtclock();
		if(stimnum>0) d_stimulate_kernel<<<dimGrid,dimBlock>>>(stimnum,d_datarr,d_stimarr,d_derivarr,varnum,step,Istimamp,Nx,Ny,stimint);
		cudaThreadSynchronize();
		stim_time += (double)(rtclock()-time_temp);

		time_temp = rtclock();
		if(blocktimenum>0) d_blockonoff_kernel<<<dimGrid,dimBlock>>>(blocktimenum, d_derivarr, d_blocktimes, d_block, Nx, Ny);
		cudaThreadSynchronize();
		block_time += (double)(rtclock()-time_temp);

		time_temp = rtclock();
		d_brgates_kernel<<<dimGrid,dimBlock>>>(varnum, d_datarr, d_derivarr, d_constarr, step, Nx, Ny);
		cudaThreadSynchronize();
		gate_time += (double)(rtclock()-time_temp);

		time_temp = rtclock();
		d_brcurrents_kernel<<<dimGrid,dimBlock>>>(stimnum, d_datarr, d_derivarr, step, Istimamp, Nx,Ny, varnum, d_constarr, d_Afield, d_block, d_Dp, dt);
		cudaThreadSynchronize();
		cur_time += (double)(rtclock()-time_temp);

		time_temp = rtclock();
		dim3 dimGrid1(1,1,1);
    	dim3 dimBlock1(1,1,1);
		kernel_call_device_bcs<<< dimGrid1, dimBlock1 >>>(dx, dy, d_D, BC, step, Nx, Ny, varnum, d_Dp, d_datarr, d_derivarr, dt);  
		cudaThreadSynchronize();		
		cutilCheckMsg("CUDA Kernel");
		bcs_time += (double)(rtclock()-time_temp);
		
		time_temp = rtclock();
		NinePointLaplacian<<< dimGrid, dimBlock >>>(step, varnum, Nx, Ny, Dp[0][0],Dp[0][1],Dp[1][0],Dp[1][1], d_datarr);  
		cudaThreadSynchronize();		
		cutilCheckMsg("Laplacian CUDA Kernel");
		update_time += (double)(rtclock()-time_temp);
		
		time_temp = rtclock();
		if (step % rpN == 0) {
			// Coalescing cudaMemcpy
			cutilSafeCall(cudaMemcpy(linear_datarr, 
				             d_datarr, 
				             (Nx+2)*(Ny+2)*varnum*2*sizeof(double),
				             cudaMemcpyDeviceToHost));
		    
			// copy host memory from device
			for (int l = 0; l < 2; l++)
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
	printf("Kernel bcs    time is %.2lf\n",bcs_time);
	printf("Kernel update time is %.2lf\n",update_time);
	printf("memory copy   time is %.2lf\n",mem_time);
	printf("GPU           time is %.2lf\n",stim_time+block_time+gate_time+cur_time+bcs_time+update_time);				   
	cudaPrintfEnd();
	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed");
	
	//cutilCheckError( cutStopTimer( timer));
	//printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
	//cutilCheckError( cutDeleteTimer( timer));

	// cleanup memory
	cudaThreadExit();
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

void blockonoff(void)
{
  int i,m,n;

  for (i=0;i<blocktimenum;++i){
    if ((derivarr[0]>=blocktimes[i][0])&&(blocktimes[i][1]==0.0)){
      printf("Changing block conditions: %4.3f msec \n",derivarr[0]);
      blocktimes[i][1]=1.0;
      for (m=1;m<Nx+1;++m){
        for (n=1;n<Ny+1;++n){
      if (block[m][n]==0) block[m][n]=1;
    }
      }
    }
  }

}

void GPU_Mem_init(void)
{
	int i, j, k;		// loop index
	long int xyzw_size;
	long int xyzw_stim_size;
	long int xyz_deriv_size;
	

	
	// Use device with highest Gflops/s
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
	cudaMemcpy(d_dy, &dy, size_double, cudaMemcpyHostToDevice);

	xyzw_size = (Nx+2) * (Ny+2) * (varnum) * (datarr4dim) * sizeof(double);
	xyzw_stim_size = Nx * Ny * stimnum * 2 * sizeof(double);
	xyz_deriv_size = Nx * Ny * derivnum * sizeof(double);
	
	// allocate host memory
	// should have already allocated host memory 

 	// allocate device memory
	cutilSafeCall(cudaMalloc((void**)&d_datarr, 
			xyzw_size));	
	cutilSafeCall(cudaMalloc((void**)&d_stimarr, 
			xyzw_stim_size));	
	cutilSafeCall(cudaMalloc((void**)&d_derivarr, 
			xyz_deriv_size));
	/* d_blocktimes */
	cutilSafeCall(cudaMalloc((void**)&d_blocktimes,
			blocktimenum*2*sizeof(double)));
	/* d_block */
	cutilSafeCall(cudaMalloc((void**)&d_block,
			(Nx+2)*(Ny+2)*sizeof(int)));
	/* d_constarr */
	cutilSafeCall(cudaMalloc((void**)&d_constarr,
			constnum*sizeof(double)));
	/* d_Afield */
	cutilSafeCall(cudaMalloc((void**)&d_Afield,
			Nx*Ny*sizeof(double)));
	/* d_Dp */
	cutilSafeCall(cudaMalloc((void**)&d_Dp,
			2*2*sizeof(double)));
	/* d_D */
	cutilSafeCall(cudaMalloc((void**)&d_D,
			2*2*sizeof(double)));
    		
	
	linear_datarr = (double *) malloc ( (unsigned int)
             (sizeof(double)*2*varnum*(Ny+2)*(Nx+2)));

	if (NULL == linear_datarr)
	{
		printf("Malloc Failed\n");
		exit(-1);
	}
                                    

    // copy host memory to device
       	for (int l = 0; l < 2; l++)
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
	cutilSafeCall(cudaMemcpy(d_datarr, 
                             linear_datarr, 
                             (Nx+2)*(Ny+2)*varnum*2*sizeof(double),
                             cudaMemcpyHostToDevice));


    
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
                		for (int l = 0; l < 2; ++l)
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

	cutilSafeCall( cudaMemcpy(
		            d_stimarr,
            		linear_stimarr, 
		            Nx*Ny*stimnum*2*sizeof(double), 
                    cudaMemcpyHostToDevice) );


    
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

	cutilSafeCall( cudaMemcpy(
		            d_derivarr,
            		linear_deriv3darr, 
		            Nx*Ny*derivnum*sizeof(double), 
                    cudaMemcpyHostToDevice) );
	
	/* 1D array just Memcpy */
	/* d_blocktimes */
	for (i = 0; i < blocktimenum; i++)
	{
		cutilSafeCall(cudaMemcpy(
					d_blocktimes+i*2,
					blocktimes[i], 
					2*sizeof(double), cudaMemcpyHostToDevice) );
	}	
   	/* d_block */
	for (i = 0; i < Nx + 2; i++)
	{	
		cutilSafeCall(cudaMemcpy(
					d_block+i*(Ny+2),
					block[i], 
					(Ny+2)*sizeof(int), cudaMemcpyHostToDevice) );
	}	
	/* d_constarr */
	cutilSafeCall(cudaMemcpy(d_constarr, constarr, constnum*sizeof(double),
			cudaMemcpyHostToDevice));
   	/* d_Afield */
	for (i = 0; i < Nx; i++)
	{	
		cutilSafeCall(cudaMemcpy(
					d_Afield+i*Ny,
					Afield[i], 
					Ny*sizeof(double), cudaMemcpyHostToDevice) );
	}
 	/* d_Dp */		
	for (i = 0; i < 2; i++)
	{	
		cutilSafeCall(cudaMemcpy(
					d_Dp+i*2,
					Dp[i], 
					2*sizeof(double), cudaMemcpyHostToDevice) );
	}
	/* d_D */
	for (i = 0; i < 2; i++)
	{	
		cutilSafeCall(cudaMemcpy(
					d_D+i*2,
					D[i], 
					2*sizeof(double), cudaMemcpyHostToDevice) );
	}
	return;
}

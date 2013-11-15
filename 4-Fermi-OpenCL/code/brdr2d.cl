#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define d4(i) (i)*varnum*(Nx+2)*(Ny+2)
#define d3(i) (i)*(Nx+2)*(Ny+2)
#define d2(i) (i)*(Ny+2) 

double d_abfun(
        double vv,
        int i
     )
{
    double t1, t2, t3;
    double ab;
    double C[12][7]={ 
    0.0005,  0.083, 50,      0,    0,     0.057, 1,
    0.0013, -0.06,  20,      0,    0,    -0.04,  1,
    0,       0,     42.65,  -0.9, 42.65, -0.22, -1,
    1.437,  -0.085, 39.75,   0,    0,     0,     0,
    0.1,    -0.193, 79.65,   0,    0,     0,     0,
    1.7,     0,     20.5,    0,    0,    -0.095, 1,
    0.055,  -0.25,  78,      0,    0,    -0.2,   1,   
    0.3,     0,     32,      0,    0,    -0.1,   1, 
    0.095,  -0.01,  -5,      0,    0,    -0.072, 1,
    0.07,   -0.017, 44,      0,    0,     0.05,  1,
    0.012,  -0.008, 28,      0,    0,     0.15,  1,
    0.0065, -0.02,  30,      0,    0,    -0.2,   1};
  
    t1 = C[i][0] * exp(C[i][1]*(vv+C[i][2]));
    t2 = C[i][3] * (vv+C[i][4]);
    t3 = C[i][6] + exp(C[i][5]*(vv+C[i][2]));
    ab = (t1 + t2) / t3;

    return ab;
}

__kernel void
d_stimulate_kernel(
        int stimnum, 
        __global double *d_datarr, 
        __global double *d_stimarr,
        __global double *derivarr,
        int varnum, 
        int step,
        double Istimamp,     
	int Nx,
        int Ny,
	double stimint		
        )
{
	
	int index;
	int i,X,Y;
	
	//X = blockIdx.y*blockDim.y+threadIdx.y+1;
	//Y = blockIdx.x*blockDim.x + threadIdx.x+1;

	X = get_global_id(1)+1;
	Y = get_global_id(0)+1;	
  	/* Remember: stimarr is structured as [0:Nx-1][0:Ny-1][0:stimnum-1][0:1]
               AND stimarr is not defined at ghost nodes!! 
           So, to colocate datarr and stimarr use
           stimarr[Xstep-1][Ystep-1][i][0:1] and 
           datarr[Xstep][Ystep][15][(step-1)%2]   */

	index = d4((step-1)%2) + d3(14) + d2(X) + Y;
  	*(d_datarr + index) = 0;
	
	for (i = 0;i < stimnum; ++i) 
	{
		index = (X-1)*(Ny)*stimnum*2 + (Y-1)*stimnum*2 + i*2 + 0;
		if ((*(d_stimarr + index) != 0) &&
			(*derivarr >= *(d_stimarr + index)) &&
			(*derivarr <= *(d_stimarr + index) + stimint))
		{
      		if (*(d_stimarr + index + 1) == 0.0) 
			{
        		//fprintf(stiminfofile,"Applying Stimulus at %4.3f msec to node (%d,%d)\n",derivarr[0],Xstep,Ystep);
    			//fflush(stiminfofile);
    			*(d_stimarr + index + 1) = 1.0;
			} //end if
	
			index = d4((step-1)%2) + d3(14) + d2(X) + Y;
      		*(d_datarr + index) = Istimamp;
    	}  //end if
	}   //end for
}

__kernel void
d_blockonoff_kernel(
            int blocktimenum,
            __global double* derivarr,
            __global double* blocktimes,
            __global int* block,
	    int Nx,
	    int Ny            
            )
{
	int i,m,n;

	//X = blockIdx.x*blockDim.x + threadIdx.x+1;
	//Y = blockIdx.y*blockDim.y + threadIdx.y+1;
	

	for (i = 0; i < blocktimenum; ++i)
	{
		if ((derivarr[0] >= *(blocktimes+i*2)) && (*(blocktimes+i*2+1) == 0.0))
		{
			//printf("Changing block conditions: %4.3f msec \n",derivarr[0]);
			*(blocktimes+2*i+1) = 1.0;
      			for (m = 1; m < Nx + 1; ++m)
			{
        			for (n = 1; n < Ny + 1; ++n)
				{
					if (*(block+m*(Ny+2)+n) == 0) *(block+m*(Ny+2)+n) = 1;
				}
			}
			  
		}
	}
}


__kernel void 
d_brgates_kernel(
        int varnum, 
        __global double *d_datarr, 
        __global double *derivarr,
        __global double *constarr,
        int step,
	int Nx,
        int Ny
       )
{
	/* Ix1 */
	double ax1,bx1,tx1,x1inf,dx1dt,x1;
	/* INa */
	double am,bm,tm,minf,ah,bh,th,hinf,dmdt,m,dhdt,h;
	/* Is */
	double ad,bd,td,dinf,af,bf,tf,finf,dddt,d,dfdt,f;
	/* Cai */
	double dCaidt,kCa,Cai, sigma;
	/* Is */
	double Is;
	/* Vm */
	double Vm;

	int index;
	int derivnum = 7;
	int X;
	int Y;

	X = get_global_id(1)+1;
	Y = get_global_id(0)+1;
	 

	/* Need these from datarr to update derivatives in derivarr */
	/* datarr values are not altered here */
	/* these are initial conditions for step=1 */
	index = d4((step-1)%2) + d2(X) + Y;
	 
	Vm = *(d_datarr + index);   /* mV */
	 
	x1 = *(d_datarr + index + d3(4));   /* unitless */
	 
	m = *(d_datarr + index + d3(6));    /* unitless */
	 
	h = *(d_datarr + index + d3(7));    /* unitless */
	 
	Is = *(d_datarr + index + d3(8));   /* uA/cm^2 */
	 
	d = *(d_datarr + index + d3(9));   /* unitless */
	 
	f =  *(d_datarr + index + d3(10));   /* unitless */
	 
	Cai =  *(d_datarr + index + d3(11)); /* moles/L */

	 /* Constants */
	kCa = constarr[6];      /* msec^-1 */
	sigma = constarr[11];   /* unitless */

    /* Ix1  */ 
    ax1 = d_abfun(Vm, 0);
    bx1 = d_abfun(Vm, 1);
    tx1 = 1 / (ax1+bx1);
    x1inf = ax1 * tx1;
    dx1dt = (x1inf - (x1)) / tx1;
	 

    /* INa */
    am = d_abfun(Vm, 2);
    bm = d_abfun(Vm, 3);
    tm = 1 / (am+bm);
    minf = am * tm;
    dmdt = (minf - (m)) / tm;
	 
    ah = d_abfun(Vm, 4);
    bh = d_abfun(Vm, 5);
    th = 1 / (ah + bh);
    hinf = ah * th;
    dhdt = (hinf - (h)) / th;
 

    /* Is */
    ad = d_abfun(Vm, 8);
    bd = d_abfun(Vm, 9);
    td = (1 / (ad+bd)) * (sigma);
    dinf = ad * td;
    dddt = (dinf - (d)) / td;
	 

    af = d_abfun(Vm, 10);
    bf = d_abfun(Vm, 11);
    tf = (1 / (af+bf)) * (sigma);
    finf = af * tf;
    dfdt = (finf - (f)) / tf;
	 

    /* Cai */
    dCaidt = (-1e-7)*(Is) + (kCa)*((1e-7)-(Cai));  /* mole/L */

	/* Derivatives */
	index = (X-1)*Ny*derivnum + (Y-1)*derivnum;
	*(derivarr+index+1)=dx1dt;/* unitless */
	*(derivarr+index+2)=dmdt; /* unitless */
	*(derivarr+index+3)=dhdt; /* unitless */
	*(derivarr+index+4)=dddt; /* unitless */
	*(derivarr+index+5)=dfdt; /* unitless */
	*(derivarr+index+6)=dCaidt;   /* mole/L */
}





__kernel void 
d_brcurrents_kernel(
            int stimnum, 
            __global double *d_datarr, 
            __global double *derivarr,
            int step,
            double Istimamp,     
	    int Nx,
            int Ny,
            int varnum, 
            __global double *constarr,
            __global double * Afield,
            __global int * block,
            __global double * Dp,
	double dt
)            
{ 
    /* IK1 */
    double IK1t1,IK1t2,IK1t3,IK1t4,gK1,IK1;
    /* Ix1 */
    double Ix1t1,Ix1t2,gx1,Ix1,dx1dt,x1;
    /* INa */
    double gNa,ENa,INa,dmdt,m,dhdt,h, A;
    /* Cai */
    double dCaidt,Cai;
    /* Is */
    double Es,gs,Is,dddt,d,dfdt,f;
    /* Other currents */
    double Isum,Istim;
    /* Vm */
    double Cm, Vm, dVmdt;
    /* Diffusion */
    double Diff;
	int index;
	int jj;
	int derivnum = 7;
	int X;
	int Y;

	//X = blockIdx.y*blockDim.y+threadIdx.y+1;
	//Y = blockIdx.x*blockDim.x + threadIdx.x+1;
	
	X = get_global_id(1)+1;
	Y = get_global_id(0)+1;
	 
    /* Constants */
    gK1 = constarr[0];            /* mmho/cm^2 */
    gNa = constarr[1];            /* mmho/cm^2 */
    ENa = constarr[2];            /* mV */
    gx1 = constarr[3];            /* mmho/cm^2 */
    gs = constarr[4];             /* mmho/cm^2 */
    Cm = constarr[5];             /* uF/cm^2 */
    //gNaC = &constarr[7];           /* mmho/cm^2 */
    A = *(Afield + (X-1)*Ny  + (Y-1)); /* unitless */  
	 
    /*X-1 be attention, distinguish from colocation */

    /* Need these from derivarr to update datarr */
    /* derivarr values are not altered here */
    //    t=&derivarr[0]     /* msec */
	index = (X-1)*Ny*derivnum+(Y-1)*derivnum;
	 
    dx1dt = *(derivarr+index+1);      /* unitless */
	 
    dmdt = *(derivarr+index+2);       /* unitless */
	 
    dhdt = *(derivarr+index+3);       /* unitless */
	 
    dddt = *(derivarr+index+4);       /* unitless */
	 
    dfdt = *(derivarr+index+5);       /* unitless */
	 
    dCaidt = *(derivarr+index+6);     /* mole/L */
	 

    index = d2(X) + Y;
    /* copy previous timestep data into current datarr */
    /* such that previous Vm is used to update next Vm */
    for (jj=0; jj<varnum-2; jj++)     /* varnum-1 b/c don't want to overwrite */
    {
        /* the stimulus and diffusion in last 2 elements of datarr */
        *(d_datarr + index + d3(jj) + d4(step%2)) = *(d_datarr + index + d3(jj) + d4((step-1)%2));    
		 
    }

    /* datarr values to update */
    index = d4(step%2) + d2(X) + Y;	 
	 
    Vm = *(d_datarr + index);        /* mV */
	 
    //dVmdt = *(d_datarr + index + 1*2);     /* mV/msec */
	
    //IK1 = *(d_datarr + index + 2*2);       /* uA/cm^2 */
	 
    //Ix1 = *(d_datarr + index + 3*2);       /* uA/cm^2 */
	 
    x1 = *(d_datarr + index + d3(4));        /* unitless */
	 
    //INa = *(d_datarr + index + 5*2);       /* uA/cm^2 */
	 
    m = *(d_datarr + index + d3(6));         /* unitless */
	 
    h = *(d_datarr + index + d3(7));         /* unitless */
	 
    //Is = *(d_datarr + index + 8*2);        /* uA/cm^2 */
	 
    d = *(d_datarr + index + d3(9));        /* unitless */
	 
    f = *(d_datarr + index + d3(10));        /* unitless */
	 
    Cai = *(d_datarr + index + d3(11));      /* mole/L */
	 
    //Isum = *(d_datarr + index + 12*2);    /* uA/cm^2 */
	 
    Diff = *(d_datarr + index + d3(13));     /* mV/msec */
	 
    Istim = *(d_datarr + index + d3(14));    /* uA/cm^2 */
	 

    /* IK1 */
    IK1t1 = 4 * (exp(0.04*(Vm+85))-1);
    IK1t2 = exp(0.08*(Vm+53)) + exp(0.04*(Vm+53));
    IK1t3 = 0.2 * (Vm+23);
    IK1t4 = 1 - exp(-0.04*(Vm+23));
    /* uA/cm^2  09/07/2005 */
    IK1 = (A) * (gK1) * (IK1t1/IK1t2 + IK1t3/IK1t4);
    *(d_datarr + index + d3(2))=IK1 ;       /* uA/cm^2 */	 

    /* Ix1 */
    x1 = x1 + dx1dt*dt;
    *(d_datarr + index + d3(4))=x1;        /* unitless */
    Ix1t1 = exp(0.04*(Vm+77)) - 1;
    Ix1t2 = exp(0.04*(Vm+35));
    Ix1 = ((gx1*Ix1t1) / (Ix1t2)) * (x1);        /* uA/cm^2 */
    *(d_datarr + index + d3(3))=Ix1;       /* uA/cm^2 */	 

    /* INa */
    m =  m+ dmdt * dt;
    *(d_datarr + index + d3(6))=m;         /* unitless */
    h =  h+dhdt * dt;
    *(d_datarr + index + d3(7))=h;         /* unitless */
    /* *INa=((gNa)*(pow((m),3.0))*(h)*(*j)+*gNaC)*(Vm-ENa); */
    /* uA/cm^2, BR Na current */
    /* uA/cm^2, no gNaC or j gate in BRDR Na current */
    INa = ((gNa) * (*(block+X*(Ny+2) +Y))
			 * (pow((m),3.0))*(h))*(Vm-ENa); 
    *(d_datarr + index + d3(5))=INa;       /* uA/cm^2 */        
	 

    /* Cai */
    Cai = Cai + dCaidt * dt;              /* moles/L */
    *(d_datarr + index + d3(11))=Cai;      /* mole/L */	 

    /* Is */
    d = d+dddt * dt;
    *(d_datarr + index + d3(9))=d;        /* unitless */
	 
    f = f+dfdt * dt;
    *(d_datarr + index + d3(10))=f;        /* unitless */

    Es = -82.3 - 13.0287 * log(Cai);
    Is = (gs) * (*(block+X*(Ny+2) +Y))
		 * (d) * (f) * (Vm-Es);  /* uA/cm^2 */
    *(d_datarr + index + d3(8)) = Is;

    /* Vm */
    Isum = (IK1 + Ix1 + INa + Is);                     /* uA/cm^2 */
    *(d_datarr + index + d3(12)) = Isum;
 
    dVmdt = (Diff) - (1/(Cm))*(Isum-Istim);        /* mV/msec */
    *(d_datarr + index + d3(1)) = dVmdt;
    *(d_datarr+index) = Vm + ((dVmdt)*dt);                          /* mV */
   
	
    //Vm = Vm + ((dVmdt)*dt);                          /* mV */

    
    
}

__kernel void
kernel_call_device_bcs(
    double dx,
    double dy,
    __global double* D,
    int    BC,
    int    step,
    int    Nx,
    int    Ny,
    int    varnum,
    __global double* Dp,
    __global double* datarr,
    __global double* derivarr,
    double dt
    )
{
    // apply Neumann boundary conditions 
    // by 'correcting' the diffusion matrix 
    
  
    int ii;
    double R0, R1;

    int r4 = d4(step%2);
    

    R0=(*(D+2) / *(D+0)) * ((dx)/(dy));
    R1=(*(D+2) / *(D+3)) * ((dy)/(dx));


    if ((BC)==1) {   // Slab
    /* First set Vm at ghost nodes */
    *(datarr+r4+1) =  *(datarr+r4+d2(2)+1);
    *(datarr+r4) = *(datarr+r4+d2(2)+2);
    *(datarr+r4+d2(1)) = *(datarr+r4+d2(1)+2);
    *(datarr+d2(Nx)+r4) = *(datarr+r4+d2(Nx)+2);

    *(datarr+r4+d2(Nx+1)) =  *(datarr+r4+d2(Nx-1)+2);
    *(datarr+r4+d2(Nx+1)+1) = *(datarr+r4+d2(Nx-1)+1);
    *(datarr+r4+d2(Nx+1)+Ny) =  *(datarr+r4+d2(Nx-1)+Ny);
    *(datarr+r4+d2(Nx+1)+Ny+1) =  *(datarr+r4+d2(Nx-1)+Ny-1);

    *(datarr+r4+d2(Nx)+Ny+1) =  *(datarr+r4+d2(Nx)+Ny-1);
    *(datarr+r4+d2(1)+Ny+1) = *(datarr+r4+d2(1)+Ny-1);
    *(datarr+r4+Ny+1) =  *(datarr+r4+d2(2)+Ny-1);
    *(datarr+r4+Ny) =  *(datarr+r4+d2(2)+Ny);

    // decouple these loops b/c (Nx) might not equal (Ny) 
    for (ii=2;ii<(Nx);++ii)
    {
    	// Eq 3 in notes 	
    	*(datarr+d2(ii)+Ny+1+r4) = 
    	    *(datarr+d2(ii)+(Ny-1)+r4) + 
    			R1 * (*(datarr+d2(ii-1)+Ny+r4) - 
    			*(datarr+d2(ii+1)+Ny+r4));  
    	// Eq 2 in notes 
    	*(datarr+d2(ii)+r4) = 
    		*(datarr+d2(ii)+r4+2) -
    			R1 * (*(datarr+d2(ii-1)+r4+1) - 
    			*(datarr+d2(ii+1)+r4+1));  
    }

    // decouple these loops b/c (Nx) might not equal (Ny) 
    for (ii=2;ii<(Ny);++ii)
    {           
    	// Eq 1 in notes 
    	*(datarr+r4+ii) = 
    		*(datarr+r4+d2(2)+ii) -
    			R0 * (*(datarr+r4+d2(1)+ii-1) - 
    			*(datarr+r4+d2(1)+(ii+1)));            
    	// Eq 4 in notes 
    	*(datarr+r4+d2(Nx+1)+ii) = 
    		*(datarr+r4+d2(Nx-1)+ii) -
    			R0 * (*(datarr+r4+d2(Nx)+ii-1) - 
    			*(datarr+r4+d2(Nx)+(ii+1)));            
    }

    
    }
    *derivarr += dt;

}
__kernel void
NinePointLaplacian(
        int step,
        int varnum,
	int Nx,
        int Ny,
        double Dp0,
        double Dp1,
        double Dp2,
        double Dp3,
        __global double* d_datarr)
{
    int X, Y, index, index2;

    X = get_global_id(1)+1;
    Y = get_global_id(0)+1;

    index = d2(X) + Y + d3(13) + d4((step-1)%2);
    index2 = d2(X) + Y + d3(0) + d4(step%2);
    *(d_datarr + index) =
//                *(d_datarr+index)
                - (2*Dp0 + 2*Dp3)*(*(d_datarr+index2))
                - Dp2 * (*(d_datarr+index2-(Ny+2)+1))
                + Dp3 * (*(d_datarr+index2+1))
                + Dp2 * (*(d_datarr+index2+(Ny+2)+1))
                + Dp0 *(*(d_datarr+index2+(Ny+2)))
                - Dp2 * (*(d_datarr+index2+(Ny+2)-1))
                + Dp3 * (*(d_datarr+index2-1))
                + Dp2 * (*(d_datarr+index2-(Ny+2)-1))
                + Dp0 * (*(d_datarr+index2-(Ny+2)));
}



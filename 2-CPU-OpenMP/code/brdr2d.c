/* g++ brdr2d.c -o brdr2d -lm */

/* for debugging (via gdb/ddd): g++ -g brdr2d.c -o brdr2d -lm */

/* brdr2d &> out.txt   */

/* nohup brdr2d &> out.txt &   */

/* Model based upon Beeler (J. Physiol., 268, 177-210, 1977) */

/* and Drouhard (Comp. Biomed. Res., 20, 333-350, 1987).     */

/* see my references: beeler and drouhard.                   */

/*                                                           */

/* MWKay, 8/29/2002                                          */

/* MWKay, 3/28/2003 added block capability in setting gna    */

/*                  and gs equal to zero.                    */

/* MWKay, 9/07/2005 added potassium scale factor 'A'         */ 



#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <time.h>

#include <omp.h>

//#include <sys/dir.h>  use to check for and/or create data directory?

#include "brdr2d.h"

#include "brdr2dinout.c"

#include "brdr2dequations.c"

#include <sys/time.h>



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



int main(int argc, char *argv[]){



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

  double abfun(double vv,int i);

  double rtclock();

 

  printf("Initializing ... \n");

  initialize();

  if (ictype==2) {

    fcfilename=argv[1];

    FILE *fcfid;

    if ((fcfid=fopen(fcfilename,"rb"))==NULL){

      printf("Unable to open %s ... \n",fcfilename);

      exit(1);}

    readbrfc();  }

  if (stimnum>0) {

    printf("Building stimulus matrix ... \n");

    buildptstim();  }

    //buildedgestim(); }

  printf("Opening files ... \n");

  openfiles();

  step=0;

  printf("Writing initial conditions ... \n");

  output();

  printf("Entering time loop ... \n");

  time_t gpu_start = time(NULL);

  double cpu_start=rtclock();

  while (derivarr[0]<=tfinal && step<=Nsteps + 1 && stable){

    step=step+1;
    derivarr[0]+=dt;        // update time (msec) 
#pragma omp parallel for private(Ystep) private(Xstep) num_threads(8)
    for(Xstep = 1;Xstep<Nx+1;Xstep++){  
{
    //#pragma omp for
    for (Ystep = 1; Ystep < Ny+1; Ystep++)
  {

	if (stimnum>0) stimulate();
	
	if (blocktimenum>0) blockonoff();      
        /* Ix1 */

	double ax1,bx1,tx1,x1inf,dx1dt,x1;

	/* INa */

	double am,bm,tm,minf,ah,bh,th,hinf;

	/* Is */

	double ad,bd,td,dinf,af,bf,tf,finf,dddt,d,dfdt,f;

	/* Cai */

	double kCa,sigma;

        /* Vm */

       double Vm;

       /* IK1 */

       double IK1t1,IK1t2,IK1t3,IK1t4,gK1,IK1;

       /* Ix1 */

       double Ix1t1,Ix1t2,gx1,Ix1;

       /* INa */

       double gNa,ENa,INa,dmdt,m,dhdt,h, A;

       /* Cai */

       double dCaidt,Cai;

       /* Is */

       double Es,gs,Is;

       /* Other currents */

       double Isum,Istim;

       /* Vm */

       double Cm, dVmdt;

       /* Diffusion */

       double Diff;

	/* Need these from datarr to update derivatives in derivarr */

	/* datarr values are not altered here */

	/* these are initial conditions for step=1 */

  Vm=datarr[Xstep][Ystep][0][(step-1)%2];   /* mV */         

  x1=datarr[Xstep][Ystep][4][(step-1)%2];   /* unitless */

  m=datarr[Xstep][Ystep][6][(step-1)%2];    /* unitless */

  h=datarr[Xstep][Ystep][7][(step-1)%2];    /* unitless */

  Is=datarr[Xstep][Ystep][8][(step-1)%2];   /* uA/cm^2 */

  d=datarr[Xstep][Ystep][9][(step-1)%2];   /* unitless */

  f=datarr[Xstep][Ystep][10][(step-1)%2];   /* unitless */

  Cai=datarr[Xstep][Ystep][11][(step-1)%2]; /* moles/L */

    /* Constants */

    kCa = constarr[6];      /* msec^-1 */

    sigma = constarr[11];   /* unitless */

    /* Ix1  */ 

    ax1 = abfun(Vm, 0);

    bx1 = abfun(Vm, 1);

    tx1 = 1 / (ax1+bx1);

    x1inf = ax1 * tx1;

    dx1dt = (x1inf - (x1)) / tx1;

    /* INa */

    am = abfun(Vm, 2);

    bm = abfun(Vm, 3);

    tm = 1 / (am+bm);

    minf = am * tm;

    dmdt = (minf - (m)) / tm;

    ah = abfun(Vm, 4);

    bh = abfun(Vm, 5);

    th = 1 / (ah + bh);

    hinf = ah * th;

    dhdt = (hinf - (h)) / th;

    /* Is */

    ad = abfun(Vm, 8);

    bd = abfun(Vm, 9);

    td = (1 / (ad+bd)) * (sigma);

    dinf = ad * td;

    dddt = (dinf - (d)) / td;

    af = abfun(Vm, 10);

    bf = abfun(Vm, 11);

    tf = (1 / (af+bf)) * (sigma);

    finf = af * tf;

    dfdt = (finf - (f)) / tf;


    /* Cai */

    dCaidt = (-1e-7)*(Is) + (kCa)*((1e-7)-(Cai));  /* mole/L */

    /* Constants */

    gK1 = constarr[0];            /* mmho/cm^2 */

    gNa = constarr[1];            /* mmho/cm^2 */

    ENa = constarr[2];            /* mV */

    gx1 = constarr[3];            /* mmho/cm^2 */

    gs = constarr[4];             /* mmho/cm^2 */

    Cm = constarr[5];             /* uF/cm^2 */

    //gNaC = &constarr[7];           /* mmho/cm^2 */

    A=Afield[Xstep-1][Ystep-1]; /* unitless */

    /* copy previous timestep data into current datarr */

    /* such that previous Vm is used to update next Vm */

  int jj;

  for (jj=0; jj<varnum-2; jj++)                                               /* varnum-1 b/c don't want to overwrite */
 {
    datarr[Xstep][Ystep][jj][step%2]=datarr[Xstep][Ystep][jj][(step-1)%2];    /* the stimulus and diffusion in last 2 elements of datarr */
 }

  Vm=datarr[Xstep][Ystep][0][step%2];        /* mV */

  dVmdt=datarr[Xstep][Ystep][1][step%2];     /* mV/msec */

  IK1=datarr[Xstep][Ystep][2][step%2];       /* uA/cm^2 */

  Ix1=datarr[Xstep][Ystep][3][step%2];       /* uA/cm^2 */

  x1=datarr[Xstep][Ystep][4][step%2];        /* unitless */

  INa=datarr[Xstep][Ystep][5][step%2];       /* uA/cm^2 */

  m=datarr[Xstep][Ystep][6][step%2];         /* unitless */

  h=datarr[Xstep][Ystep][7][step%2];         /* unitless */

  Is=datarr[Xstep][Ystep][8][step%2];        /* uA/cm^2 */

  d=datarr[Xstep][Ystep][9][step%2];        /* unitless */

  f=datarr[Xstep][Ystep][10][step%2];        /* unitless */

  Cai=datarr[Xstep][Ystep][11][step%2];      /* mole/L */

  Isum=datarr[Xstep][Ystep][12][step%2];    /* uA/cm^2 */

  Diff=datarr[Xstep][Ystep][13][step%2];     /* mV/msec */

  Istim=datarr[Xstep][Ystep][14][step%2];    /* uA/cm^2 */

    /* IK1 */

    IK1t1 = 4 * (exp(0.04*(Vm+85))-1);

    IK1t2 = exp(0.08*(Vm+53)) + exp(0.04*(Vm+53));

    IK1t3 = 0.2 * (Vm+23);

    IK1t4 = 1 - exp(-0.04*(Vm+23));

    /* uA/cm^2  09/07/2005 */

    IK1 = (A) * (gK1) * (IK1t1/IK1t2 + IK1t3/IK1t4);

    datarr[Xstep][Ystep][2][step%2]=IK1 ;       /* uA/cm^2 */	 

    /* Ix1 */

    x1 = x1 + dx1dt*dt;

    datarr[Xstep][Ystep][4][step%2]=x1;        /* unitless */

    Ix1t1 = exp(0.04*(Vm+77)) - 1;

    Ix1t2 = exp(0.04*(Vm+35));

    Ix1 = ((gx1*Ix1t1) / (Ix1t2)) * (x1);        /* uA/cm^2 */

    datarr[Xstep][Ystep][3][step%2]=Ix1;       /* uA/cm^2 */	 

    /* INa */

    m =  m+ dmdt * dt;

    datarr[Xstep][Ystep][6][step%2]=m;         /* unitless */

    h =  h+dhdt * dt;

    datarr[Xstep][Ystep][7][step%2]=h;         /* unitless */

    /* *INa=((gNa)*(pow((m),3.0))*(h)*(*j)+*gNaC)*(Vm-ENa); */

    /* uA/cm^2, BR Na current */

    /* uA/cm^2, no gNaC or j gate in BRDR Na current */

    INa = ((gNa) * block[Xstep][Ystep] * (pow((m),3.0))*(h))*(Vm-ENa); 

    datarr[Xstep][Ystep][5][step%2]=INa;       /* uA/cm^2 */        

    /* Cai */

    Cai = Cai + dCaidt * dt;              /* moles/L */

    datarr[Xstep][Ystep][11][step%2]=Cai;      /* mole/L */	 

    /* Is */

    d = d+dddt * dt;

    datarr[Xstep][Ystep][9][step%2]=d;        /* unitless */

    f = f+dfdt * dt;

    datarr[Xstep][Ystep][10][step%2]=f;        /* unitless */

    Es = -82.3 - 13.0287 * log(Cai);
    Is = (gs) * block[Xstep][Ystep] * (d) * (f) * (Vm-Es);  /* uA/cm^2 */

    datarr[Xstep][Ystep][8][step%2]=Is;        /* uA/cm^2 */	 

    /* Vm */

    Isum = (IK1 + Ix1 + INa + Is);                     /* uA/cm^2 */

    datarr[Xstep][Ystep][12][step%2]=Isum;    /* uA/cm^2 */

    dVmdt = (Diff) - (1/(Cm))*(Isum-Istim);        /* mV/msec */

    datarr[Xstep][Ystep][1][step%2]=dVmdt;     /* mV/msec */   	 

    Vm = Vm + ((dVmdt)*dt);     

    datarr[Xstep][Ystep][0][step%2] = Vm;                       /* mV */


    datarr[Xstep][Ystep][13][step%2]=0.0;   // zero used diffusion terms     

   }  // end Xstep loop

}// end openmp pragma  

  }    // end Ystep loop


    bcs();      // apply Neumann boundary conditions by 'correcting' the diffusion matrix

#pragma omp parallel for private(Xstep,Ystep) num_threads(8)
    for (Xstep=1; Xstep<Nx+1; ++Xstep) {
{
//        #pragma omp for
        for (Ystep=1; Ystep<Ny+1; ++Ystep) {
//Response to MBEC Reviewer: Scattering To Collecting

/* Diffusion: Compute the new Vm's contribution to the next diffusion matrix */
/* (step-1)%2 for next timestep and step%2 for current timestep */
/* Dp stands for Dprime, the Laplacian multiplier form of the diffusion tensor D */
/* Remember: diffusion at ghost points doesn't matter  */

datarr[Xstep][Ystep][13][(step-1)%2] =
        datarr[Xstep][Ystep][13][(step-1)%2]
        - (2*Dp[0][0] + 2*Dp[1][1]) * datarr[Xstep][Ystep][0][step%2]
        - Dp[1][0] * datarr[Xstep-1][Ystep+1][0][step%2]
        + Dp[1][1] * datarr[Xstep][Ystep+1][0][step%2]
        + Dp[1][0] * datarr[Xstep+1][Ystep+1][0][step%2]
        + Dp[0][0] * datarr[Xstep+1][Ystep][0][step%2]
        - Dp[1][0] * datarr[Xstep+1][Ystep-1][0][step%2]
        + Dp[1][1] * datarr[Xstep][Ystep-1][0][step%2]
        + Dp[1][0] * datarr[Xstep-1][Ystep-1][0][step%2]
        + Dp[0][0] * datarr[Xstep-1][Ystep][0][step%2];
        }
    }

}
    if (step%rpN==0) {          // update user

      printf("%4.4e msec, Vm(%d,%d): %3.2f mV\n",derivarr[0],mNx,mNy,datarr[mNx][mNy][0][step%2]); 

      fflush(stdout);

    } 

    if (step%wN==0) output();   // write data to files

  }  // end time loop

  double cpu_end = rtclock();

  time_t gpu_end = time(NULL);    

  printf("total time is %.2lf\n",(double)(cpu_end-cpu_start));

  if (stable){

    printf("\nSimulation Finished!\n");

  }

  else {

    printf("\nSimulation Aborted!\n");

  }

  printf("Saving final conditions...\n\n");

  brfc();

  printf("         tfinal: %5.3f msec\n",tfinal);

  printf("     Final time: %5.3f msec\n",derivarr[0]);

  printf("         Nsteps: %10.2f\n",Nsteps);

  printf("Number of steps: %d\n",step);

  printf("             Nx: %d\n",Nx);

  printf("             Ny: %d\n",Ny);

  closefiles();

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



double abfun(double vv,int i)

{

  double t1, t2, t3;

  double ab;

  t1=C[i][0]*exp(C[i][1]*(vv+C[i][2]));

  t2=C[i][3]*(vv+C[i][4]);

  t3=C[i][6]+exp(C[i][5]*(vv+C[i][2]));

  ab=(t1+t2)/t3;

  return ab;

}



/******************************************************/

void stability(void)

// stability check

{

  if (datarr[Xstep][Ystep][0][step%2]-datarr[Xstep][Ystep][0][step%2]!=0){

    stable=0;  // UNSTABLE!!

    printf("Unstable at node %d,%d !! Check dt and dx.\n",Xstep,Ystep);

  }    

}





void stimulate(void)

{

  int i;



  /* Remember: stimarr is structured as [0:Nx-1][0:Ny-1][0:stimnum-1][0:1]

               AND stimarr is not defined at ghost nodes!! 

	       So, to colocate datarr and stimarr use

	       stimarr[Xstep-1][Ystep-1][i][0:1] and 

	       datarr[Xstep][Ystep][15][(step-1)%2]   */



  datarr[Xstep][Ystep][14][(step-1)%2]=0;

  for (i=0;i<stimnum;++i){

    if ((stimarr[Xstep-1][Ystep-1][i][0]!=0)&&(derivarr[0]>=stimarr[Xstep-1][Ystep-1][i][0])&&(derivarr[0]<=stimarr[Xstep-1][Ystep-1][i][0]+stimint)){

      if (stimarr[Xstep-1][Ystep-1][i][1]==0.0) {

  //      fprintf(stiminfofile,"Applying Stimulus at %4.3f msec to node (%d,%d)\n",derivarr[0],Xstep,Ystep); 
        printf("Applying Stimulus at %4.3f msec to node (%d,%d)\n",derivarr[0],Xstep,Ystep); 

	//fflush(stiminfofile);

	stimarr[Xstep-1][Ystep-1][i][1]=1.0;

      } //end if

      datarr[Xstep][Ystep][14][(step-1)%2]=Istimamp; 

    }   //end if

  }     //end for

  

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





void buildedgestim(void){

/* line stimulus at an edge*/

  int i,j,k;

  i=1;

  printf("Line stimulus at left vertical edge...");

  for (k=0;k<stimnum;++k){

    for (j=0;j<Ny;++j){  

      stimarr[i][j][k][0]=stimes[k];  // msec

    }

  }

  

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



void build2ptstims(void){ 

/* Two point stimuli  */

  int i,j,k;

  int Nxx,Nyy;

  double stimsizeir;

  double radius;

  

  Nxx=(unsigned int)(floor(Nx/4));

  Nyy=(unsigned int)(floor(3*Ny/4));

  printf("First point stimulus centered at %d,%d\n",Nxx,Nyy);

  stimsizeir=floor(stimsize1/dx);

  printf("First point stimulus radius: %4.3f cm, %4.3f pixels\n",stimsize1,stimsizeir);

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

  

  Nxx=(unsigned int)(floor(3*Nx/4));

  Nyy=(unsigned int)(floor(3*Ny/4));

  printf("Second point stimulus centered at %d,%d\n",Nxx,Nyy);

  printf("Second point stimulus radius: %4.3f cm, %4.3f pixels\n",stimsize1,stimsizeir);

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





void buildbarstim1(void){ 

/* bar stimulus in the middle of the domain*/

  int i,j,k;

  int Nxx,Nyy;

  int stimsizeix, stimsizeiy;

  Nxx=(unsigned int)(floor(Nx/2));

  Nyy=(unsigned int)(floor(Ny/2));

  printf("Bar stimuli centered at %d,%d\n",Nxx,Nyy);

  stimsizeix=(unsigned int)(floor(stimsize1/(2*dx)));

  stimsizeiy=(unsigned int)(floor(stimsize2/(2*dy)));

  printf("Bar stimulus size: %4.3f X %4.3f cm\n",stimsize1,stimsize2);

  printf("Bar stimulus applied to x nodes %d:%d\n",Nxx-stimsizeix+1,Nxx+stimsizeix+1);  // add 1 because of ghost nodes

  printf("Bar stimulus applied to y nodes %d:%d\n",Nyy-stimsizeiy+1,Nyy+stimsizeiy+1);  // add 1 because of ghost nodes

  printf("Bar stimulus unitless dimensions: %d,%d\n",2*stimsizeix+1,2*stimsizeiy+1);

  

  for (k=0;k<stimnum;++k){

    for (i=Nxx-stimsizeix;i<=Nxx+stimsizeix;++i){  

      for (j=Nyy-stimsizeiy;j<=Nyy+stimsizeiy;++j){  

        stimarr[i][j][k][0]=stimes[k];  

      }

    }

  }

}



void buildbarstim2(void){ 

/* bar stimulus at left side of the domain*/

  int i,j,k;

  int Nxx,Nyy;

  int stimsizeix, stimsizeiy;

  Nxx=0;

  Nyy=(unsigned int)(floor(Ny/2));

  printf("Bar stimuli centered at %d,%d\n",Nxx+1,Nyy+1);  // add 1 because of ghost nodes

  stimsizeix=(unsigned int)(floor(stimsize1/(dx)));

  stimsizeiy=(unsigned int)(floor(stimsize2/(2*dy)));

  printf("Bar stimulus size: %4.3f X %4.3f cm\n",stimsize1,stimsize2);

  printf("Bar stimulus applied to x nodes %d:%d\n",Nxx+1,Nxx+stimsizeix+1);  // add 1 because of ghost nodes

  printf("Bar stimulus applied to y nodes %d:%d\n",Nyy-stimsizeiy+1,Nyy+stimsizeiy+1);  // add 1 because of ghost nodes

  printf("Bar stimulus unitless dimensions: %d,%d\n",stimsizeix+1,2*stimsizeiy+1);

  

  for (k=0;k<stimnum;++k){

    for (i=Nxx;i<=Nxx+stimsizeix;++i){  

      for (j=Nyy-stimsizeiy;j<=Nyy+stimsizeiy;++j){  

        stimarr[i][j][k][0]=stimes[k];  

      }

    }

  }

}





void buildcrossstim(void){ 

/* crossfield stimulus */

  int i,j,k;

  int Nyy1, Nyy2, Nxx; 

  Nyy1=(unsigned int)(floor(Ny/4));

  Nyy2=(unsigned int)(floor(Ny/2));

  Nxx=(unsigned int)(floor(Nx/2));

  

  for (k=0;k<stimnum;++k){

    for (i=0;i<Nx;++i){

      for (j=0;j<Nyy1;++j){

        stimarr[i][j][k][0]=stimes[k];   // msec

      }

    }

  }

  

}







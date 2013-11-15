/* MWKay, 8/27/2002                                          */
/* MWKay, 9/07/2005                                          */


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
// datarr[][][14][] is Istim   (uA/cm^2) Istim should always be the last variable in datarr
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

void brgates(void) 
{

  double abfun (double,int);
  /* Ix1 */
  double ax1,bx1,tx1,x1inf,*dx1dt,*x1;
  /* INa */
  double am,bm,tm,minf,ah,bh,th,hinf,*dmdt,*m,*dhdt,*h;
  /* Is */
  double ad,bd,td,dinf,af,bf,tf,finf,*dddt,*d,*dfdt,*f;
  /* Cai */
  double *dCaidt,*kCa,*Cai, *sigma;
  /* Is */
  double *Is;
  /* Vm */
  double *Vm;
  
  /* Need these from datarr to update derivatives in derivarr */
  /* datarr values are not altered here */
  /* these are initial conditions for step=1 */
  Vm=&datarr[Xstep][Ystep][0][(step-1)%2];   /* mV */         
  x1=&datarr[Xstep][Ystep][4][(step-1)%2];   /* unitless */
  m=&datarr[Xstep][Ystep][6][(step-1)%2];    /* unitless */
  h=&datarr[Xstep][Ystep][7][(step-1)%2];    /* unitless */
  Is=&datarr[Xstep][Ystep][8][(step-1)%2];   /* uA/cm^2 */
  d=&datarr[Xstep][Ystep][9][(step-1)%2];   /* unitless */
  f=&datarr[Xstep][Ystep][10][(step-1)%2];   /* unitless */
  Cai=&datarr[Xstep][Ystep][11][(step-1)%2]; /* moles/L */
  
  /* Derivatives */
  //   t=&derivarr[0]      /* msec */
  dx1dt=&derivarr[1];    /* unitless */
  dmdt=&derivarr[2];     /* unitless */
  dhdt=&derivarr[3];     /* unitless */
  dddt=&derivarr[4];     /* unitless */
  dfdt=&derivarr[5];     /* unitless */
  dCaidt=&derivarr[6];   /* mole/L */
  
  /* Constants */
  kCa=&constarr[6];      /* msec^-1 */
  sigma=&constarr[11];   /* unitless */
  
  /* Ix1  */ 
  ax1=abfun(*Vm,0);
  bx1=abfun(*Vm,1);
  tx1=1/(ax1+bx1);
  x1inf=ax1*tx1;
  *dx1dt=(x1inf-(*x1))/tx1;
	
  /* INa */
  am=abfun(*Vm,2);
  bm=abfun(*Vm,3);
  tm=1/(am+bm);
  minf=am*tm;
  *dmdt=(minf-(*m))/tm;
  
  ah=abfun(*Vm,4);
  bh=abfun(*Vm,5);
  th=1/(ah+bh);
  hinf=ah*th;
  *dhdt=(hinf-(*h))/th;
    
  /* Is */
  ad=abfun(*Vm,8);
  bd=abfun(*Vm,9);
  td=(1/(ad+bd))*(*sigma);
  dinf=ad*td;
  *dddt=(dinf-(*d))/td;
  
  af=abfun(*Vm,10);
  bf=abfun(*Vm,11);
  tf=(1/(af+bf))*(*sigma);
  finf=af*tf;
  *dfdt=(finf-(*f))/tf;
  
  /* Cai */
  *dCaidt=(-1e-7)*(*Is)+(*kCa)*((1e-7)-(*Cai));  /* mole/L */
  
}

void brcurrents(void) 
{
  /* IK1 */
  double IK1t1,IK1t2,IK1t3,IK1t4,*gK1,*IK1;
  /* Ix1 */
  double Ix1t1,Ix1t2,*gx1,*Ix1,*dx1dt,*x1;
  /* INa */
  double *gNa,*gNaC,*ENa,*INa,*dmdt,*m,*dhdt,*h, *A;
  /* Cai */
  double *dCaidt,*Cai;
  /* Is */
  double Es,*gs,*Is,*dddt,*d,*dfdt,*f;
  /* Other currents */
  double *Isum,*Istim;
  /* Vm */
  double *Cm, *Vm, *dVmdt;
  /* Diffusion */
  double *Diff;
  
  /* Constants */
  gK1=&constarr[0];            /* mmho/cm^2 */
  gNa=&constarr[1];            /* mmho/cm^2 */
  ENa=&constarr[2];            /* mV */  
  gx1=&constarr[3];            /* mmho/cm^2 */
  gs=&constarr[4];             /* mmho/cm^2 */
  Cm=&constarr[5];             /* uF/cm^2 */
  gNaC=&constarr[7];           /* mmho/cm^2 */
  A=&Afield[Xstep-1][Ystep-1]; /* unitless */
  
  /* Need these from derivarr to update datarr */
  /* derivarr values are not altered here */
  //    t=&derivarr[0]     /* msec */
  dx1dt=&derivarr[1];      /* unitless */
  dmdt=&derivarr[2];       /* unitless */
  dhdt=&derivarr[3];       /* unitless */
  dddt=&derivarr[4];       /* unitless */
  dfdt=&derivarr[5];       /* unitless */
  dCaidt=&derivarr[6];     /* mole/L */
  
  /* copy previous timestep data into current datarr */
  /* such that previous Vm is used to update next Vm */
  int jj;
  for (jj=0; jj<varnum-2; jj++)                                               /* varnum-1 b/c don't want to overwrite */
    datarr[Xstep][Ystep][jj][step%2]=datarr[Xstep][Ystep][jj][(step-1)%2];    /* the stimulus and diffusion in last 2 elements of datarr */
  
  /* datarr values to update */ 
  Vm=&datarr[Xstep][Ystep][0][step%2];        /* mV */
  dVmdt=&datarr[Xstep][Ystep][1][step%2];     /* mV/msec */
  IK1=&datarr[Xstep][Ystep][2][step%2];       /* uA/cm^2 */
  Ix1=&datarr[Xstep][Ystep][3][step%2];       /* uA/cm^2 */
  x1=&datarr[Xstep][Ystep][4][step%2];        /* unitless */
  INa=&datarr[Xstep][Ystep][5][step%2];       /* uA/cm^2 */
  m=&datarr[Xstep][Ystep][6][step%2];         /* unitless */
  h=&datarr[Xstep][Ystep][7][step%2];         /* unitless */
  Is=&datarr[Xstep][Ystep][8][step%2];        /* uA/cm^2 */
  d=&datarr[Xstep][Ystep][9][step%2];        /* unitless */
  f=&datarr[Xstep][Ystep][10][step%2];        /* unitless */
  Cai=&datarr[Xstep][Ystep][11][step%2];      /* mole/L */
  Isum=&datarr[Xstep][Ystep][12][step%2];    /* uA/cm^2 */
  Diff=&datarr[Xstep][Ystep][13][step%2];     /* mV/msec */
  Istim=&datarr[Xstep][Ystep][14][step%2];    /* uA/cm^2 */
  
  /* IK1 */
  IK1t1=4*(exp(0.04*(*Vm+85))-1);
  IK1t2=exp(0.08*(*Vm+53))+exp(0.04*(*Vm+53));
  IK1t3=0.2*(*Vm+23);
  IK1t4=1-exp(-0.04*(*Vm+23));
  *IK1=(*A)*(*gK1)*(IK1t1/IK1t2+IK1t3/IK1t4);     /* uA/cm^2  09/07/2005 */
  
  /* Ix1 */
  *x1=*x1+*dx1dt*dt;
  Ix1t1=exp(0.04*(*Vm+77))-1;
  Ix1t2=exp(0.04*(*Vm+35));
  *Ix1=((*gx1*Ix1t1)/(Ix1t2))*(*x1);        /* uA/cm^2 */
 
  /* INa */
  *m=*m+*dmdt*dt;
  *h=*h+*dhdt*dt;
/* *INa=((*gNa)*(pow((*m),3.0))*(*h)*(*j)+*gNaC)*(*Vm-*ENa); */ /* uA/cm^2, BR Na current */
  *INa=((*gNa)*block[Xstep][Ystep]*(pow((*m),3.0))*(*h))*(*Vm-*ENa);                /* uA/cm^2, no gNaC or j gate in BRDR Na current */
  
  /* Cai */ 
  *Cai=*Cai+*dCaidt*dt;              /* moles/L */
  
  /* Is */ 
  *d=*d+*dddt*dt;
  *f=*f+*dfdt*dt;
  Es=-82.3-13.0287*log(*Cai);
  *Is=(*gs)*block[Xstep][Ystep]*(*d)*(*f)*(*Vm-Es);         /* uA/cm^2 */
  
  /* Vm */
  *Isum=(*IK1+*Ix1+*INa+*Is);                     /* uA/cm^2 */
  *dVmdt=(*Diff)-(1/(*Cm))*(*Isum-*Istim);        /* mV/msec */  
  *Vm=*Vm+((*dVmdt)*dt);                          /* mV */
  
  /* Diffusion: Compute the new Vm's contribution to the next diffusion matrix */
  /* (step-1)%2 for next timestep and step%2 for current timestep */
  /* Dp stands for Dprime, the Laplacian multiplier form of the diffusion tensor D */
  /* Remember: diffusion at ghost points doesn't matter  */
  
  datarr[Xstep][Ystep][13][(step-1)%2] -= (2*(Dp[0][0]) + 2*(Dp[1][1]))*(*Vm);
  datarr[Xstep-1][Ystep+1][13][(step-1)%2] -= (Dp[1][0])*(*Vm);
  datarr[Xstep][Ystep+1][13][(step-1)%2] += (Dp[1][1])*(*Vm); 
  datarr[Xstep+1][Ystep+1][13][(step-1)%2] += (Dp[1][0])*(*Vm);
  datarr[Xstep+1][Ystep][13][(step-1)%2] += (Dp[0][0])*(*Vm);
  datarr[Xstep+1][Ystep-1][13][(step-1)%2] -= (Dp[1][0])*(*Vm);
  datarr[Xstep][Ystep-1][13][(step-1)%2] += (Dp[1][1])*(*Vm);
  datarr[Xstep-1][Ystep-1][13][(step-1)%2] += (Dp[1][0])*(*Vm);
  datarr[Xstep-1][Ystep][13][(step-1)%2] += (Dp[0][0])*(*Vm);
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

void bcs(void) 
{
  int ii;
  double R0, R1;
  
  R0=(D[1][0]/D[0][0])*(dx/dy);
  R1=(D[1][0]/D[1][1])*(dy/dx);
  
  
  if (BC==1){   // Slab
    /* First set Vm at ghost nodes */
    datarr[0][1][0][step%2]=datarr[2][1][0][step%2];
    datarr[0][0][0][step%2]=datarr[2][2][0][step%2];
    datarr[1][0][0][step%2]=datarr[1][2][0][step%2];
    datarr[Nx][0][0][step%2]=datarr[Nx][2][0][step%2];
    datarr[Nx+1][0][0][step%2]=datarr[Nx-1][2][0][step%2];
    datarr[Nx+1][1][0][step%2]=datarr[Nx-1][1][0][step%2];
    datarr[Nx+1][Ny][0][step%2]=datarr[Nx-1][Ny][0][step%2];
    datarr[Nx+1][Ny+1][0][step%2]=datarr[Nx-1][Ny-1][0][step%2];
    datarr[Nx][Ny+1][0][step%2]=datarr[Nx][Ny-1][0][step%2];
    datarr[1][Ny+1][0][step%2]=datarr[1][Ny-1][0][step%2];
    datarr[0][Ny+1][0][step%2]=datarr[2][Ny-1][0][step%2];
    datarr[0][Ny][0][step%2]=datarr[2][Ny][0][step%2];  
    for (ii=2;ii<Nx;++ii){            /* decouple these loops b/c Nx might not equal Ny */
      datarr[ii][Ny+1][0][step%2]=datarr[ii][Ny-1][0][step%2]+R1*(datarr[ii-1][Ny][0][step%2]-datarr[ii+1][Ny][0][step%2]);  /* Eq 3 in notes */
      datarr[ii][0][0][step%2]=datarr[ii][2][0][step%2]-R1*(datarr[ii-1][1][0][step%2]-datarr[ii+1][1][0][step%2]);          /* Eq 2 in notes */
    }
    for (ii=2;ii<Ny;++ii){           /* decouple these loops b/c Nx might not equal Ny */
      datarr[0][ii][0][step%2]=datarr[2][ii][0][step%2]-R0*(datarr[1][ii-1][0][step%2]-datarr[1][ii+1][0][step%2]);           /* Eq 1 in notes */
      datarr[Nx+1][ii][0][step%2]=datarr[Nx-1][ii][0][step%2]+R0*(datarr[Nx][ii-1][0][step%2]-datarr[Nx][ii+1][0][step%2]);   /* Eq 4 in notes */
    }
    
    /* Now compute boundary node contributions to the next diffusion term: mV/msec */ 
    for (ii=2;ii<Nx;++ii){           /* decouple these loops b/c Nx might not equal Ny */
      datarr[ii][1][13][(step-1)%2] += Dp[1][0]*datarr[ii-1][0][0][step%2]+Dp[1][1]*datarr[ii][0][0][step%2]-Dp[1][0]*datarr[ii+1][0][0][step%2];
      datarr[ii][Ny][13][(step-1)%2] += Dp[1][0]*datarr[ii+1][Ny+1][0][step%2]+Dp[1][1]*datarr[ii][Ny+1][0][step%2]-Dp[1][0]*datarr[ii-1][Ny+1][0][step%2];
    }
    for (ii=2;ii<Ny;++ii){           /* decouple these loops b/c Ny might not equal Ny */
      datarr[1][ii][13][(step-1)%2] += Dp[1][0]*datarr[0][ii-1][0][step%2]+Dp[0][0]*datarr[0][ii][0][step%2]-Dp[1][0]*datarr[0][ii+1][0][step%2];
      datarr[Nx][ii][13][(step-1)%2] += Dp[1][0]*datarr[Nx+1][ii+1][0][step%2]+Dp[0][0]*datarr[Nx+1][ii][0][step%2]-Dp[1][0]*datarr[Nx+1][ii-1][0][step%2];
    }
    /* Now for corner nodes (1,1),(0,Ny),(Nx,0),(Nx,Ny) */
    datarr[1][1][13][(step-1)%2] += Dp[1][1]*datarr[1][0][0][step%2]-Dp[1][0]*datarr[2][0][0][step%2]+Dp[1][0]*datarr[0][0][0][step%2]+Dp[0][0]*datarr[0][1][0][step%2]-Dp[1][0]*datarr[0][2][0][step%2];    
    datarr[1][Ny][13][(step-1)%2] += Dp[1][0]*datarr[0][Ny-1][0][step%2]+Dp[0][0]*datarr[0][Ny][0][step%2]-Dp[1][0]*datarr[0][Ny+1][0][step%2]+Dp[1][1]*datarr[1][Ny+1][0][step%2]+Dp[1][0]*datarr[2][Ny+1][0][step%2];
    datarr[Nx][Ny][13][(step-1)%2] += Dp[1][1]*datarr[Nx][Ny+1][0][step%2]-Dp[1][0]*datarr[Nx-1][Ny+1][0][step%2]+Dp[1][0]*datarr[Nx+1][Ny+1][0][step%2]+Dp[0][0]*datarr[Nx+1][Ny][0][step%2]-Dp[1][0]*datarr[Nx+1][Ny-1][0][step%2];
    datarr[Nx][1][13][(step-1)%2] += Dp[1][0]*datarr[Nx+1][2][0][step%2]+Dp[0][0]*datarr[Nx+1][1][0][step%2]-Dp[1][0]*datarr[Nx+1][0][0][step%2]+Dp[1][1]*datarr[Nx][0][0][step%2]+Dp[1][0]*datarr[Nx-1][0][0][step%2];
  }
}









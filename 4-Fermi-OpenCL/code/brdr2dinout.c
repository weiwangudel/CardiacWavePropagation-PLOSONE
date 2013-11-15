/* MWKay, 3/26/2003                                          */
/* MWKay, 9/07/2005                                          */
/* MWKay, 9/08/2005                                          */

void readAfield();
double abfun (double,int);

void initialize(void)
/* Read constants and initial conditions from br2dtask.dat */
{
  int i,j,k,l;
  double pi=3.141592653589793;
  int icewidthn, icbwidthn, iclengthactn, iclengthblockn, middlex, middley, b1, b2;
  FILE *fp;
  
  if((fp=fopen("brdr2dtask.dat","r"))==NULL) {
    printf("Cannot open task file: brdr2dtask.dat \n");
    exit(1);
  }
  else {
  
    // initialize constant array
    printf("Initializing constant array ... \n");   
    constarr=(double *)malloc((unsigned int)(constnum*sizeof(double)));
    for (i=0;i<constnum;++i) constarr[i]=0;
  
    // initialize derivative array
    printf("Initializing derivative array ... \n"); 
    derivarr=(double *)malloc((unsigned int)(derivnum*sizeof(double)));
    for (i=0;i<derivnum;++i) derivarr[i]=0;

	
        
    next_line(fp);
    next_line(fp);
    fscanf(fp,"%lf",&tfinal); next_line(fp);        /* tfinal (msec) */  
    fscanf(fp,"%lf",&dt); next_line(fp);            /* dt (mesc) */
    fscanf(fp,"%lf",&Lx); next_line(fp);            /* Lx (cm) */
    fscanf(fp,"%d",&Nx); next_line(fp);            /* Nx (unitless integer) */
    fscanf(fp,"%lf",&Ly); next_line(fp);            /* Ly (cm) */
    fscanf(fp,"%d",&Ny); next_line(fp);            /* Ny (unitless integer) */
    fscanf(fp,"%lf",&constarr[0]); next_line(fp);   /* gK1 */
    fscanf(fp,"%lf",&constarr[1]); next_line(fp);   /* gNa */
    fscanf(fp,"%lf",&constarr[2]); next_line(fp);   /* ENa */
    fscanf(fp,"%lf",&constarr[3]); next_line(fp);   /* gx1 */
    fscanf(fp,"%lf",&constarr[4]); next_line(fp);   /* gs */
    fscanf(fp,"%lf",&constarr[5]); next_line(fp);   /* Cm */
    fscanf(fp,"%lf",&constarr[6]); next_line(fp);   /* kCa */
    fscanf(fp,"%lf",&constarr[7]); next_line(fp);   /* gNaC */
    fscanf(fp,"%lf",&constarr[8]); next_line(fp);   /* Dpara   (cm^2/msec) */
    fscanf(fp,"%lf",&constarr[9]); next_line(fp);   /* Dperpen (cm^2/msec) */
    fscanf(fp,"%lf",&constarr[10]); next_line(fp);  /* theta   (degrees)   */
    fscanf(fp,"%lf",&constarr[11]); next_line(fp);  /* sigma   (unitless)   */
    fscanf(fp,"%lf",&constarr[12]); next_line(fp);  /* A       (unitless)   */
    printf("Initializing 3D derivative array ... \n"); 
	deriv3darr = (double ***)malloc((unsigned int)((Nx)*sizeof(double**)));   
	assert(deriv3darr != NULL);
      for (i=0;i<Nx;++i){
        deriv3darr[i]=(double **)malloc((unsigned int)((Ny)*sizeof(double*)));
		assert(deriv3darr[i] != NULL);
        for (j=0;j<Ny;++j){
          deriv3darr[i][j]=(double *)malloc((unsigned int)(derivnum*sizeof(double)));
		  assert(deriv3darr[i][j] != NULL);
	 	 }
       }
	for (i=0; i<Nx; ++i)
	{
		for (j=0; j<Ny; ++j) 
		{
			for (k=0; k<derivnum; ++k)
			{
				deriv3darr[i][j][k] = 0.0;
			}
		}
	}
    
    Nsteps=ceil(tfinal/dt);      // total number of time steps
    dx=Lx/Nx;                    // (cm)
    dy=Ly/Ny;                    // (cm)
    middlex=(unsigned int)(floor(Nx/2));
    middley=(unsigned int)(floor(Ny/2));  /* changed to 1/3 on 9/9/05 */
    
    // Compute Diffusion Tensor  (cm^2/msec)
    D[0][0]=constarr[8]*pow(cos(constarr[10]*pi/180),2)+constarr[9]*pow(sin(constarr[10]*pi/180),2);
    D[1][0]=(constarr[8]-constarr[9])*cos(constarr[10]*pi/180)*sin(constarr[10]*pi/180);
    D[0][1]=D[1][0];
    D[1][1]=constarr[8]*pow(sin(constarr[10]*pi/180),2)+constarr[9]*pow(cos(constarr[10]*pi/180),2);
    
    // Compute Laplacian multipliers: Dp for Dprime
    Dp[0][0]=D[0][0]/(dx*dx);          /* msec^-1 */
    Dp[1][0]=D[1][0]/(2*dx*dy);        /* msec^-1 */
    Dp[0][1]=Dp[1][0];                 /* msec^-1 */
    Dp[1][1]=D[1][1]/(dy*dy);          /* msec^-1 */
        
    // Allocate memory and define datarr size
    printf("Allocating memory for data array ... \n");                                                    
    datarr=(double ****)malloc((unsigned int)(datarr4dim*sizeof(double***))); // Nx+2 rows, Ny+2 columns, depth of varnum, 4th dim of datarr4dim
    for (k=0;k<datarr4dim;++k){
          datarr[k]=(double ***)malloc((unsigned int)(varnum*sizeof(double**)));
      for (j=0;j<varnum;++j){
        datarr[k][j]=(double **)malloc((unsigned int)((Nx+2)*sizeof(double*)));
    for (i=0;i<Nx+2;++i){
      	datarr[k][j][i]=(double *)malloc((unsigned int)((Ny+2)*sizeof(double)));
	}
      }
    }
       
    //datarr_gpu=(double ****)malloc((unsigned int)((Nx+2)*sizeof(double***)));   // Nx+2 rows, Ny+2 columns, depth of varnum, 4th dim of datarr4dim
    //for (i=0;i<Nx+2;++i){
    //  datarr_gpu[i]=(double ***)malloc((unsigned int)((Ny+2)*sizeof(double**)));
    //  for (j=0;j<Ny+2;++j){
    //    datarr_gpu[i][j]=(double **)malloc((unsigned int)(varnum*sizeof(double*)));
    //    for (k=0;k<varnum;++k){
    //      datarr_gpu[i][j][k]=(double *)malloc((unsigned int)(datarr4dim*sizeof(double)));
    //    }
    //  }
    //}
    // initialize datarr
    printf("Initializing data array ... \n");   
    for (l=0;l<datarr4dim;++l) {
      for (k=0;k<varnum;++k){ 
        for (i=0;i<Nx+2;++i){  
          for (j=0;j<Ny+2;++j){
			 datarr[l][k][i][j]=0.0;
		}
	}
      }
    }
 
    printf("Initializing block matrix ... \n");
    block=(int **)malloc((unsigned int)((Nx+2)*sizeof(int*)));         
    for (i=0;i<Nx+2;++i) block[i]=(int *)malloc((unsigned int)((Ny+2)*sizeof(int)));
    for (i=0;i<Nx+2;++i){  
      for (j=0;j<Ny+2;++j){
        block[i][j]=1;
      }
    }
    
    next_line(fp);
    next_line(fp);
    next_line(fp);
    /* Fill first node with initial conditions: Read ics */
    fscanf(fp,"%lf",&datarr[0][0][1][1]); next_line(fp);    /* Vm (mV) */
    fscanf(fp,"%lf",&datarr[0][11][1][1]); next_line(fp);   /* Cai   (mole/L) */
    /* Get spiral stimulus options */
    fscanf(fp,"%d",&ictype); next_line(fp);          /* type of initial conditions (0, 1, or 2) */
    next_line(fp);
    next_line(fp);
    next_line(fp);
    /* Get block stuff */
    fscanf(fp,"%lf",&icewidth); next_line(fp);       /* width of excited ic bar (cm) */
    fscanf(fp,"%lf",&icbwidth); next_line(fp);       /* width of block ic bar (cm) */
    fscanf(fp,"%lf",&iclengthact); next_line(fp);   /* length of ic active bar (cm) */
    fscanf(fp,"%lf",&iclengthblock); next_line(fp); /* length of ic block bar (cm) */
    fscanf(fp,"%d",&blocktimenum); next_line(fp);  /* number of block times */
    
    if (blocktimenum<0){
      printf("Invalid number of block times: %d \n",blocktimenum);
      exit(1);}
    if ((blocktimenum==0)&&(ictype==1)){
      printf("Invalid blocktimenum %d for spiral ics! \n",blocktimenum);
      exit(1);} 
    else if (blocktimenum>0){
      /* Allocate memory for block time array. */
      blocktimes=(double **)malloc((unsigned int)(blocktimenum*sizeof(double*)));
      for (i=0;i<blocktimenum;++i) blocktimes[i]=(double *)malloc((unsigned int)(2*sizeof(double)));
      /* Initialize blocktimes  */
      for (i=0;i<blocktimenum;++i){  
        blocktimes[i][0]=0.0;
	blocktimes[i][1]=0.0;
      }
      for (i=0;i<blocktimenum;++i) fscanf(fp,"%lf",&blocktimes[i][0]);      // read block times from br2dtask.dat
    }
     
    /* Fill first node with initial conditions: Compute ics */
    /* Generate initial conditions for gate variables from infinity constants */
    datarr[0][4][1][1]=abfun(datarr[0][0][1][1],0)/(abfun(datarr[0][0][1][1],0)+abfun(datarr[0][0][1][1],1));      // x1 initial condition
    datarr[0][6][1][1]=abfun(datarr[0][0][1][1],2)/(abfun(datarr[0][0][1][1],2)+abfun(datarr[0][0][1][1],3));      // m initial condition
    datarr[0][7][1][1]=abfun(datarr[0][0][1][1],4)/(abfun(datarr[0][0][1][1],4)+abfun(datarr[0][0][1][1],5));      // h initial condition
    datarr[0][9][1][1]=abfun(datarr[0][0][1][1],8)/(abfun(datarr[0][0][1][1],8)+abfun(datarr[0][0][1][1],9));     // d initial condition
    datarr[0][10][1][1]=abfun(datarr[0][0][1][1],10)/(abfun(datarr[0][0][1][1],10)+abfun(datarr[0][0][1][1],11));  // f initial condition

    // Propagate initial conditions to all other nodes
    for (i=1;i<Nx+1;++i){
      for (j=1;j<Ny+1;++j){
        if (i>1 | j>1) { for (k=0;k<varnum;++k) datarr[0][k][i][j]=datarr[0][k][1][1]; }
      }
    }
     
    next_line(fp);
    next_line(fp);
    next_line(fp);
    next_line(fp);
    fscanf(fp,"%d",&stimnum); next_line(fp);           /* num of stimuli */
    fscanf(fp,"%lf",&Istimamp); next_line(fp);         /* stimulus amplitude (uA/cm^2) */
    fscanf(fp,"%lf",&stimint); next_line(fp);          /* stimulus interval, duration (msec) */
    fscanf(fp,"%lf",&stimsize1); next_line(fp);        /* first stimulus size (cm) */
    fscanf(fp,"%lf",&stimsize2); next_line(fp);        /* second stimulus size (cm) */
    
    if (ictype==1) {
      printf("Establishing initial conditions for spiral wave ... \n");      
      icewidthn=(unsigned int)(round(icewidth/dy));
      icbwidthn=(unsigned int)(round(icbwidth/dy));
      iclengthactn=(unsigned int)(floor(iclengthact/dx));  
      iclengthblockn=(unsigned int)(floor(iclengthblock/dx));  
      b1=middley-icewidthn;
      if (b1<1) b1=1;
      b2=middley+icbwidthn+1;
      if (b2>Ny) b2=Ny;
      for (i=1;i<=iclengthactn;++i){
        for (j=b1;j<=middley;++j){
          datarr[0][0][i][j]=-20;     // Vm 
	  datarr[0][6][i][j]=1;       // m 
	}
      }
      for (i=1;i<=iclengthblockn;++i){
        for (j=middley+1;j<=b2;++j){
	  block[i][j]=0;  // set to zero to turn on block
        }
      }
      printf("b1: %d, b2: %d, middley: %d \n",b1,b2,middley);
    }
    else if (ictype==0) printf("Using normal initial conditions ... \n");
    else if (ictype==2) printf("Using final conditions of a previous run to establish initial conditions ... \n");
    else {
      printf("Invalid ic type value in brdr2dtask.dat: %d \n",ictype);
      exit(1);}
    
    if (stimnum<0){
      printf("Invalid stimulus number: %d \n",stimnum);
      exit(1);}
    else if (stimnum>0){
      /* Allocate memory for stimulus matrix. Stimarr is undefined at ghost nodes */
      /* Nx rows, Ny columns, depth of stimnum, 4th dim of 2 */
      printf("Initializing stimulus array for %d stimul[us/i] ... \n",stimnum);
      stimarr=(double ****)malloc((unsigned int)((Nx)*sizeof(double***)));   
      for (i=0;i<Nx;++i){
        stimarr[i]=(double ***)malloc((unsigned int)((Ny)*sizeof(double**)));
        for (j=0;j<Ny;++j){
          stimarr[i][j]=(double **)malloc((unsigned int)(stimnum*sizeof(double*)));
          for (k=0;k<stimnum;++k){
            stimarr[i][j][k]=(double *)malloc((unsigned int)(2*sizeof(double)));
	  }
        }
      } 
      /* Initialize stimarr  */
      for (i=0;i<Nx;++i){  
        for (j=0;j<Ny;++j){
          for (k=0;k<stimnum;++k){ 
	    for (l=0;l<2;++l) stimarr[i][j][k][l]=0.0;
	  }
        }
      }
      stimes=(double *)malloc((unsigned int)(stimnum*sizeof(double)));
      for (i=0;i<stimnum;++i) fscanf(fp,"%lf",&stimes[i]);      // read stimulus times from br2dtask.dat
    }

    next_line(fp);
    next_line(fp);
    next_line(fp);
    next_line(fp);
    fscanf(fp,"%lf",&msecW); next_line(fp);          /* msec per write to datafile */
    fscanf(fp,"%d",&wdN); next_line(fp);             /* spatial sampling for writes to datafiles */
    wN=(unsigned int)(floor(Nsteps*msecW/tfinal));   /* time steps per write to datafile */
    nodeWx=wdN*dx;                                   /* dx per write to datafile (cm) */
    nodeWy=wdN*dy;                                   /* dy per write to datafile (cm) */
    
    next_line(fp);
    next_line(fp);
    next_line(fp);
    fscanf(fp,"%lf",&msecRP); next_line(fp);          /* msec per user update (print time to screen) */  
    rpN=(unsigned int)(floor(Nsteps*msecRP/tfinal)); /* time steps per user update (print time to screen) */
    fscanf(fp,"%d",&mNx); next_line(fp);             /* x index of node to monitor */
    fscanf(fp,"%d",&mNy); next_line(fp);             /* y index of node to monitor */
    
    if (mNx==-1) mNx=Nx;
    else if (mNx==-2) mNx=middlex;
    else if (mNx>Nx) mNx=Nx;
    else if (mNx==0) mNx=1;
    else if (mNx<-2) mNx=1;
    
    if (mNy==-1) mNy=Ny;
    else if (mNy==-2) mNy=middley;
    else if (mNy>Ny) mNy=Ny;
    else if (mNy==0) mNy=1;
    else if (mNy<-2) mNy=1;
    
    next_line(fp);
    next_line(fp);
    next_line(fp);
    fscanf(fp,"%d",&BC); next_line(fp);     /* Boundary Conditions */
    
    next_line(fp);
    next_line(fp);
    next_line(fp);
    fscanf(fp,"%d",&Af); next_line(fp);     /* Field flag for reading "A.field" file */
    
    next_line(fp);
    next_line(fp);
    next_line(fp);                          /* Data to save to disk */
    fscanf(fp,"%d",&outfilef[0]); next_line(fp);
    fscanf(fp,"%d",&outfilef[1]); next_line(fp);
    fscanf(fp,"%d",&outfilef[2]); next_line(fp);
    fscanf(fp,"%d",&outfilef[3]); next_line(fp);
    fscanf(fp,"%d",&outfilef[4]); next_line(fp);
    fscanf(fp,"%d",&outfilef[5]); next_line(fp);
    fscanf(fp,"%d",&outfilef[6]); next_line(fp);
    fscanf(fp,"%d",&outfilef[7]); next_line(fp);
    fscanf(fp,"%d",&outfilef[8]); next_line(fp);
    fscanf(fp,"%d",&outfilef[9]); next_line(fp);
    fscanf(fp,"%d",&outfilef[10]); next_line(fp);
    fscanf(fp,"%d",&outfilef[11]); next_line(fp);
    fscanf(fp,"%d",&outfilef[12]); next_line(fp);
    fscanf(fp,"%d",&outfilef[13]); next_line(fp);
    outfilef[14]=1;  // always save the time data
    fclose(fp);
  
    /* Tell the user what data is to be used. */
    printf("**********************************************************\n\n");
    printf("Model parameters: \n");
    printf("                 tfinal (msec): %5.2f\n",tfinal);
    printf("                     dt (msec): %1.4f\n",dt);
    printf("                        Nsteps: %10.2f\n",Nsteps);
    printf("                       Lx (cm): %4.2f\n",Lx);
    printf("                       dx (cm): %3.4f\n",dx);
    printf("                            Nx: %d\n",Nx);
    printf("                       Ly (cm): %4.2f\n",Ly);
    printf("                       dy (cm): %3.4f\n",dy);
    printf("                            Ny: %d\n",Ny);
    printf("        dt/[dx*dy] (msec/cm^2): %1.4f\n",dt/(dx*dy));
    printf("\n");
    printf("                msec per write: %4.3f\n",wN*dt);
    printf("              Nsteps per write: %d\n",wN);
    printf("Spatial rate for write (nodes): %d\n",wdN);
    printf("              datafile dx (cm): %1.4f\n",nodeWx);
    printf("              datafile dy (cm): %1.4f\n",nodeWy);
    printf("         datafile Nx (columns): %d\n",(unsigned int)ceil((double)Nx/wdN));
    printf("            datafile Ny (rows): %d\n",(unsigned int)ceil((double)Ny/wdN));
    printf("\n");
    
    printf("        Nsteps per user update: %d\n",rpN);
    printf("              msec user update: %4.3f\n",rpN*dt);
    printf("   Monitor this node [mNx,mNy]: %d, %d\n",mNx,mNy);
    printf("\n");

    printf("\nBR Constants: \n");
    printf("    gK1 (mmho/cm^2): %3.3f\n",constarr[0]);
    printf("    gNa (mmho/cm^2): %3.3f\n",constarr[1]);
    printf("           ENa (mV): %3.3f\n",constarr[2]);
    printf("    gx1 (mmho/cm^2): %3.3f\n",constarr[3]);
    printf("     gs (mmho/cm^2): %3.3f\n",constarr[4]); 
    printf("       Cm (uF/cm^2): %3.3f\n",constarr[5]);
    printf("      kCa (msec^-1): %3.3f\n",constarr[6]);
    printf("   gNaC (mmho/cm^2): %3.3f\n",constarr[7]);
    printf("  Dpara (cm^2/msec): %3.6f\n",constarr[8]);
    printf("Dperpen (cm^2/msec): %3.6f\n",constarr[9]);
    printf("    theta (degrees): %3.3f\n",constarr[10]);
    printf("   sigma (unitless): %3.3f\n",constarr[11]);
    printf("       A (unitless): %3.3f\n",constarr[12]);
    printf("\n");
    
    /* See courtemanche for Euler mesh ratio requirement */
    printf("\nMesh Ratios: \n");
    printf("(dx*dy)/dt [%1.5f (cm^2/msec)] should be greater than 4*Dpara [%1.5f (cm^2/msec)].\n",(dx*dy)/dt,4*constarr[8]);
    printf("(dx*dy)/dt [%1.5f (cm^2/msec)] should be greater than 4*Dperpen [%1.5f (cm^2/msec)].\n",(dx*dy)/dt,4*constarr[9]);
    printf("\n");
    
    printf("\nDiffusion Tensor: \n");
    printf(" D11 (cm^2/msec): %3.6f\n",D[0][0]);
    printf(" D12 (cm^2/msec): %3.6f\n",D[0][1]);
    printf(" D21 (cm^2/msec): %3.6f\n",D[1][0]);
    printf(" D22 (cm^2/msec): %3.6f\n",D[1][1]);
    printf("\n");
    
    printf("\nLaplacian Multipliers: \n");
    printf(" Dp11 (msec^-1): %3.4f\n",Dp[0][0]);
    printf(" Dp12 (msec^-1): %3.4f\n",Dp[0][1]);
    printf(" Dp21 (msec^-1): %3.4f\n",Dp[1][0]);
    printf(" Dp22 (msec^-1): %3.4f\n",Dp[1][1]);
    printf("\n");
    
    printf(" Af=%d\n",Af);
    /* Allocate memory for Afield. */
    Afield=(double **)malloc((unsigned int)(Nx*sizeof(double*)));
    for (i=0;i<Nx;++i) Afield[i]=(double *)malloc((unsigned int)(Ny*sizeof(double)));
    if (Af) {
      readAfield();}
    else {
      printf(" Initializing the A field as a homogenous distribution, A=%3.3f \n",constarr[12]);
      /* Establish homogenous A field */
      for (i=0;i<Nx;++i){ 
        for (j=0;j<Ny;++j){ 
          Afield[i][j]=constarr[12];
        }
      }
    }
    if (ictype==1) {
      printf("\nInitial conditions set for spiral wave.\n");
    }
    if (blocktimenum>0){
      printf("\nBlock information: \n");
      printf("       icewidth (cm): %3.4f \n",icewidth);       // Spiral ics and block stuff are still integretated at this point: 3/28/03
      printf("       icbwidth (cm): %3.4f \n",icbwidth);       // Spiral ics and block stuff are still integretated at this point: 3/28/03
      printf("    iclengthact (cm): %3.4f \n",iclengthact);   // Spiral ics and block stuff are still integretated at this point: 3/28/03
      printf("  iclengthblock (cm): %3.4f \n",iclengthblock); // Spiral ics and block stuff are still integretated at this point: 3/28/03
      printf("          icewidthn : %d \n",icewidthn+1);
      printf("          icbwidthn : %d \n",icbwidthn+1);
      printf("       iclengthactn : %d \n",iclengthactn);
      printf("     iclengthblockn : %d \n",iclengthblockn);
      printf("       blocktimenum : %d \n",blocktimenum);
      printf("   blocktimes (msec): ");
      for (i=0;i<blocktimenum;++i) printf("%4.2f ",blocktimes[i][0]);
      printf("\n\n");
    }
    
    if (ictype!=2) {
      printf("\nInitial Conditions: \n"); 
      printf("        Vm (mV): %3.3f\n",datarr[0][0][middlex][middley]);
      printf("  IK1 (uA/cm^2): %3.3f\n",datarr[0][2][middlex][middley]);
      printf("  Ix1 (uA/cm^2): %3.3f\n",datarr[0][3][middlex][middley]);
      printf("  x1 (unitless): %3.3f\n",datarr[0][4][middlex][middley]);
      printf("  INa (uA/cm^2): %3.3f\n",datarr[0][5][middlex][middley]);
      printf("   m (unitless): %3.3f\n",datarr[0][6][middlex][middley]);
      printf("   h (unitless): %3.3f\n",datarr[0][7][middlex][middley]);
      printf("   Is (uA/cm^2): %3.3f\n",datarr[0][8][middlex][middley]);
      printf("   d (unitless): %3.3f\n",datarr[0][9][middlex][middley]);
      printf("   f (unitless): %3.3f\n",datarr[0][10][middlex][middley]);
      printf("   Cai (mole/L): %3.3e\n",datarr[0][11][middlex][middley]);  
      printf("\n");}
    
    if (stimnum>0) {
      printf("\nStimuli: \n");
      printf("           stimnum: %d\n",stimnum);
      printf("Istimamp (uA/cm^2): %4.3f\n",Istimamp);   // (uA/cm^2)
      printf("    stimint (msec): %2.3f\n",stimint);
      printf("    stimsize1 (cm): %2.3f\n",stimsize1);
      printf("    stimsize2 (cm): %2.3f\n",stimsize2);
      printf("      times (msec): ");
      for (i=0;i<stimnum;++i) printf("%4.2f ",stimes[i]);
      printf("\n\n");}
    if (BC==1) printf("End conditions: Slab\n\n");
    else printf("End conditions: Cylinder\n\n");
    
    printf("\nVariables to save to disk: \n");
    if (outfilef[0]==1) printf("    Vm\n");
    if (outfilef[1]==1) printf("    dVmdt\n");
    if (outfilef[2]==1) printf("    IK1\n");
    if (outfilef[3]==1) printf("    Ix1\n");
    if (outfilef[4]==1) printf("    x1\n");
    if (outfilef[5]==1) printf("    INa\n");
    if (outfilef[6]==1) printf("    m\n");
    if (outfilef[7]==1) printf("    h\n");
    if (outfilef[8]==1) printf("    Is\n");
    if (outfilef[9]==1) printf("    d\n");
    if (outfilef[10]==1) printf("    f\n");
    if (outfilef[11]==1) printf("    Cai\n");
    if (outfilef[12]==1) printf("    Isum\n");
    if (outfilef[13]==1) printf("    Diff\n"); 
    printf("**********************************************************\n");
  }
}

void next_line(FILE *filep)
/* Skips to next input line. */
{
  int dummy;
  while( (dummy=getc(filep)) != '\n') ;
}

void openfiles(void)
{
  
  // mkdir command to create data directory?

  if (outfilef[0]==1) outfile[0]=fopen("data/Vm.dat","wb");       // all of these are binary files
  if (outfilef[1]==1) outfile[1]=fopen("data/dVmdt.dat","wb");
  if (outfilef[2]==1) outfile[2]=fopen("data/IK1.dat","wb");
  if (outfilef[3]==1) outfile[3]=fopen("data/Ix1.dat","wb");
  if (outfilef[4]==1) outfile[4]=fopen("data/x1.dat","wb");
  if (outfilef[5]==1) outfile[5]=fopen("data/INa.dat","wb");
  if (outfilef[6]==1) outfile[6]=fopen("data/m.dat","wb");
  if (outfilef[7]==1) outfile[7]=fopen("data/h.dat","wb");
  if (outfilef[8]==1) outfile[8]=fopen("data/Is.dat","wb");
  if (outfilef[9]==1) outfile[9]=fopen("data/d.dat","wb");
  if (outfilef[10]==1) outfile[10]=fopen("data/f.dat","wb");
  if (outfilef[11]==1) outfile[11]=fopen("data/Cai.dat","wb");
  if (outfilef[12]==1) outfile[12]=fopen("data/Isum.dat","wb");
  if (outfilef[13]==1) outfile[13]=fopen("data/Diff.dat","wb");
  outfile[14]=fopen("data/time.dat","wb");    // last datafile is always the time file
  fcfile=fopen("data/br2dfc.dat","wb");       // final conditions file
  stiminfofile=fopen("stim_info.txt","w");   // textfile with real-time stimulation info

}

void closefiles(void)
{
  int i;
  FILE *infofile;

  for(i=0; i<outfilenum; i++) {
    if (outfilef[i]==1) fclose(outfile[i]);}
  fclose(fcfile);
  fclose(stiminfofile);
  
  // Update info file
  infofile=fopen("data/brdr2dinfo.txt","w");
  fprintf(infofile,"%d\n",(unsigned int)ceil((double)Nx/wdN));  // Number of columns in datafiles
  fprintf(infofile,"%d\n",(unsigned int)ceil((double)Ny/wdN));  // Number of rows in datafiles
  fprintf(infofile,"%d",Nstepssaved);      // Number of snapshots saved
  fclose(infofile);              
}  

void output(void)
/* Save to binary files */
/* Spatial sampling rate of wdN */
// datarr: (rows)X(columns)X(depth)X(4dim)=(Nx)X(Ny)X(varnum)X(datarr4dim)

{
  int i,j,k;
  
  Nstepssaved=Nstepssaved+1;
  fwrite(&derivarr[0], sizeof(double),1,outfile[outfilenum-1]);  // save the time vector
  
  for (i=1;i<Nx+1;i=i+wdN){
    for (j=1;j<Ny+1;j=j+wdN){
      for (k=0;k<outfilenum-1;++k) {                                                   // save the data
        if (k==13 && outfilef[13]==1) fwrite(&datarr[(step-1)%2][k][i][j], sizeof(double),1,outfile[k]);  // save diffusion
        else if (outfilef[k]==1) fwrite(&datarr[step%2][k][i][j], sizeof(double),1,outfile[k]);            // save others
      }
    }
  }
  // Update info file
  FILE *infofile;
  infofile=fopen("data/brdr2dinfo.txt","w");
  fprintf(infofile,"%d\n",(unsigned int)ceil((double)Nx/wdN));  // Number of columns in datafiles
  fprintf(infofile,"%d\n",(unsigned int)ceil((double)Ny/wdN));  // Number of rows in datafiles
  fprintf(infofile,"%d",Nstepssaved);      // Number of snapshots saved
  fclose(infofile);  
}

void brfc(void)
/* Save final conditions to a binary file.
   Save data at EVERY node to accurately re-establish
   continuation simulations. */
// datarr: (rows)X(columns)X(depth)X(4dim)=(Nx)X(Ny)X(varnum)X(datarr4dim)
{
  int i,j,k;
  
  // First three elements are Nx, Ny, and final time
  
  fwrite(&Nx,sizeof(unsigned int),1,fcfile);     // Nx
  fwrite(&Ny,sizeof(unsigned int),1,fcfile);     // Ny
  fwrite(&derivarr[0],sizeof(double),1,fcfile);  // final time
  
  for (k=0;k<varnum;++k) { 
    if (k==13) {                     // Save the next diffusion matrix: [(step-1)%2]
      for (i=1;i<Nx+1;++i){
        for (j=1;j<Ny+1;++j) fwrite(&datarr[(step-1)%2][k][i][j],sizeof(double),1,fcfile);  // save the data
      }
    }
    else {                        // Save current data for all other variables: [(step)%2]
      for (i=1;i<Nx+1;++i){
        for (j=1;j<Ny+1;++j) fwrite(&datarr[step%2][k][i][j],sizeof(double),1,fcfile);  // save the data
      }
    }  
  }
}


void readbrfc(void)
/* Read final conditions from a binary file. */
// datarr: (rows)X(columns)X(depth)X(4dim)=(Nx)X(Ny)X(varnum)X(datarr4dim)
{
  int i,j,k;
  double dblhold, readTfinal;
  int inthold, readNx, readNy;
  FILE *fcfid;
  fcfid=fopen(fcfilename,"rb");
  
  printf("Reading initial conditions from %s ... \n",fcfilename);
  
  // First three elements are Nx, Ny, and final time 
  fread(&inthold, sizeof(int), 1, fcfid);
  readNx=inthold;
  fread(&inthold, sizeof(int), 1, fcfid);
  readNy=inthold;
  fread(&dblhold, sizeof(double), 1, fcfid);
  readTfinal=dblhold;
  
  if (Nx!=readNx){
    printf("Nx (%d) in brdr2dtask.dat does not match Nx (%d) of %s!\n",Nx,readNx,fcfilename);
    exit(1);}
  if (Ny!=readNy){
    printf("Ny (%d) in brdr2dtask.dat does not match Ny (%d) of %s!\n",Ny,readNy,fcfilename);
    exit(1);}
  printf("Simulation time of conditions in %s is %4.3f\n",fcfilename,readTfinal);
  
  for (k=0;k<varnum;++k) { 
    for (i=1;i<Nx+1;++i){
      for (j=1;j<Ny+1;++j){
        fread(&dblhold, sizeof(double), 1, fcfid);
        datarr[0][k][i][j]=dblhold;
      } 
    }
  }  
  fclose(fcfid);
}


void readAfield(void)
/* Read field data for variable A from file [fname defined in openfiles()] */
{
  int i, j, inthold, readNx, readNy;
  double dblhold, Amean, Astd;

  printf(" Reading field data for variable A from file ... \n");

  if ((Afieldfile=fopen("field/A.field","rb"))==NULL) {
    printf(" Unable to open field/A.field\n");
    exit(1);
  }

  fread(&inthold, sizeof(int), 1, Afieldfile);
  readNx=inthold;
  fread(&inthold, sizeof(int), 1, Afieldfile);
  readNy=inthold;
  if (Nx!=readNx){
    printf(" Nx (%d) in A.field does not match Nx (%d) of brdr2dtask.dat!\n",readNx,Nx);
    exit(1);}
  if (Ny!=readNy){
    printf(" Ny (%d) in A.field does not match Ny (%d) of brdr2dtask.dat!\n",readNy,Ny);
    exit(1);}

  for (i=0;i<Nx;++i){
      for (j=0;j<Ny;++j){
        if (feof(Afieldfile)) {
	  printf(" Prematurely reached EOF!\n");
	  exit(1);}
        fread(&dblhold, sizeof(double), 1, Afieldfile);
        Afield[i][j]=dblhold;
      } 
  }
  fread(&dblhold, sizeof(double), 1, Afieldfile);
  if (!feof(Afieldfile)) {
    printf(" Did not read to EOF!\n");
    exit(1);}
    
  printf(" Read %dx%d values of A\n",readNx,readNy);
  fclose(Afieldfile);
  
  Amean=0;
  for (i=0;i<Nx;++i){
    for (j=0;j<Ny;++j){
      Amean=Amean+Afield[i][j];
    }
  }
  Amean=Amean/(Nx*Ny);
  
  Astd=0;
  for (i=0;i<Nx;++i){
    for (j=0;j<Ny;++j){
      Astd=Astd+(Afield[i][j]-Amean)*(Afield[i][j]-Amean);
    }
  }
  Astd=sqrt(Astd/(Nx*Ny-1));
  
  printf(" Values of A have mean of %4.3f\n",Amean);
  printf("               and std of %4.3f\n",Astd);

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

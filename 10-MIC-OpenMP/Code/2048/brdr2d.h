/* MWKay, 3/28/2003 */
/* MWKay, 9/07/2005 */

#define NX 2048 
#define NY 2048 
#define MAXSTIM 1
#define MAXBLOCKTIMES 100

/* Model Constants  */
  int constnum=13;      // number of rows in constarr 9/07/2005
//  double *constarr;     // constnum x 1 array
  double constarr[13];     // constnum x 1 array
  
/* Diffusion Tensors */  
  double D[2][2];  // Diffusion tensor
  double Dp[2][2]; // Dp for Dprime: Laplacian multiplier form of D
   
/* Model Gates/derivates */
  int derivnum=7;               // num of rows in derivarr
  double derivarr[7];     	// derivnum x 1 array

/* Data Matrix */
// datarr: (rows)X(columns)X(depth)X(4dim)=(Nx)X(Ny)X(varnum)X(datarr4dim)
  int varnum=15;      // varnum is 'depth' of datarr, Nx and Ny defined in task.dat
  int datarr4dim=2;   // 4th dim of datarr: (step-1)%2 for previous timestep and step%2 for current timestep 
//  double ****datarr;   // note: % is modulus operator (division remainder)
  double datarr[2][15][NX+2][NY+2];
    
/* Time Stuff */
  double Nsteps;       // total # of time steps
  double dt, tfinal;   // (msec)
  int step;            // time step counter 
  double msecW=1;      // msec per write to datafile
  int wN=100;          // steps per write to datafile
  double msecRP=1;     // msec per user update (ie, print current time to screen)
  int rpN=1;         // steps per user update (ie, print current time to screen)
  int mNx=0;           // x index of node to monitor 
  int mNy=0;           // y index of node to monitor
  int stable=1;        // flag for stability check
  
/* Space Stuff */
  double Lx;           // width (cm)
  double dx;           // x grid spacing (cm)
  int Nx;              // number of space nodes in x dir
  int Xstep;           // x dir step counter    
  double Ly;           // height (cm)
  double dy;           // y grid spacing (cm)
  int Ny;              // number of space nodes in y dir
  int Ystep;           // y dir step counter  
  double nodeWx;       // dx per write to datafile (cm)
  double nodeWy;       // dy per write to datafile (cm)
  int wdN;             // spatial sampling for saving data (nodes)
  
/* next_line pointer */
  void next_line(FILE[]); 

/* Stimulus stuff */
  int stimnum;
  double stimint, Istimamp, stimsize1, stimsize2;
  //double ****stimarr;         // Stimulus matrix [Nx]x[Ny]x[stimnum]x2, [1:Nx],[1:Ny],[1:stimnum],[0:1] 
  double stimarr[NX][NY][MAXSTIM][2];
  double *stimes;             // Stimulus times from br2dtask.dat
  
/* Output Files */
  int outfilenum=15;    // this is varnum (don't save the stimulus but do save the time)
  int Nstepssaved=0;    // save to datafiles counter
  FILE *outfile[15];    // this is varnum (don't save the stimulus but do save the time)
  FILE *fcfile;         // final conditions
  FILE *stiminfofile;   // textfile with real-time stimulation info
  int outfilef[15];     // outfile flag for save or not save. Defined in brdr2dtask.dat.

/* Final Conditions */
  char *fcfilename;
  
/* Boundary Conditions */
  int BC;               // 1 for slab, 2 for cylinder
  
/* Block stuff */
  int block[NX+2][NY+2];     // block matrix is multiplied to gNa AND gs.
  int ictype;      // type of initial conditions
  double icewidth, icbwidth, iclengthact, iclengthblock;   // ics for spiral initialization
  int blocktimenum;          // number of block times
  //double **blocktimes;       // Deactivate/activate block at these times (msec)
  double blocktimes[MAXBLOCKTIMES][2];       // Deactivate/activate block at these times (msec)
  
/* Field data */
  int Af;    // flag for reading field data for variable A (IK1 multiplier)
  double Afield[NX][NY];
  FILE *Afieldfile;

double abfun_0(double vv);
double abfun_1(double vv);
double abfun_2(double vv);
double abfun_3(double vv);
double abfun_4(double vv);
double abfun_5(double vv);
double abfun_6(double vv);
double abfun_7(double vv);
double abfun_8(double vv);
double abfun_9(double vv);
double abfun_10(double vv);
double abfun_11(double vv);

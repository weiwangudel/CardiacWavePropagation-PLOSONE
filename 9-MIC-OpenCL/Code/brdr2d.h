/* MWKay, 3/28/2003 */
/* MWKay, 9/07/2005 */
#include <CL/cl.h>
int THREAD_DIMX;
int THREAD_DIMY;

int BLOCK_DIMX;
int BLOCK_DIMY;

/* Model Constants  */
  int constnum=13;      // number of rows in constarr 9/07/2005
  double *constarr;     // constnum x 1 array
  
  double C[12][7]={ /* Rate constants for conduction parameters */
  0.0005,  0.083, 50,      0,    0,     0.057, 1,
  0.0013, -0.06,  20,      0,    0,    -0.04,  1, 
  0,       0,     42.65,  -0.9, 42.65, -0.22, -1,
  1.437,  -0.085, 39.75,   0,    0,     0,     0,
  0.1,    -0.193, 79.65,   0,    0,     0,     0,
  1.7,     0,     20.5,    0,    0,    -0.095, 1,
  0.055,  -0.25,  78,      0,    0,    -0.2,   1,   /* rows 7 and 8 are constants for j gate, */
  0.3,     0,     32,      0,    0,    -0.1,   1,   /* which is not used in BRDR. */
  0.095,  -0.01,  -5,      0,    0,    -0.072, 1,
  0.07,   -0.017, 44,      0,    0,     0.05,  1,
  0.012,  -0.008, 28,      0,    0,     0.15,  1,
  0.0065, -0.02,  30,      0,    0,    -0.2,   1};
  /* rows 7 & 8 (j gate) are used in BR but not used in BRDR */
  /* C[:][1]  msec^-1  */
  /* C[:][2]  mV^-1  */
  /* C[:][3]  mV  */
  /* C[:][4]  (mV*msec)^-1  */
  /* C[:][5]  mV  */
  /* C[:][6]  mV^-1  */
  /* C[:][7]  dimensionless  */
  
/* Diffusion Tensors */  
  double D[2][2];  // Diffusion tensor
  double Dp[2][2]; // Dp for Dprime: Laplacian multiplier form of D
   
/* Model Gates/derivates */
  int derivnum=7;               // num of rows in derivarr
  double *derivarr;     	// derivnum x 1 array	
  double ***deriv3darr;

/* Data Matrix */
// datarr: (rows)X(columns)X(depth)X(4dim)=(Nx)X(Ny)X(varnum)X(datarr4dim)
  int varnum=15;      // varnum is 'depth' of datarr, Nx and Ny defined in task.dat
  int datarr4dim=2;   // 4th dim of datarr: (step-1)%2 for previous timestep and step%2 for current timestep 
  double ****datarr;   // note: % is modulus operator (division remainder)
  double ****datarr_gpu; /* for comparision */
  
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
  double ****stimarr;         // Stimulus matrix [Nx]x[Ny]x[stimnum]x2, [1:Nx],[1:Ny],[1:stimnum],[0:1]
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
  int **block;     // block matrix is multiplied to gNa AND gs.
  int ictype;      // type of initial conditions
  double icewidth, icbwidth, iclengthact, iclengthblock;   // ics for spiral initialization
  int blocktimenum;          // number of block times
  double **blocktimes;       // Deactivate/activate block at these times (msec)
  
/* Field data */
  int Af;    // flag for reading field data for variable A (IK1 multiplier)
  double **Afield;
  FILE *Afieldfile;

/*OpenCL parameter */
FILE *fp;
char *source_str;
char str_temp[1024];
size_t source_size;

cl_platform_id *platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_kernel clKernel_stimulate;
cl_kernel clKernel_blockonoff;
cl_kernel clKernel_brgates;
cl_kernel clKernel_brcurrents;
cl_kernel clKernel_bcs;
cl_kernel clKernel_update;

/* GPU data */
cl_mem d_datarr;
cl_mem d_stimarr;
cl_mem d_derivarr;
cl_mem d_blocktimes;
cl_mem d_block;
cl_mem d_constarr;
cl_mem d_Afield;
cl_mem d_Dp;
cl_mem d_D;
#define MAX_SOURCE_SIZE (0x100000)

double* linear_datarr;
double* linear_stimarr;
double* linear_deriv3darr; 
/*double* d_datarr;
double* d_stimarr;
double* d_derivarr;
double* d_blocktimes;
int*    d_block;
double* d_constarr;
double* d_Afield;
double* d_Dp;
double* d_D;
double* linear_datarr;
double* linear_stimarr;
double* linear_deriv3darr; */ 
/*********************************************/
// datarr: (rows)X(columns)X(depth)X(4dim)=(Nx)X(Ny)X(varnum)X(datarr4dim)
// 4th dim is a flip-flop b/w current and previous states
//
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
// datarr[][][13][] is Diff    (mV/msec) Diffusion: Laplacian and D
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
// constarr[0] is gK1     (mmho/cm^2)
// constarr[1] is gNa     (mmho/cm^2)
// constarr[2] is ENa     (mV)  
// constarr[3] is gx1     (mmho/cm^2)
// constarr[4] is gs      (mmho/cm^2)
// constarr[5] is Cm      (uF/cm^2)
// constarr[6] is kCa     (msec^-1)
// constarr[7] is gNaC    (mmho/cm^2)
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


#include <mex.h>
#include <stdlib.h> 
#include "fenchel_dual_svm.h"

typedef unsigned char uint8;
typedef unsigned int uint32;

// w=mexFunction(X,y,qids,r,T,err,verbose)
// prhs[0] X        double
// prhs[1] y        int
// prhs[2] qids     int
// prhs[3] r        double
// prhs[4] T        int
// prhs[5] err      double
// prhs[6] verbose  double
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  double r;
  uint32 T;
  double err;
  double *X;
  uint32 *y;
  uint32 verbose = 1;

  int n_features = mxGetN( prhs[0] );   
  int n_samples = mxGetM( prhs[0] );

  uint32 *qids;
  X = (double*) mxGetData(prhs[0]);
  y = (uint32*) mxGetData(prhs[1]);
  qids = (uint32*) mxGetData(prhs[2]);
  r = (double) mxGetScalar(prhs[3]);
  T = (uint32) mxGetScalar(prhs[4]);
  err = (double) mxGetScalar(prhs[5]);
  verbose = (uint32) mxGetScalar(prhs[6]);

  if(verbose > 0) {
    mexPrintf("m %d\t", n_samples);
    mexPrintf("d %d\t", n_features);
    mexPrintf("r %f\t", r);
    mexPrintf("err %f\t", err);
    mexPrintf("T %d\n", T);
  }

  int nonzero;
  int m = n_samples;
  int d = n_features;

  int* rIndex;
  int* cIndex;
  double* value;
  int *label;//label
  int *queryId;//query id

  nonzero = m*d;
  value = new double[nonzero];
  cIndex = new int[nonzero];
  rIndex = new int[m+1];
  queryId = new int[m];
  label = new int[m];

  int j = 0;
  for(int i=0;i<m;i++) {
      rIndex[i]=j;
      queryId[i] = qids[i];
      label[i] = y[i];
      for(int k=0;k<d;k++) {
          cIndex[j] = k;
          value[j] = X[i*d+k];
          ++j;
      }  
  }
  rIndex[m]=nonzero;

  pairwise_fenchel_dual_svm sparse_svm(value, label, queryId, rIndex, cIndex, err, r, T, m, d);
  if(verbose > 0) {
    mexPrintf("training....\n");
  }
  sparse_svm.learn();

  plhs[0] = mxCreateNumericMatrix(d, 1, mxDOUBLE_CLASS, mxREAL);  
  double *p2 = mxGetPr(plhs[0]);
  double *weight = sparse_svm.get_w();
  for( int i=0; i<d; i++ ){
      p2[i] = weight[i];
      // p2[i] = i;
  }

  // mexErrMsgTxt("Mismatch between data types.");
}

#define DLLEXPORT extern "C"
#include <stdio.h>	/* printf */
#include <math.h>	/* log */
#include "mkl.h"

/* XX' is calculated frequently
   for a IxN matrix X.
   I is the number of SNPs.
   N is the number of samples
   Hence a dedicated function  */

void axxT (int I, int N, double alpha, double* A, double* C) {

    double zero = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, I, I, N, alpha, A, N, A, N, zero, C, I);

}

/* Calculate X'BX given X and B.
   X is a I x N matrix.
   B is a I x I matrix.
   result is N x N
*/

DLLEXPORT int xTBx(int I, int N, double alpha, double* X, double* B, double* C) {

    double zero = 0.0;
    double one = 1.0;
    double *D;

    D = (double *)mkl_malloc( I*N*sizeof( double ), 64 );
    if (D == NULL) {
        printf( "C++ Error: Can't allocate memory for matrices. Aborting... \n");
        mkl_free(D);
        exit(0);
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, N, I, one,   B, I, X, N, zero, D, N);
    cblas_dgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, N, N, I, alpha, X, N, D, N, zero, C, N);

    mkl_free(D);
    return 0;
}


DLLEXPORT double matloginv (int n, double* A) {

    int i;
    int info;
    double ldet;

    // Cholesky factorization of a symmetric (Hermitian) positive-definite matrix
    dpotrf("L", &n, A, &n, &info); 
    if (info != 0) {
        printf ("C++ Error: Cholesky factorization failed. Aborting... \n");
        exit(0);
    }


    // get the log determinant
    ldet = 0.0;
    for (i = 0; i < n; i++) {
        ldet += 2 * log(A[i * n + i]);
    }
    //*log_det = ldet;

    // inverse of a symmetric (Hermitian) positive-definite matrix using the Cholesky factorization.
    dpotri("L", &n, A, &n, &info);
    if(info != 0) {
        printf ("C++ Error: Matrix inversion failed. Aborting... \n");
        exit(0);
    }

    return ldet;
}

#include <stdio.h>	/* printf */
#include <math.h>	/* log */
#include "mkl.h"

// Calculate v'Bv where B is a NxN symmetric matrix
// and v is a vector of length N
//
double vTBSYMv( int n, double* v, double* B) {

    double one = 1.0;
    double zero = 0.0;
    double* D;
    double res;

    D = (double *)mkl_malloc( n*sizeof( double ), 64 );
    if (D == NULL) {
        printf( "C++ Error: Can't allocate memory for matrices. Aborting... \n");
        mkl_free(D);
        exit(0);
    }

    cblas_dsymv(CblasRowMajor, CblasLower, n, one, B, n, v, 1, zero, D, 1);
    res = cblas_ddot(n, v, 1, D, 1);

    mkl_free(D);
    return res;
}

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
int xTBx(int I, int N, double alpha, double* X, double* B, double* C) {

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


double matloginv (int n, double* A) {

    int i, j;
    int info;
    double ldet;

    /* Cholesky factorization of a symmetric (Hermitian) positive-definite matrix */
    info = LAPACKE_dpotrf (LAPACK_ROW_MAJOR, 'L', n , A, n);
    if (info != 0) {
        printf ("C++ Error: Cholesky factorization failed. Aborting... \n");
        exit(0);
    }


    ldet = 0.0;
    for (i = 0; i < n; i++) {
        ldet += 2 * log(A[i * n + i]);
    }

 /* inverse of a symmetric (Hermitian) positive-definite matrix using the Cholesky factorization.
 *
 */
    info = LAPACKE_dpotri (LAPACK_ROW_MAJOR, 'L', n, A, n);
    if(info != 0) {
        printf ("C++ Error: Matrix inversion failed. Aborting... \n");
        exit(0);
    } else {
    /* overwrite the upper half */
        for (i = 0; i < n; i++) {
            for (j = i+1; j < n; j++) {
                A[ i*n+j ] = A[ j*n+i];
            }
        }
    }

    return ldet;
}

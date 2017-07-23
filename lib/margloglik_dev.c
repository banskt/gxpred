#include <stdbool.h>    /* datatype bool */
#include <stdio.h>      /* printf */
#include <math.h>       /* log */
#include "mkl.h"

#define _PI 3.141592653589793

double vT_BSYM_v( int n, double* v, double* B) {
    // Calculate v'Bv where B is a NxN symmetric matrix
    // and v is a vector of length N

    double one = 1.0;
    double zero = 0.0;
    double* D;
    double res;

    D = (double *)mkl_malloc( n*sizeof( double ), 64 );
    if (D == NULL) {
        printf( "C++ Error: Can't allocate memory for matrix in function vT_BSYM_v. Aborting... \n");
        mkl_free(D);
        exit(0);
    }

    cblas_dsymv(CblasRowMajor, CblasLower, n, one, B, n, v, 1, zero, D, 1);
    res = cblas_ddot(n, v, 1, D, 1);

    mkl_free(D);
    return res;
}

/*  XX' is calculated frequently
 *     for a IxN matrix X.
 *        I is the number of SNPs.
 *           N is the number of samples
 *              Hence a dedicated function  */
void axxT (int I, int N, double alpha, double* A, double* C) {

    double zero = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, I, I, N, alpha, A, N, A, N, zero, C, I);

}

/*  Calculate X'BX given X and B.
 *     X is a I x N matrix.
 *        B is a I x I matrix.
 *           result is N x N
 *           */
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


double matlogdetinv (int n, double* A) {

    int i, j;
    int info;
    double ldet;

    /*  Cholesky factorization of a symmetric (Hermitian) positive-definite matrix */
    info = LAPACKE_dpotrf (LAPACK_ROW_MAJOR, 'L', n , A, n);
    if (info != 0) {
        printf ("C++ Error: Cholesky factorization failed. Aborting... \n");
        exit(0);
    }


    ldet = 0.0;
    for (i = 0; i < n; i++) {
        ldet += 2 * log(A[i * n + i]);
    }

 /*  inverse of a symmetric (Hermitian) positive-definite matrix using the Cholesky factorization.
 *   *
 *    */ 
    info = LAPACKE_dpotri (LAPACK_ROW_MAJOR, 'L', n, A, n);
    if(info != 0) {
        printf ("C++ Error: Matrix inversion failed. Aborting... \n");
        exit(0);
    } else {
    /*  overwrite the upper half */
        for (i = 0; i < n; i++) {
            for (j = i+1; j < n; j++) {
                A[ i*n+j ] = A[ j*n+i];
            }
        }
    }

    return ldet;
}

// Update B_INV
// Keep a copy of A
// B[i,j] = Acopy[i, j] - mod * Acopy[i, zpos] * Acopy[zpos, i]
// A = B
// delete Acopy
void binv_update(int nsnps, int zpos, double mod, double* A) {

    int i, j;
    int ij, ik, kj;
    double* D;

    D = (double *)mkl_malloc( nsnps * nsnps * sizeof( double ), 64 );
    if (D == NULL) {
        printf( "C++ Error: Can't allocate memory for matrix in function binv_update. Aborting... \n");
        mkl_free(D);
        exit(0);
    }

    cblas_dcopy ( nsnps*nsnps, A, 1, D, 1 );
    for (i = 0; i < nsnps; i++) {
        for (j = 0; j < nsnps; j++) {
            ij = i * nsnps + j;
            ik = i * nsnps + zpos;
            kj = zpos * nsnps + j;
            A[ij] = D[ij] - mod * D[ik] * D[kj];
        }
    }

    mkl_free(D);
}


void zcomps(int     nsnps,
            int     nsample,
            int     zlen,
            double  pi,
            double  mu,
            double  sigma,
            double  sigmabg,
            double  tau,
            int*    ZARR,
            int*    ZNORM,
            double* GT,
            double* GX,
            double* ZCOMPS,
            double* BZINV,
            double* SZINV) {

    int     i, j, k;
    int     z, nz, zindx, zpos;

    double  sigma2;
    double  sigmabg2;
    double  hdiff;
    double  logB0det;
    double  logBZdet;
    double  logSZdet;
    double  nterm;
    double  mod_denom, mod;
    double  log_probz;
    double  log_normz;


    double* B_INV;
    double* S_INV;
    double* GX_MZ; // length of N

    B_INV = (double *)mkl_malloc(        nsnps   * nsnps   * sizeof( double ), 64 );
    S_INV = (double *)mkl_malloc(        nsample * nsample * sizeof( double ), 64 );
    GX_MZ = (double *)mkl_malloc(                  nsample * sizeof( double ), 64 );

    if (B_INV == NULL || S_INV == NULL || GX_MZ == NULL) {
        printf( "C++ Error: Can't allocate memory for z-specific Bz / Sz. Aborting... \n");
        mkl_free(B_INV);
        mkl_free(S_INV);
        mkl_free(GX_MZ);
        exit(0);
    }

    sigma2 = sigma * sigma;
    sigmabg2 = sigmabg * sigmabg;
    hdiff = (1 / sigma2) - (1 / sigmabg2);

    /* Initialize the calculation with B_INV and logBdet for zstate = [[]] */

    for (i = 0; i < (nsnps*nsnps); i++) {
        B_INV[i] = 0.0;
    }

    axxT(nsnps, nsample, tau, GT, B_INV);
    for (i = 0; i < nsnps; i++) {
        B_INV[ i*nsnps + i ] += 1 / sigmabg2;
    }
    logB0det = matlogdetinv(nsnps, B_INV);
    for (i = 0; i < (nsnps*nsnps); i++) {
        BZINV[i] = B_INV[i];
    }

    /* Loop over the zstates */
    zindx = 0;
    for (z = 0; z < zlen; z++) {
        nz = ZNORM[z];

        // initialize with B0INV, logB0det and GX
        for (i = 0; i < (nsnps*nsnps); i++) {
            B_INV[i] = BZINV[i];
        }
        for (i = 0; i < nsample; i++) {
            GX_MZ[i] = GX[i];
        }
        logBZdet = logB0det;

        // loop over the causal SNPs, for nz = 0, it skips the loop.
        for (i = 0; i < nz; i++) {
            zpos = ZARR[zindx + i];
            mod_denom = 1 + hdiff * B_INV[ zpos*nsnps + zpos ];
            mod = hdiff / mod_denom;
            logBZdet += log(mod_denom);
            binv_update(nsnps, zpos, mod, B_INV);
            for (j = 0; j < nsample; j++) {
                GX_MZ[j] -= mu * GT[ zpos*nsample + zpos ];
            }
        }

        log_probz = nz * log(pi) + (nsnps - nz) * log(1 - pi);

        for (i = 0; i < (nsnps * nsnps); i++) {
            S_INV[i] = 0.0;
        }
        xTBx( nsnps, nsample, (-tau*tau), GT, B_INV, S_INV );
        for (i = 0; i < nsample; i++) {
            S_INV[ i*nsample + i ] += tau;
        }
        logSZdet = - (nsample * log(tau)) + ((nsnps - nz) * log(sigmabg2)) + (nz * log(sigma2)) + logBZdet;

        nterm = vT_BSYM_v( nsample, GX_MZ, S_INV );
        log_normz = - 0.5 * (logSZdet + (nsample * log(2 * _PI)) + nterm);

        ZCOMPS[z] = log_probz + log_normz;

        for (i = 0; i < (nsnps*nsnps); i++) {
            BZINV[ z*nsnps*nsnps + i ] = B_INV[i];
        }
        for (i = 0; i < (nsample*nsample); i++) {
            SZINV[ z*nsample*nsample + i ] = S_INV[i];
        }

        zindx += nz;

    }

    mkl_free(B_INV);
    mkl_free(S_INV);
    mkl_free(GX_MZ);
}

//void grad (nsnps, nsample, zlen, pi, mu, sigma, sigmabg, tau, ZARR, ZNORM, GT, GX, ZCOMPS, BZINV, SZINV, ZPROB, GRAD);

double margloglik(int     nsnps,
                  int     nsample,
                  int     zlen,
                  double  pi,
                  double  mu,
                  double  sigma,
                  double  sigmabg,
                  double  tau,
                  bool    get_gradient,
                  int*    ZARR,
                  int*    ZNORM,
                  double* GT,
                  double* GX,
                  double* ZCOMPS,
                  double* GRAD) {

    int     i, k;
    double  logk;
    double  zcompsum;
    double  margloglik;

    double* BZINV; //for each zstate BZ is a matrix of size I x I
    double* SZINV; //for each zstate S  is a matrix of size N x N

    printf ("Hello from C \n");

    BZINV = (double *)mkl_malloc( zlen * nsnps   * nsnps   * sizeof( double ), 64 );
    SZINV = (double *)mkl_malloc( zlen * nsample * nsample * sizeof( double ), 64 );

    if (BZINV == NULL || SZINV == NULL) {
        printf( "C++ Error: Can't allocate memory for Bz / Sz. Aborting... \n");
        mkl_free(BZINV);
        mkl_free(SZINV);
        exit(0);
    }

    zcomps(nsnps, nsample, zlen, pi, mu, sigma, sigmabg, tau, ZARR, ZNORM, GT, GX, ZCOMPS, BZINV, SZINV);

    k = cblas_idamax (zlen, ZCOMPS, 1);
    logk = ZCOMPS[k];
    zcompsum = 0.0;
    for (i = 0; i < zlen; i++) {
        zcompsum += exp(ZCOMPS[i] - logk);
    }
    margloglik = log(zcompsum) + logk ;
    

    if (get_gradient) {

    }

    mkl_free(BZINV);
    mkl_free(SZINV);

    return margloglik;
}

int main() {
    return 0;
}

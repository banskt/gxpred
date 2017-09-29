/*
 * =====================================================================================
 *
 *       Filename:  logmarglik.c
 *
 *    Description:  Log marginal likelihood and gradient for the optimization
 *
 *        Version:  1.0
 *        Created:  22/07/17 11:36:53
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Saikat Banerjee (banskt), bnrj.saikat@gmail.com
 *   Organization:  Max Planck Institute for Biophysical Chemistry
 *
 * =====================================================================================
 */


#include <stdbool.h>    /* datatype bool */
#include <stdio.h>      /* printf */ 
#include <math.h>       /* log, exp */
#include "mkl.h"
//#include "profiler.h"

#define _PI 3.141592653589793


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  vecT_smat_vec
 *  Description:  calculate v'Av where v is a vector of length n, 
 *                and A is a n-by-n symmetric matrix
 *                D is a dummy vector of length n
 * =====================================================================================
 */
    double
vecT_smat_vec ( int n, double* v, double* A, double* D )
{
    double one = 1.0;
    double zero = 0.0;
    double res;

    cblas_dsymv(CblasRowMajor, CblasLower, n, one, A, n, v, 1, zero, D, 1);
    res = cblas_ddot(n, v, 1, D, 1);

    return res;
}		/* -----  end of function vecT_smat_vec  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  vecAT_smat_vecB
 *  Description:  calculate v'Aw where v and w are vectors of length n,
 *                A is a n-by-n symmetric matrix
 *                D is a dummy vector of length n
 * =====================================================================================
 */
    double
vecAT_smat_vecB ( int n, double* v, double* A, double* w, double* D )
{
    double res;
    double one = 1.0;
    double zero = 0.0;

    cblas_dsymv(CblasRowMajor, CblasLower, n, one, A, n, w, 1, zero, D, 1);
    res = cblas_ddot(n, v, 1, D, 1);

    return res;

}		/* -----  end of function vecAT_smat_vecB  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  a_mat_matT
 *  Description:  calculate aAA' where a is scalar,
 *                A is a m-by-n matrix
 *                C holds the result and is of size m-by-m
 * =====================================================================================
 */
    void
a_mat_matT ( int m, int n, double alpha, double* A, double* C )
{
    double zero = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, m, n, alpha, A, n, A, n, zero, C, m);

}		/* -----  end of function a_mat_matT  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  a_matT_matb_mat
 *  Description:  calculate aA'BA 
 *                A is a m-by-n matrix
 *                B is a m-by-m matrix
 *                C holds the result and is of size n-by-n
 *                D is a dummy matrix of size m-by-n
 * =====================================================================================
 */
    void
a_matT_matb_mat ( int m, int n, double alpha, double* A, double* B, double* C, double* D )
{
    double zero = 0.0;
    double one = 1.0;
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, m, one,   B, m, A, n, zero, D, n);
    cblas_dgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, n, n, m, alpha, A, n, D, n, zero, C, n);
    
}		/* -----  end of function a_matT_matb_mat  ----- */

    void
a_matT_matb_mat_debug ( int m, int n, double alpha, double* A, double* B, double* C, double* D )
{
    double zero = 0.0;
    double one = 1.0;

    printf("m, n, alpha: %d, %d, %f\n", m, n, alpha);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, m, one,   B, m, A, n, zero, D, n);
    printf("D = Binv * x done.\n");
    //cblas_dgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, n, n, m, alpha, A, n, D, n, zero, C, n);
    //printf("C = x * D done.\n");
 
}               /*  -----  end of function a_matT_matb_mat  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  mat_trace
 *  Description:  calculates the trace of a matrix A of size n-by-n
 * =====================================================================================
 */
    double
mat_trace ( int n, double* A )
{
    int i;
    double trace;

    trace = 0.0; 
    for ( i = 0; i < n; i++ ) {
        trace += A[ i*n+i ];
    }
    return trace;

}		/* -----  end of function mat_trace  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  logdet_inverse_of_mat
 *  Description:  calculate the logdet and inverse 
 *                of a symmetric (Hermitian) positive-definite matrix.
 *                Inverse is calculated in place.
 *                logdet is returned as result.
 * =====================================================================================
 */
    double
logdet_inverse_of_mat ( int n, double* A )
{
    int i, j;
    int info;
    double logdet;

//  Cholesky factorization of a symmetric (Hermitian) positive-definite matrix
    info = LAPACKE_dpotrf (LAPACK_ROW_MAJOR, 'L', n , A, n);
    if ( info != 0 ) {
        printf ("C++ Error: Cholesky factorization failed. Aborting... \n");
        exit(0);
    }

    logdet = 0.0;
    for (i = 0; i < n; i++) {
        logdet += 2 * log(A[ i*n+i ]);
    }

//  inverse of a symmetric (Hermitian) positive-definite matrix using the Cholesky factorization.
    info = LAPACKE_dpotri (LAPACK_ROW_MAJOR, 'L', n, A, n);
    if(info != 0) {
        printf ("C++ Error: Matrix inversion failed. Aborting... \n");
        exit(0);
    } else {
//      overwrite the upper half
        for (i = 0; i < n; i++) {
            for (j = i+1; j < n; j++) {
                A[ i*n+j ] = A[ j*n+i ];
            }
        }
    }

    return logdet;
}		/* -----  end of function logdet_inverse_of_mat  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  binv_update
 *  Description:  Update B_INV for the next causal SNP.
 *                B_INV is of size nsnps-by-nsnps
 *                Copy zpos-row and zpos-column to vectors Drow and Dcol
 *                Drow = B[zpos, :]
 *                Dcol = B[:, zpos]
 *                B[ij] = B[ij] - mod * Dcol[i] * Drow[j]
 * =====================================================================================
 */
    void
binv_update ( int nsnps, int zpos, double mod, double* B, double* Drow, double* Dcol )
{
    int i, j;
    
    for ( i=0; i < nsnps; i++ ) {
        Drow[i] = B[ zpos * nsnps + i ];
        Dcol[i] = B[ i * nsnps + zpos ];
    }


    for ( i=0; i < nsnps; i++ ) {
        for ( j=0; j < nsnps; j++ ) {
            B[ i*nsnps + j ] -= mod * Dcol[i] * Drow[j];
        }
    }

}		/* -----  end of function binv_update  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  get_zcomps
 *  Description:  Calculate zcomps for each z-state
 *                mL = sum_z (Pz * Nz)
 *                zcomps = log(Pz) + log(Nz)
 * =====================================================================================
 */
    void
get_zcomps ( int nsnps, int nsample, int zlen, 
             double pi, double mu, double sigma, double sigmabg, double tau,
             int* ZARR, int* ZNORM, 
             double* GT, double* GX,
             double* ZCOMPS, double* BZINV, double* SZINV )
{
    int     i, j, k, z;
    int     nz, zindx, zpos;
    
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
    double* GX_MZ;                              // length of N

    double  *DUM1, *DUM2, *DUM3, *DUM4;

    B_INV = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    S_INV = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );
    GX_MZ = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    DUM1  = (double *)mkl_malloc(           nsnps   * sizeof( double ), 64 );
    DUM2  = (double *)mkl_malloc(           nsnps   * sizeof( double ), 64 );
    DUM3  = (double *)mkl_malloc( nsnps   * nsample * sizeof( double ), 64 );
    DUM4  = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );


    if (B_INV == NULL || S_INV == NULL || GX_MZ == NULL || DUM1 == NULL || DUM2 == NULL || DUM3 == NULL || DUM4 == NULL) {
        printf( "C++ Error: Can't allocate memory for z-specific Bz / Sz. Aborting... \n");
        mkl_free(B_INV);
        mkl_free(S_INV);
        mkl_free(GX_MZ);
        mkl_free(DUM1);
        mkl_free(DUM2);
        mkl_free(DUM3);
        mkl_free(DUM4);
        exit(0);
    }

    sigma2 = sigma * sigma;
    sigmabg2 = sigmabg * sigmabg;
    hdiff = (1 / sigma2) - (1 / sigmabg2);

//   Initialize B_INV and S_INV
    for (i = 0; i < (nsnps*nsnps); i++) {
        B_INV[i] = 0.0;
    }
    for (i = 0; i < (nsample*nsample); i++) {
        S_INV[i] = 0.0;
    }


    a_mat_matT(nsnps, nsample, tau, GT, B_INV);
    for (i = 0; i < nsnps; i++) {
        B_INV[ i*nsnps + i ] += 1 / sigmabg2;
    }
    logB0det = logdet_inverse_of_mat(nsnps, B_INV);
    for (i = 0; i < (nsnps*nsnps); i++) {
        BZINV[i] = B_INV[i];
    }

    printf("Initialized matrices for %d zcomps\n", zlen);
    zindx = 0;
    for (z = 0; z < zlen; z++) {
        nz = ZNORM[z];
        
        for (i = 0; i < (nsnps*nsnps); i++) {
            B_INV[i] = BZINV[i];
        }
        for (i = 0; i < nsample; i++) {
            GX_MZ[i] = GX[i];
        }
        logBZdet = logB0det;
        
        if (z > 2050) {
            printf("zstate: %d, Norm: %d, Elements: ", z, nz);
        }
        for (i = 0; i < nz; i++) {
            zpos = ZARR[zindx + i];
            if (z > 2050) {
                printf("%d ", zpos);
            }
            mod_denom = 1 + hdiff * B_INV[ zpos*nsnps + zpos ];
            mod = hdiff / mod_denom;
            logBZdet += log(mod_denom);
            if (mod != 0) {
                binv_update(nsnps, zpos, mod, B_INV, DUM1, DUM2);
            }
            for (j = 0; j < nsample; j++) {
                GX_MZ[j] -= mu * GT[ zpos*nsample + j ];
            }
        }
        if (z > 2050) {
            printf("\n");
        }
        
        if (z > 2050) {
            printf("log_probz \n");
        }
        log_probz = nz * log(pi) + (nsnps - nz) * log(1 - pi);
        
//      No need to initialize S_INV, because it will be overwritten.
//      DUM3 is a nsnps-by-nsample matrix which is only used as a scratch
        if (z > 2050) {
            printf("xT_Binv_x \n");
            printf("mod: %f %f %f\n", hdiff, mod_denom, mod);
            printf("nsnps, nsample, alpha: %d, %d, %f\n", nsnps, nsample, (-tau * tau));
            printf("%f %f %f\n", B_INV[0], B_INV[1], B_INV[2]);
            a_matT_matb_mat_debug ( nsnps, nsample, (-tau*tau), GT, B_INV, S_INV, DUM3);
        }
        a_matT_matb_mat ( nsnps, nsample, (-tau*tau), GT, B_INV, S_INV, DUM3 );
        if (z > 2050) {
            printf("add tau \n");
        }
        for (i = 0; i < nsample; i++) {
            S_INV[ i*nsample + i ] += tau;
        }

        if (z > 2050) {
            printf("logSZdet \n");
        }
        logSZdet = - (nsample * log(tau)) + ((nsnps - nz) * log(sigmabg2)) + (nz * log(sigma2)) + logBZdet;
        if (z > 2050) {
            printf("nterm \n");
        }
        nterm = vecT_smat_vec ( nsample, GX_MZ, S_INV, DUM4 );

        if (z > 2050) {
            printf("log_normz \n");
        }
        log_normz = - 0.5 * (logSZdet + (nsample * log(2 * _PI)) + nterm);
        
        //ZCOMPS[z] = GX[0];
        if (z > 2050) {
            printf("ZCOMPS[z] \n");
        }
        ZCOMPS[z] = log_probz + log_normz;
        
        if (z > 2050) {
            printf("BZINV \n");
        }
        for (i = 0; i < (nsnps*nsnps); i++) {
            BZINV[ z*nsnps*nsnps + i ] = B_INV[i];
        }
        if (z > 2050) {
            printf("SZINV \n");
        }
        for (i = 0; i < (nsample*nsample); i++) {
            SZINV[ z*nsample*nsample + i ] = S_INV[i];
        }
        
        zindx += nz;
    
    }
    printf("Computed ZCOMPS\n");

    mkl_free(B_INV);
    mkl_free(S_INV);
    mkl_free(GX_MZ);
    mkl_free(DUM1);
    mkl_free(DUM2);
    mkl_free(DUM3);
    mkl_free(DUM4);

}		/* -----  end of function zcomps  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  get_grad
 *  Description:  Compute the gradient of the log marginal likelihood
 * =====================================================================================
 */
    void
get_grad ( int nsnps, int nsample, int zlen, 
             double pi, double mu, double sigma, double sigmabg, double tau,
             int* ZARR, int* ZNORM, 
             double* GT, double* GX,
             double* ZCOMPS, double* BZINV, double* SZINV, double* GRAD )
{
    int i, j, k, z;
    int nz, zindx, zpos;

    double  sigma2, sigma3;
    double  sigmabg2, sigmabg3;
    double  tau2;

    double  picomp, mucomp, sigmacomp, sigmabgcomp, taucomp;
    double  pi_grad, mu_grad, sigma_grad, sigmabg_grad, tau_grad;
    double  dlogdetS_dsigma, dlogdetS_dsigmabg, dlogdetS_dtau;
    
    double* B_INV;
    double* S_INV;
    double* GX_MZ;                              // length of N
    double* ZT_GT;
    double* DBINV_DSIGMA;
    double* DSINV_DSIGMA;
    double* DBINV_DSIGMABG;
    double* DSINV_DSIGMABG;
    double* DSINV_DTAU;
    double* DUM1;
    double* DUM3;

    B_INV = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    S_INV = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );
    GX_MZ = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    ZT_GT = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    DUM1  = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    DUM3  = (double *)mkl_malloc( nsnps   * nsample * sizeof( double ), 64 );

    DBINV_DSIGMA   = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    DSINV_DSIGMA   = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );
    DBINV_DSIGMABG = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    DSINV_DSIGMABG = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );
    DSINV_DTAU     = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );

    if (B_INV == NULL || S_INV == NULL || GX_MZ == NULL || ZT_GT == NULL || 
            DBINV_DSIGMA == NULL || DSINV_DSIGMA == NULL || DBINV_DSIGMABG == NULL || DSINV_DSIGMABG == NULL || DSINV_DTAU == NULL ||
            DUM1 == NULL || DUM3 == NULL) {
        printf( "C++ Error: Can't allocate memory for z-specific Bz / Sz. Aborting... \n");
        mkl_free(B_INV);
        mkl_free(S_INV);
        mkl_free(GX_MZ);
        mkl_free(ZT_GT);
        mkl_free(DUM1);
        mkl_free(DUM3);
        mkl_free(DBINV_DSIGMA);
        mkl_free(DSINV_DSIGMA);
        mkl_free(DBINV_DSIGMABG);
        mkl_free(DSINV_DSIGMABG);
        mkl_free(DSINV_DTAU);
        exit(0);
    }
    else {
//        printf ("Successfully allocated memories \n");
    }


    sigma2 = sigma * sigma;
    sigma3 = sigma2 * sigma;
    sigmabg2 = sigmabg * sigmabg;
    sigmabg3 = sigmabg2 * sigmabg;
    tau2 = tau * tau;

    pi_grad = 0.0;
    mu_grad = 0.0;
    sigma_grad = 0.0;
    sigmabg_grad = 0.0;
    tau_grad = 0.0;

    zindx = 0;
    for ( z = 0; z < zlen; z++ ) {

        nz = ZNORM[z];


        for ( i = 0; i < (nsnps*nsnps); i++ ) {
            B_INV[i] = BZINV[ z*nsnps*nsnps + i ];
            DBINV_DSIGMA[i] = 0.0;
            DBINV_DSIGMABG[i] = 0.0;
        }
        a_mat_matT ( nsnps, nsnps, ( 2.0 / sigmabg3 ), B_INV, DBINV_DSIGMABG );
        for ( i = 0; i < (nsample*nsample); i++ ) {
            S_INV[i] = SZINV[ z*nsample*nsample + i ];
            DSINV_DSIGMA[i] = 0.0;
            DSINV_DSIGMABG[i] = 0.0;
            DSINV_DTAU[i] = 0.0;
        }
        for (i = 0; i < nsample; i++) {
            GX_MZ[i] = GX[i];
            ZT_GT[i] = 0.0;
        }
 
        picomp = (nz / pi) - ((nsnps - nz) / (1 - pi));
        pi_grad += ZCOMPS[z] * picomp;

//        printf ("Number of causal SNPS in zstate %d: %d. ZCOMPS = %f \n", z, nz, ZCOMPS[z]);
        for (i = 0; i < nz; i++) {
            zpos = ZARR[zindx + i];
//            printf("Updating GX_MZ and ZT_GT for zpos = %d\n", zpos);
            for (j = 0; j < nsample; j++) {
                GX_MZ[j] -= mu * GT[ zpos*nsample + j ];
                ZT_GT[j] += GT[ zpos*nsample + j ];
            }
        }
        mucomp = vecAT_smat_vecB( nsample, ZT_GT, S_INV, GX_MZ, DUM1 );
        mu_grad += ZCOMPS[z] * mucomp;

        dlogdetS_dsigma   = 0.0;
        dlogdetS_dsigmabg = (nsnps * sigmabg2) - mat_trace( nsnps, B_INV );
        for ( i = 0; i < nz; i++ ) {
            zpos = ZARR[zindx + i];
            dlogdetS_dsigma   += sigma2   - B_INV[ zpos*nsnps+zpos ];
            dlogdetS_dsigmabg -= sigmabg2 - B_INV[ zpos*nsnps+zpos ];
            for ( j = 0; j < nsnps; j++ ) {
                for ( k = 0; k < nsnps; k++ ) {
                    DBINV_DSIGMA  [ j*nsnps+k ] += 2.0 * B_INV[ j*nsnps+zpos ] * B_INV[ zpos*nsnps+k ] / sigma3;
                    DBINV_DSIGMABG[ j*nsnps+k ] -= 2.0 * B_INV[ j*nsnps+zpos ] * B_INV[ zpos*nsnps+k ] / sigmabg3;
                }
            }
        }
        dlogdetS_dsigma   = 2.0 * dlogdetS_dsigma   / sigma3;
        dlogdetS_dsigmabg = 2.0 * dlogdetS_dsigmabg / sigmabg3;
        a_matT_matb_mat ( nsnps, nsample, (-tau*tau), GT, DBINV_DSIGMA,   DSINV_DSIGMA,   DUM3 );
        a_matT_matb_mat ( nsnps, nsample, (-tau*tau), GT, DBINV_DSIGMABG, DSINV_DSIGMABG, DUM3 );
        sigmacomp     = - 0.5 * ( dlogdetS_dsigma   + vecAT_smat_vecB( nsample, GX_MZ, DSINV_DSIGMA,   GX_MZ, DUM1 ) );
        sigmabgcomp   = - 0.5 * ( dlogdetS_dsigmabg + vecAT_smat_vecB( nsample, GX_MZ, DSINV_DSIGMABG, GX_MZ, DUM1 ) );
        sigma_grad   += ZCOMPS[z] * sigmacomp;
        sigmabg_grad += ZCOMPS[z] * sigmabgcomp;
        
        dlogdetS_dtau = - mat_trace( nsample, S_INV ) / tau2;
        a_mat_matT ( nsample, nsample, (1 / tau2), S_INV, DSINV_DTAU );
        taucomp = - 0.5 * ( dlogdetS_dtau + vecAT_smat_vecB( nsample, GX_MZ, DSINV_DTAU, GX_MZ, DUM1 ) );
        tau_grad += ZCOMPS[z] * taucomp;
        
        zindx += nz;
    }

    GRAD[0] = pi_grad;
    GRAD[1] = mu_grad;
    GRAD[2] = sigma_grad;
    GRAD[3] = sigmabg_grad;
    GRAD[4] = tau_grad;

    mkl_free(B_INV);
    mkl_free(S_INV);
    mkl_free(GX_MZ);
    mkl_free(ZT_GT);
    mkl_free(DUM1);
    mkl_free(DUM3);
    mkl_free(DBINV_DSIGMA);
    mkl_free(DSINV_DSIGMA);
    mkl_free(DBINV_DSIGMABG);
    mkl_free(DSINV_DSIGMABG);
    mkl_free(DSINV_DTAU);

}		/* -----  end of function get_grad  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  logmarglik
 *  Description:  Wrapper function providing the marginal likelihood and the gradient
 * =====================================================================================
 */
    double
logmarglik ( int     nsnps,
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
             double* GRAD )
{
    int     i, k;
    double  logk;
    double  zcompsum;
    double  logmL;
    
    double* BZINV; //for each zstate BZ is a matrix of size I x I
    double* SZINV; //for each zstate S  is a matrix of size N x N

    //ProfilerStart("czcompgrad_from_c.prof");
    printf("Entering C function\n");
    
    BZINV = (double *)mkl_malloc( zlen * nsnps   * nsnps   * sizeof( double ), 64 );
    SZINV = (double *)mkl_malloc( zlen * nsample * nsample * sizeof( double ), 64 );
    
    if (BZINV == NULL || SZINV == NULL) {
        printf( "C++ Error: Can't allocate memory for Bz / Sz. Aborting... \n");
        mkl_free(BZINV);
        mkl_free(SZINV);
        exit(0);
    }
    
    get_zcomps ( nsnps, nsample, zlen, pi, mu, sigma, sigmabg, tau, ZARR, ZNORM, GT, GX, ZCOMPS, BZINV, SZINV );
    
    logk = ZCOMPS[0];
    for ( i=1; i < zlen; i++ ) {
        if ( ZCOMPS[i] > logk ) {
            logk = ZCOMPS[i];
        }
    }
    zcompsum = 0.0;
    for (i = 0; i < zlen; i++) {
        zcompsum += exp(ZCOMPS[i] - logk);
    }
    logmL = log(zcompsum) + logk ;

    for ( i=0; i < zlen; i++ ) {
        ZCOMPS[i] = exp(ZCOMPS[i] - logmL);
    }
        
    if (get_gradient) {
//      printf ("I have to calculate gradients \n");
        get_grad ( nsnps, nsample, zlen, pi, mu, sigma, sigmabg, tau, ZARR, ZNORM, GT, GX, ZCOMPS, BZINV, SZINV, GRAD);
    }
    
    mkl_free(BZINV);
    mkl_free(SZINV);

    //ProfilerStop();
    
    return logmL;

}		/* -----  end of function logmarglik  ----- */

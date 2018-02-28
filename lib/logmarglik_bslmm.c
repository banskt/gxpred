/*
 * =====================================================================================
 *
 *       Filename:  logmarglik.c
 *
 *    Description:  Log marginal likelihood and gradient for the optimization
 *
 *        Version:  1.0
 *        Created:  22/07/17 11:36:53
 *       Revision:  1.1
 *     Revised on:  27/10/17
 *       Compiler:  icc / gcc
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
 *         Name:  a_mat_mat
 *  Description:  calculate aAB where a is scalar,
 *                A is a m-by-n matrix
 *                B is a n-by-k matrix
 *                C holds the result and is of size m-by-k
 * =====================================================================================
 */
    void
a_mat_mat ( int m, int n, int k, double alpha, double* A, double* B, double* C )
{
    double zero = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, alpha, A, n, B, k, zero, C, k);

}       /* -----  end of function a_mat_matT  ----- */


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


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  smat_vec
 *  Description:  calculate Aw, where
 *                A is a n-by-n symmetric matrix
 *                w is a vector of size n
 *                D holds the result, vector of size n
 * =====================================================================================
 */
    void
smat_vec ( int n, double* A, double* w, double* D )
{
    double zero = 0.0;
    double one  = 1.0;
    cblas_dsymv(CblasRowMajor, CblasLower, n, one, A, n, w, 1, zero, D, 1);

}		/* -----  end of function smat_vec  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  mat_vec
 *  Description:  calculate alpha * Aw, where:
 *                A is a m-by-n general matrix
 *                w is a vector of size n
 *                D holds the result, vector of size m
 * =====================================================================================
 */
    void
mat_vec ( int m, int n, double alpha, double* A, double* w, double* D )
{
    double zero = 0.0;
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, w, 1, zero, D, 1);

}		/* -----  end of function mat_vec  ----- */




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
    bool
logdet_inverse_of_mat ( int n, double* A, double tau, double sigmabg2, double* logdet )
{
    int i, j;
    int info = 1;

    // Cholesky factorization of a symmetric (Hermitian) positive-definite matrix
    info = LAPACKE_dpotrf (LAPACK_ROW_MAJOR, 'L', n , A, n);

    if ( info != 0 ) {
        printf ("C Error: Cholesky factorization failed with errorcode %d.\n", info);
        *logdet = 0.0;
	return false;
    }

    *logdet = 0.0;
    for (i = 0; i < n; i++) {
        *logdet += 2 * log(A[ i*n+i ]);
    }

    // inverse of a symmetric (Hermitian) positive-definite matrix using the Cholesky factorization.
    info = LAPACKE_dpotri (LAPACK_ROW_MAJOR, 'L', n, A, n);

    if (info != 0) {
	printf ("C Error: Matrix inversion failed.\n");
        return false;
    }

    // succesful inversion. overwrite the upper half
    for (i = 0; i < n; i++) {
        for (j = i+1; j < n; j++) {
	    A[ i*n+j ] = A[ j*n+i ];
        }
    }

    return true;
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
    bool
get_zcomps ( int nsnps, int nsample, int zlen, 
             double* PI, double mu, double sigma, double sigmabg, double tau,
             int* ZARR, int* ZNORM, 
             double* GT, double* GX,
             double* ZCOMPS, double* BZINV, double* SZINV, bool debug )
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
    double  log_probz0;
    double  log_normz;
    bool    success;

    double* B_INV;
    double* S_INV;
    double* GX_MZ;                              // length of N

    double  *DUM1, *DUM2, *DUM3, *DUM4;

    B_INV = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    if (B_INV == NULL) {success = false; goto cleanup_zcomps_B_INV;}

    S_INV = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );
    if (S_INV == NULL) {success = false; goto cleanup_zcomps_S_INV;}

    GX_MZ = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    if (GX_MZ == NULL) {success = false; goto cleanup_zcomps_GX_MZ;}

    DUM1  = (double *)mkl_malloc(           nsnps   * sizeof( double ), 64 );
    if (DUM1 == NULL)  {success = false; goto cleanup_zcomps_DUM1;}

    DUM2  = (double *)mkl_malloc(           nsnps   * sizeof( double ), 64 );
    if (DUM2 == NULL)  {success = false; goto cleanup_zcomps_DUM2;}

    DUM3  = (double *)mkl_malloc( nsnps   * nsample * sizeof( double ), 64 );
    if (DUM3 == NULL)  {success = false; goto cleanup_zcomps_DUM3;}

    DUM4  = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    if (DUM4 == NULL)  {success = false; goto cleanup_zcomps_DUM4;}

    if (debug) {
        printf ( "Succesfully allocated memories for internal ZCOMPS.\n" );
    }

    sigma2 = sigma * sigma;
    sigmabg2 = sigmabg * sigmabg;
    hdiff = (1 / (sigma2 + sigmabg2)) - (1 / sigmabg2);

//   Initialize B_INV and S_INV
    for (i = 0; i < (nsnps*nsnps); i++) {
        B_INV[i] = 0.0;
    }
    for (i = 0; i < (nsample*nsample); i++) {
        S_INV[i] = 0.0;
    }
    if (debug) {
        printf ( "BINV and SINV updated.\n" );
    }


    a_mat_matT(nsnps, nsample, tau, GT, B_INV);
    for (i = 0; i < nsnps; i++) {
        B_INV[ i*nsnps + i ] += 1 / sigmabg2;
    }

    success = logdet_inverse_of_mat(nsnps, B_INV, tau, sigmabg2, &logB0det);

    if (success == false) {
	goto cleanup_zcomps;
    }

    for (i = 0; i < (nsnps*nsnps); i++) {
        BZINV[i] = B_INV[i];
    }

    log_probz0 = 0.0;
    for (i = 0; i < nsnps; i++) {
        if ( PI[i] < 1.0 ) {
            log_probz0 += log(1 - PI[i]);
        } else {
            success = false;
            printf ( "Error. Pi value reached %f. Aborting...\n", PI[i] );
            goto cleanup_zcomps;
        }
    }

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

        log_probz = log_probz0;
        for (i = 0; i < nz; i++) {
            zpos = ZARR[zindx + i];
            mod_denom = 1 + hdiff * B_INV[ zpos*nsnps + zpos ];
            mod = hdiff / mod_denom;
            logBZdet += log(mod_denom);
            binv_update(nsnps, zpos, mod, B_INV, DUM1, DUM2);
            for (j = 0; j < nsample; j++) {
                GX_MZ[j] -= mu * GT[ zpos*nsample + j ];
            }
            if ( PI[zpos] > 0.0 ) {
                log_probz += log(PI[zpos] / (1 - PI[zpos]));
            } else {
                success = false;
                printf ( "Error. Pi value reached %f. Aborting...\n", PI[i] );
                goto cleanup_zcomps;
            }
        }
        
//      No need to initialize S_INV, because it will be overwritten.
//      DUM3 is a nsnps-by-nsample matrix which is only used as a scratch
        a_matT_matb_mat ( nsnps, nsample, (-tau*tau), GT, B_INV, S_INV, DUM3 );
        for (i = 0; i < nsample; i++) {
            S_INV[ i*nsample + i ] += tau;
        }
        logSZdet = - (nsample * log(tau)) + ((nsnps - nz) * log(sigmabg2)) + (nz * log(sigma2 + sigmabg2)) + logBZdet;
        nterm = vecT_smat_vec ( nsample, GX_MZ, S_INV, DUM4 );

        log_normz = - 0.5 * (logSZdet + (nsample * log(2 * _PI)) + nterm);
        
        ZCOMPS[z] = log_probz + log_normz;

        for (i = 0; i < (nsnps*nsnps); i++) {
            BZINV[ (unsigned long)z*nsnps*nsnps + i ] = B_INV[i];
        }

        for (i = 0; i < (nsample*nsample); i++) {
            SZINV[ (unsigned long)z*nsample*nsample + i ] = S_INV[i];
        }
        
        zindx += nz;
    
    }

cleanup_zcomps:
cleanup_zcomps_DUM4:
    mkl_free(DUM4);

cleanup_zcomps_DUM3:
    mkl_free(DUM3);

cleanup_zcomps_DUM2:
    mkl_free(DUM2);

cleanup_zcomps_DUM1:
    mkl_free(DUM1);

cleanup_zcomps_GX_MZ:
    mkl_free(GX_MZ);

cleanup_zcomps_S_INV:
    mkl_free(S_INV);

cleanup_zcomps_B_INV:
    mkl_free(B_INV);

    return success;
}		/* -----  end of function zcomps  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  get_grad
 *  Description:  Compute the gradient of the log marginal likelihood
 * =====================================================================================
 */
    void
get_grad ( int nsnps, int nsample, int zlen, int nfeat,
             double* PI, double mu, double sigma, double sigmabg, double tau,
             int* ZARR, int* ZNORM, 
             double* GT, double* GX, double* FEAT,
             double* ZCOMPS, double* BZINV, double* SZINV, double* GRAD )
{
    int i, j, k, z;
    int nz, zindx, zpos;

    double  sigma2;
    double  sigmabg2;
    double  sigmabg3;
    double  sigmabg4;
    double  tau2;
    double  sigmaz4;
    double  sigma_by_sigmaz4;
    double  sigmabg_by_sigmaz4;

    double  mucomp, sigmacomp, sigmabgcomp, taucomp;
    double  mu_grad, sigma_grad, sigmabg_grad, tau_grad;
    double  dlogdetS_dsigma, dlogdetS_dsigmabg, dlogdetS_dtau;
    double  innersumpi;
    
    double* INVSIGMAZ4;
    double* INVSIGMAZ4_0;
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
    double* DUM5;
    double* PICOMP;

    B_INV  = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    S_INV  = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );
    GX_MZ  = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    ZT_GT  = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    DUM1   = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    DUM3   = (double *)mkl_malloc( nsnps   * nsample * sizeof( double ), 64 );
    DUM5   = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    PICOMP = (double *)mkl_malloc( nfeat             * sizeof( double ), 64 );

    DBINV_DSIGMA   = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    DSINV_DSIGMA   = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );
    DBINV_DSIGMABG = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    DSINV_DSIGMABG = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );
    DSINV_DTAU     = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );
    INVSIGMAZ4        = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    INVSIGMAZ4_0   = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );

    if (B_INV == NULL || S_INV == NULL || GX_MZ == NULL || ZT_GT == NULL || 
            DBINV_DSIGMA == NULL || DSINV_DSIGMA == NULL || DBINV_DSIGMABG == NULL || DSINV_DSIGMABG == NULL || DSINV_DTAU == NULL ||
            DUM1 == NULL || DUM3 == NULL || INVSIGMAZ4 == NULL || INVSIGMAZ4_0 == NULL || DUM5 == NULL || PICOMP == NULL) {
        printf( "C Error: Can't allocate memory for z-specific Bz / Sz. Aborting... \n");
        mkl_free(B_INV);
        mkl_free(S_INV);
        mkl_free(GX_MZ);
        mkl_free(ZT_GT);
        mkl_free(DUM1);
        mkl_free(DUM3);
        mkl_free(DUM5);
        mkl_free(DBINV_DSIGMA);
        mkl_free(DSINV_DSIGMA);
        mkl_free(DBINV_DSIGMABG);
        mkl_free(DSINV_DSIGMABG);
        mkl_free(DSINV_DTAU);
        mkl_free(INVSIGMAZ4);
        mkl_free(INVSIGMAZ4_0);
        mkl_free(PICOMP);
        exit(0);
    }
    else {
//        printf ("Successfully allocated memories for gradient calculation.\n");
    }


    sigma2 = sigma * sigma;
    sigmabg2 = sigmabg * sigmabg;
    sigmabg4 = sigmabg2 * sigmabg2;
    tau2 = tau * tau;

    mu_grad = 0.0;
    sigma_grad = 0.0;
    sigmabg_grad = 0.0;
    tau_grad = 0.0;

    for ( i = 0; i < (nsnps*nsnps); i++ ) {
        INVSIGMAZ4_0[i] = 0.0;
    }

    for (i = 0; i < nsnps; i++) {
        INVSIGMAZ4_0[i*nsnps+i] = (-2 * sigmabg) / sigmabg4;
    }

    for (k = 0; k < nfeat; k++) {
        PICOMP[k] = 0.0;
        GRAD[k] = 0.0;
    }

    for (i = 0; i < nsnps; ++i) {
        for (k = 0; k < nfeat; k++) {
            PICOMP[k] += - PI[i] * FEAT[ i * nfeat + k ];
        }
    }

    zindx = 0;
    for ( z = 0; z < zlen; z++ ) {

        nz = ZNORM[z];

        for ( i = 0; i < (nsnps*nsnps); i++ ) {
            B_INV[i] = BZINV[ (unsigned long)z*nsnps*nsnps + i ];
            DBINV_DSIGMA[i] = 0.0;
            DBINV_DSIGMABG[i] = 0.0;
            INVSIGMAZ4[i] = INVSIGMAZ4_0[i];
        }

        for ( i = 0; i < nz ; i++) {
            zpos = ZARR[zindx + i];
            INVSIGMAZ4[ zpos*nsnps + zpos] = (-2 * sigmabg) / (( sigma2 + sigmabg2)*( sigma2 + sigmabg2));
        }

        // TODO: this could be improved for speed and memory
        a_mat_mat(nsnps, nsnps, nsnps, 1, INVSIGMAZ4, B_INV, DUM5 );
        a_mat_mat(nsnps, nsnps, nsnps, -1, B_INV, DUM5, DBINV_DSIGMABG );

        for ( i = 0; i < (nsample*nsample); i++ ) {
            S_INV[i] = SZINV[ (unsigned long)z*nsample*nsample + i ];
            DSINV_DSIGMA[i] = 0.0;
            DSINV_DSIGMABG[i] = 0.0;
            DSINV_DTAU[i] = 0.0;
        }
        for (i = 0; i < nsample; i++) {
            GX_MZ[i] = GX[i];
            ZT_GT[i] = 0.0;
        }

        for (k = 0; k < nfeat; k++) {
            innersumpi = PICOMP[k];
            for (i = 0; i < nz; i++) {
                zpos = ZARR[zindx + i];
                innersumpi += FEAT[ zpos * nfeat + k ];
            }
            GRAD[k] += ZCOMPS[z] * innersumpi;
        }

        for (i = 0; i < nz; i++) {
            zpos = ZARR[zindx + i];
            for (j = 0; j < nsample; j++) {
                GX_MZ[j] -= mu * GT[ zpos*nsample + j ];
                ZT_GT[j] += GT[ zpos*nsample + j ];
            }
        }
        mucomp = vecAT_smat_vecB( nsample, ZT_GT, S_INV, GX_MZ, DUM1 );
        mu_grad += ZCOMPS[z] * mucomp;

        dlogdetS_dsigma   = 0.0;
        dlogdetS_dsigmabg = nsnps * (2 / sigmabg ) - ( 2 * (sigmabg / sigmabg4) * mat_trace( nsnps, B_INV ) );
        sigmaz4 =  (sigma2 + sigmabg2) * (sigma2 + sigmabg2);
        sigma_by_sigmaz4 = sigma / sigmaz4;
        sigmabg_by_sigmaz4 = sigmabg / sigmaz4;
        for ( i = 0; i < nz; i++ ) {
            zpos = ZARR[zindx + i];
            dlogdetS_dsigma   += sigma2 + sigmabg2 - B_INV[ zpos*nsnps+zpos ];
            dlogdetS_dsigmabg += 2 * sigmabg_by_sigmaz4 * (sigma2 + sigmabg2 - B_INV[ zpos*nsnps+zpos ]);
            dlogdetS_dsigmabg -= ((2 * sigmabg) / sigmabg4) * (sigmabg2 - B_INV[ zpos*nsnps+zpos]);
            for ( j = 0; j < nsnps; j++ ) {
                for ( k = 0; k < nsnps; k++ ) {
                    DBINV_DSIGMA  [ j*nsnps+k ] += 2.0 * B_INV[ j*nsnps+zpos ] * B_INV[ zpos*nsnps+k ] * sigma_by_sigmaz4 ;
                    // DBINV_DSIGMABG[ j*nsnps+k ] -= 2.0 * B_INV[ j*nsnps+zpos ] * B_INV[ zpos*nsnps+k ] / sigmabg3;
                }
            }
        }
        dlogdetS_dsigma   = 2.0 * dlogdetS_dsigma   * sigma_by_sigmaz4;
        // dlogdetS_dsigmabg = 2.0 * dlogdetS_dsigmabg ;
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

    GRAD[ nfeat + 0 ] = mu_grad;
    GRAD[ nfeat + 1 ] = sigma_grad;
    GRAD[ nfeat + 2 ] = sigmabg_grad;
    GRAD[ nfeat + 3 ] = tau_grad;

    mkl_free(B_INV);
    mkl_free(S_INV);
    mkl_free(GX_MZ);
    mkl_free(ZT_GT);
    mkl_free(DUM1);
    mkl_free(DUM3);
    mkl_free(DUM5);
    mkl_free(PICOMP);
    mkl_free(INVSIGMAZ4);
    mkl_free(INVSIGMAZ4_0);
    mkl_free(DBINV_DSIGMA);
    mkl_free(DSINV_DSIGMA);
    mkl_free(DBINV_DSIGMABG);
    mkl_free(DSINV_DSIGMABG);
    mkl_free(DSINV_DTAU);

}		/* -----  end of function get_grad  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  get_zexp
 *  Description:  Calculate m_vz (eq. 3.22)
 *                It is the mean of P(v_tg | x, y, theta, tau) Eq 3.13
 *                For every z-state, m_vz is a vector of size I.
 *                The result is stored in a single array ZEXP of size nz * I.
 * =====================================================================================
 */
    void
get_zexp ( int nsnps, int nsample, int zlen, 
             double mu, double sigma, double sigmabg, double tau,
             int* ZARR, int* ZNORM, 
             double* GT, double* GX,
             double* BZINV, double* ZEXP )
{

    double* GT_GX;
    double* M_VZ_;
    double* FACT0;
    double* FACTZ;
    double* S_VZ_;

    double sigma2;
    double sigmabg2;

    int i, z, zpos, zindx, nz;

    S_VZ_ = (double *)mkl_malloc( nsnps * nsnps * sizeof( double ), 64 );
    M_VZ_ = (double *)mkl_malloc(         nsnps * sizeof( double ), 64 );
    GT_GX = (double *)mkl_malloc(         nsnps * sizeof( double ), 64 );
    FACT0 = (double *)mkl_malloc(         nsnps * sizeof( double ), 64 );
    FACTZ = (double *)mkl_malloc(         nsnps * sizeof( double ), 64 );

    if ( S_VZ_ == NULL || M_VZ_ == NULL || GT_GX == NULL || FACT0 == NULL || FACTZ == NULL ) {
        printf( "C Error: Can't allocate memory GT_GX. Aborting... \n");
        mkl_free(M_VZ_);
        mkl_free(S_VZ_);
        mkl_free(GT_GX);
        mkl_free(FACT0);
        mkl_free(FACTZ);
        exit(0);
    }

    sigma2 = sigma * sigma;
    sigmabg2 = sigmabg * sigmabg;

    for ( i = 0; i < nsnps; i++ ) {
        GT_GX[i] = 0.0;
    }
    mat_vec ( nsnps, nsample, tau, GT, GX, GT_GX );
    for ( i = 0; i < nsnps; i++ ) {
        FACT0[i] = GT_GX[i];// + (mu / sigmabg2);
    }

    zindx = 0;
    for ( z = 0; z < zlen; z++ ) {

        nz = ZNORM[z];

        for ( i = 0; i < nsnps; i++ ) {
            FACTZ[i] = FACT0[i];
        }
        for ( i = 0; i < nz; i++ ) {
            zpos = ZARR[zindx + i];
            FACTZ[zpos] += (mu / (sigma2 + sigmabg2) );// - (mu / sigmabg2);
        }
        for (i = 0; i < (nsnps*nsnps); i++) {
            S_VZ_[i] = BZINV[ (unsigned long)z*nsnps*nsnps + i ];
        }
        smat_vec ( nsnps, S_VZ_, FACTZ, M_VZ_ );

        for ( i = 0; i < nsnps; i++ ) {
            ZEXP[ z * nsnps + i ] = M_VZ_[i];
        }

        zindx += nz;

    }

    mkl_free(M_VZ_);
    mkl_free(S_VZ_);
    mkl_free(GT_GX);
    mkl_free(FACT0);
    mkl_free(FACTZ);

}		/* -----  end of function get_zexp  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  logmarglik
 *  Description:  Wrapper function providing the marginal likelihood and the gradient
 * =====================================================================================
 */
    bool
logmarglik ( int     nsnps,
             int     nsample,
             int     zlen,
             int     nfeat,
             double* PI,
             double  mu,
             double  sigma,
             double  sigmabg,
             double  tau,
             bool    get_gradient,
             bool    get_Mvz,
             int*    ZARR,
             int*    ZNORM,
             double* GT,
             double* GX,
             double* FEAT,
             double* ZCOMPS,
             double* GRAD,
             double* ZEXP,
             double* logmL )
{
    int     i, k;
    double  logk;
    double  zcompsum;
    bool    success;
    bool    debug;
    
    double* BZINV; //for each zstate BZ is a matrix of size I x I
    double* SZINV; //for each zstate S  is a matrix of size N x N

    debug = false;

    //if (zlen > 10000) {
    //    debug = true;
    //}

    if (debug) {
        printf ("%d zstates, %d SNPs and %d samples\n", zlen, nsnps, nsample);
        printf ("No. of features: %d", nfeat);
        printf ("Size of double: %lu bytes\n", sizeof( double ));
        printf ("Size of float:  %lu bytes\n", sizeof( float ));
        printf ("Size required:  %f Gb\n", (double)((unsigned long)zlen * nsnps   * nsnps * sizeof(double)) / (1024 * 1024 * 1024));
    }
    
    BZINV = (double *)mkl_malloc( (unsigned long)zlen * nsnps   * nsnps   * sizeof( double ), 64 );
    if (BZINV == NULL) {success = false; goto cleanup_main_BZINV;}

    SZINV = (double *)mkl_malloc( (unsigned long)zlen * nsample * nsample * sizeof( double ), 64 );
    if (SZINV == NULL) {success = false; goto cleanup_main_SZINV;}
    
    success = get_zcomps ( nsnps, nsample, zlen, PI, mu, sigma, sigmabg, tau, ZARR, ZNORM, GT, GX, ZCOMPS, BZINV, SZINV, debug );
    if (debug) {
        printf ( "ZCOMPS calculated.\n" );
    }

    if (success == false) {
        logmL[0] = 0.0;
        printf( "C error: Zcomps calculation did not succeed.\n" );
        goto cleanup_main;
    } 

    logk = ZCOMPS[0];
    for ( i=1; i < zlen; i++ ) {
        if ( ZCOMPS[i] > logk ) {
            logk = ZCOMPS[i];
        }
    }
    if (debug) {
        printf ( "logk calculated.\n" );
    }

    zcompsum = 0.0;
    for (i = 0; i < zlen; i++) {
        zcompsum += exp(ZCOMPS[i] - logk);
    }
    if (debug) {
        printf ( "logmL calculating %f.\n", logmL[0] );
    }
    logmL[0] = log(zcompsum) + logk ;

    for ( i=0; i < zlen; i++ ) {
        ZCOMPS[i] = exp(ZCOMPS[i] - logmL[0]);
    }
    if (debug) {
        printf ( "ZCOMPS updated.\n" );
    }
        
    if (get_gradient) {
        get_grad ( nsnps, nsample, zlen, nfeat, PI, mu, sigma, sigmabg, tau, ZARR, ZNORM, GT, GX, FEAT, ZCOMPS, BZINV, SZINV, GRAD );
    }
    if (debug) {
        printf ( "Gradients calculated.\n" );
    }

    if (get_Mvz) {
        get_zexp ( nsnps, nsample, zlen, mu, sigma, sigmabg, tau, ZARR, ZNORM, GT, GX, BZINV, ZEXP );
    }
    if (debug) {
        printf ( "Everything done upto cleanup.\n" );
    }

cleanup_main:
cleanup_main_SZINV:
    mkl_free(SZINV);

cleanup_main_BZINV:
    mkl_free(BZINV);
    
    return success;

}		/* -----  end of function logmarglik  ----- */

/*
 * =====================================================================================
 *
 *       Filename:  logmarglik.c
 *
 *    Description:  Log marginal likelihood and gradient for the optimization
 *
 *        Version:  1.0
 *        Created:  05/07/17
 *       Revision:  1.1
 *     Revised on:  05/07/17
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
}       /* -----  end of function vecT_smat_vec  ----- */


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

}       /* -----  end of function vecAT_smat_vecB  ----- */


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

}       /* -----  end of function a_mat_matT  ----- */

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

}       /* -----  end of function a_mat_mat  ----- */


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
    
}       /* -----  end of function a_matT_matb_mat  ----- */


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

}       /* -----  end of function smat_vec  ----- */


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

}       /* -----  end of function mat_vec  ----- */




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

}       /* -----  end of function mat_trace  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  gauss_jordan_inversion
 *  Description:  calculate matrix inverse using Gauss-Jordan elimination method
 *                fast for small matrices
 *                
 * =====================================================================================
 */
    void
gauss_jordan_inversion ( int n, double *matrix, double *logdet, double *signdet )
{
    int i, j, k;
    double ratio;
    double a;
    int ncol = 2 * n;

//  hardcoded for fast calculation of single elements
    if (n == 1) {
        a = matrix[0];
        matrix[1] = 1 / a;
        *logdet = log(abs(a));
        *signdet = 1;
        if (a < 0) {
            *signdet *= -1;
        }

    } else {

//      creating the identity matrix beside the original one
        for(i = 0; i < n; ++i){
            for(j = n; j < 2*n; ++j){
                if(i==(j-n)) {
                    matrix[i * ncol + j] = 1.0;
                } else {
                    matrix[i * ncol + j] = 0.0;
                }
            }
        }

//      row operations to reduce original matrix to a diagonal matrix
        for(i = 0; i < n; ++i){
            for(j = 0; j < n; ++j){
                if(i != j){
                    ratio = matrix[j * ncol + i]/matrix[i * ncol + i];
                    for(k = 0; k < 2*n; ++k){
                        matrix[j * ncol + k] -= ratio * matrix[i * ncol + k];
                    }
                }
            }
        }

//      reducing to unit matrix, determinant is calculated from the diagonal matrix before the reduction
        *logdet = 0.0;
        *signdet = 1;
        for(i = 0; i < n; ++i){
            a = matrix[i * ncol + i];
            if (a < 0) {
                *signdet *= -1;
                *logdet += log(-a);
            } else {
                *logdet += log(a);
            }
            for(j = 0; j < 2*n; ++j){
				matrix[i * ncol + j] /= a;
            }
        }
    }
}		/* -----  end of function gauss_jordan_inversion  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  row_dot
 *  Description:  find u.dot.u where u is the kth row of a matrix A of dim m x n.
 * =====================================================================================
 */
	double
row_dot ( double* A, int n, int k )
{
	int i;
	double res;
	res = 0;
	for (i = 0; i < n; i++) {
		res += A[k * n + i] * A[k * n + i];
	}
	return res;
}		/* -----  end of function row_dot  ----- */


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
             double* PI, double mu, double sigma, double tau,
             int* ZARR, int* ZNORM, 
             double* GT, double* GX,
             double* ZCOMPS, double* BZINV, double* SZINV, double* logmL, bool debug )
{
    int     i, j, k, z;
    int     nz, zindx, zpos;
	int     ncol, zposi, zposj;
    
    double  sigma2;
    double  logdet, signdet;
    double  logSZdet;
    double  nterm;
    double  log_normz;
    bool    success;
    bool    this_is_causal;
    bool    initialize_logk;
    double  logk;
    double  zcompsum;

    double* B_INV;
    double* S_INV;
    double* GX_MZ;                              // length of N
	double* tauXXT;
    bool*   ZCOMPS_DEFINED;
    double* LOG_PROBZ;
	double* B_HAT;

    double  *DUM1, *DUM2;

    double *TMP1;

	success = true;

    B_INV = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    if (B_INV == NULL) {success = false; goto cleanup_zcomps_B_INV;}

    S_INV = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );
    if (S_INV == NULL) {success = false; goto cleanup_zcomps_S_INV;}

    GX_MZ = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    if (GX_MZ == NULL) {success = false; goto cleanup_zcomps_GX_MZ;}

	tauXXT  = (double*)mkl_malloc( nsnps * nsnps * sizeof( double ), 64 );
	if (tauXXT == NULL) {success = false; goto cleanup_zcomps_tauXXT;}

    DUM1  = (double *)mkl_malloc( nsnps   * nsample * sizeof( double ), 64 );
    if (DUM1 == NULL)  {success = false; goto cleanup_zcomps_DUM1;}

    DUM2  = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    if (DUM2 == NULL)  {success = false; goto cleanup_zcomps_DUM2;}

    ZCOMPS_DEFINED = (bool *)mkl_malloc(    zlen * sizeof( bool ), 64 );
    if (ZCOMPS_DEFINED == NULL)  {success = false; goto cleanup_zcomps_ZCOMPS_DEFINED;}

    LOG_PROBZ = (double *)mkl_malloc(    zlen * sizeof( double ), 64 );
    if (LOG_PROBZ == NULL)  {success = false; goto cleanup_zcomps_LOG_PROBZ;}

	TMP1 = (double *)mkl_malloc( nsnps * nsnps * sizeof( double ), 64 );

    if (debug) {
        printf ( "Succesfully allocated memories for internal ZCOMPS.\n" );
    }

    sigma2 = sigma * sigma;

	a_mat_matT ( nsnps, nsample, tau, GT, tauXXT );
	a_mat_matT ( nsnps, nsample, 1.0, GT, TMP1 );

	// check if any zcomps are undefined and calculate log(P(z))
	// loops over all zstates, very expensive
    zindx = 0;

    for (z = 0; z < zlen; z++) {
        nz = ZNORM[z];
        LOG_PROBZ[z] = 0.0;
        ZCOMPS_DEFINED[z] = true;
        for (i = 0; i < nsnps; i++) {
            // python equivalent: if i not in ZARR[zindx: (zindx + nz)]
            this_is_causal = false;
            for (j = 0; j < nz; j++) {
                zpos = ZARR[zindx + j];
                if (i == zpos) {
                    this_is_causal = true;
                }
            }
            if ( ! this_is_causal ) {
                if (PI[i] < 1.0) {
                   LOG_PROBZ[z] += log(1 - PI[i]);
                } else {
                    // impossible, PI = 1, z = 0
                    ZCOMPS_DEFINED[z] = false;
                    break;
                }
            } else { // is causal
                if (PI[i] > 0.0) {
                    LOG_PROBZ[z] += log(PI[i]);
                } else {
                    // impossible, PI = 0, z = 1
                    ZCOMPS_DEFINED[z] = false;
                    break;
                }
            }
        }
        zindx += nz;
    }

	if (debug) {printf("Calculated log P(z)\n");}

    // Re-initiate zindx for the rest of the calculation
    zindx = 0;

    for (z = 0; z < zlen; z++) {
        if (ZCOMPS_DEFINED[z]) {
            nz = ZNORM[z];
            for (i = 0; i < (nsnps*nsnps); i++) {
                B_INV[i] = 0.0; // initiate B_INV to a zero matrix
            }
            for (i = 0; i < nsample; i++) {
                GX_MZ[i] = GX[i]; 
            }

			// create the small matrix for inversion
			// in the Gauss-Jordan elimination scheme, we send a matrix with two halves
			// first half contains the original matrix, second half will contain the inverse after output
			// changes made in place.
			if (nz > 0) {
				B_HAT = (double *)mkl_malloc( 2 * nz * nz * sizeof( double ), 64 );
				ncol = 2 * nz;
				for (i = 0; i < nz; i++) {
					zposi = ZARR[zindx + i];
					for (j = 0; j < nz; j++) { // first half
		                zposj = ZARR[zindx + j];
						B_HAT[i * ncol + j] = tauXXT[zposi * nsnps + zposj];
	                }
					for (j = nz; j < 2 * nz; j++) { // second half
						B_HAT[i * ncol + j] = 0.0;
					}
					B_HAT[i * ncol + i] += 1 / sigma2;
				}
/*
				for (i = 0; i < nz; i++) {
					for (j = 0; j < nz; j++) {
						printf("%g ", B_HAT[i * ncol + j]);
					}
					printf("\n");
				}
*/
	
				// invert B_HAT and update B_INV
				gauss_jordan_inversion(nz, B_HAT, &logdet, &signdet); // we don't need the logdet but still...
				for (i = 0; i < nz; i++) {
					zposi = ZARR[zindx + i];
					for (j = 0; j < nz; j++) {
						zposj = ZARR[zindx + j];
						B_INV[zposi * nsnps + zposj] = B_HAT[i * ncol + nz + j];
					}
				}
				mkl_free(B_HAT);
			}

			logSZdet = - nsample * log(tau);
            for (i = 0; i < nz; i++) {
                zpos = ZARR[zindx + i];
                double tmpa = 1 + tau * sigma2 * row_dot(GT, nsample, zpos);
				logSZdet += log(tmpa);
                for (j = 0; j < nsample; j++) {
                    GX_MZ[j] -= mu * GT[ zpos*nsample + j ];
                }
            }
            
    //      No need to initialize S_INV, because it will be overwritten.
    //      DUM1 is a nsnps-by-nsample matrix which is only used as a scratch
            a_matT_matb_mat ( nsnps, nsample, (-tau*tau), GT, B_INV, S_INV, DUM1 );
            for (i = 0; i < nsample; i++) {
                S_INV[ i*nsample + i ] += tau;
            }
/*
			if (debug) {
				printf ("S_INV:\n");
				for (i = 0; i < nsnps; i++) {
					for (j = 0; j < nsnps; j++) {
						printf("%g ", B_INV[i * nsnps + j]);
					}
					printf("\n");
				}
			}
*/

            nterm = vecT_smat_vec ( nsample, GX_MZ, S_INV, DUM2 );

            log_normz = - 0.5 * (logSZdet + (nsample * log(2 * _PI)) + nterm);
            
            ZCOMPS[z] = LOG_PROBZ[z] + log_normz;

			//printf("%d\t%g\t%g\t%g\t%g\n", z, log_normz, logSZdet, nterm, (nsample * log(2 * _PI)));
            //printf("Zstate: %d\t%g\t%g\t%g\t%g\n", z, LOG_PROBZ[z], log_normz, logSZdet, nterm);

            for (i = 0; i < (nsnps*nsnps); i++) {
                BZINV[ (unsigned long)z*nsnps*nsnps + i ] = B_INV[i];
            }

            for (i = 0; i < (nsample*nsample); i++) {
                SZINV[ (unsigned long)z*nsample*nsample + i ] = S_INV[i];
            }
            
            zindx += nz;
        }
        else {
            printf("ZCOMPS_DEFINED[%d]: %d\n", z, ZCOMPS_DEFINED[z]);
        }
    }
  

    // ZCOMPS holds the values of log [ P(z|theta) * N(y|m_yz,S_yz) ] for each Z-state (numerator in log P(z|X,y,theta,tau))

    // We need to calculate the sum over ZCOMPS for the denominator of log P(z|X,y,theta,tau)
    // Note that, logmL(theta) = log sum[exp(ZCOMPS)]

    // logk workaround for overflow error of exp function
    // divide by logk and multiply later by logk
    // where logk = max(ZCOMPS)

    initialize_logk = true;
    for ( i=0; i < zlen; i++ ) {
        if ( ZCOMPS_DEFINED[i] ) {
            if (initialize_logk) {
                logk = ZCOMPS[i];
                initialize_logk = false;
            }
            if ( ZCOMPS[i] > logk) {
                logk = ZCOMPS[i];
            }
        }
    }

    if (debug) {
        printf ( "logk calculated.\n" );
    }

    zcompsum = 0.0;
    for (i = 0; i < zlen; i++) {
        if (ZCOMPS_DEFINED[i]) {
            //printf ("ZCOMPS[%d]: %f\n", i, ZCOMPS[i]);
            zcompsum += exp(ZCOMPS[i] - logk);    
        } else {
            //printf ("ZCOMPS[%d]: -infinity?\n", i);
        }
    }

    if (debug) {
        printf ( "logmL calculating %f.\n", logmL[0] );
    }
    logmL[0] = log(zcompsum) + logk ;

    if (debug) {
        printf ( "logmL calculated %f.\n", logmL[0] );
        printf ( "zcompsum: %f, logk: %f\n", log(zcompsum), logk);
    }

    // end of logk workaround

    // Eq. 3.25
    // Calculate P(z|X,y,theta,tau)
    // We are replacing the values of ZCOMPS with the actual probabilities
    for ( i=0; i < zlen; i++ ) {
        if (ZCOMPS_DEFINED[i]) {
            ZCOMPS[i] = exp(ZCOMPS[i] - logmL[0]);    
        } else {
            ZCOMPS[i] = 0;
        }
    }
    if (debug) {
        printf ( "ZCOMPS updated.\n" );
    }



cleanup_zcomps:
	mkl_free(TMP1);

cleanup_zcomps_LOG_PROBZ:
    mkl_free(LOG_PROBZ);

cleanup_zcomps_ZCOMPS_DEFINED:
    mkl_free(ZCOMPS_DEFINED);

cleanup_zcomps_DUM2:
    mkl_free(DUM2);

cleanup_zcomps_DUM1:
    mkl_free(DUM1);

cleanup_zcomps_tauXXT:
	mkl_free(tauXXT);

cleanup_zcomps_GX_MZ:
    mkl_free(GX_MZ);

cleanup_zcomps_S_INV:
    mkl_free(S_INV);

cleanup_zcomps_B_INV:
    mkl_free(B_INV);

    return success;
}       /* -----  end of function zcomps  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  get_grad
 *  Description:  Compute the gradient of the log marginal likelihood
 * =====================================================================================
 */
    void
get_grad ( int nsnps, int nsample, int zlen, int nfeat,
             double* PI, double mu, double sigma, double tau,
             int* ZARR, int* ZNORM, 
             double* GT, double* GX, double* FEAT,
             double* ZCOMPS, double* BZINV, double* SZINV, double* GRAD )
{
    int i, j, k, z;
    int nz, zindx, zpos;

    double  sigma2, sigma3;
    double  tau2;
    double  sigmaz4;
    double  sigma_by_sigmaz4;

    double  mucomp, sigmacomp, taucomp;
    double  mu_grad, sigma_grad, tau_grad;
    double  dlogdetS_dsigma, dlogdetS_dtau;
    double  innersumpi;
    
    double* B_INV;
    double* S_INV;
    double* GX_MZ;                              // length of N
    double* ZT_GT;
    double* DBINV_DSIGMA;
    double* DSINV_DSIGMA;
    double* DSINV_DTAU;
    double* DUM1;
    double* DUM3;
    double* PICOMP;

    B_INV  = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    S_INV  = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );
    GX_MZ  = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    ZT_GT  = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    DUM1   = (double *)mkl_malloc(           nsample * sizeof( double ), 64 );
    DUM3   = (double *)mkl_malloc( nsnps   * nsample * sizeof( double ), 64 );
    PICOMP = (double *)mkl_malloc( nfeat             * sizeof( double ), 64 );

    DBINV_DSIGMA   = (double *)mkl_malloc( nsnps   * nsnps   * sizeof( double ), 64 );
    DSINV_DSIGMA   = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );
    DSINV_DTAU     = (double *)mkl_malloc( nsample * nsample * sizeof( double ), 64 );

    if (B_INV == NULL || S_INV == NULL || GX_MZ == NULL || ZT_GT == NULL || 
            DBINV_DSIGMA == NULL || DSINV_DSIGMA == NULL || DSINV_DTAU == NULL ||
            DUM1 == NULL || DUM3 == NULL || PICOMP == NULL) {
        printf( "C Error: Can't allocate memory for z-specific Bz / Sz. Aborting... \n");
        mkl_free(B_INV);
        mkl_free(S_INV);
        mkl_free(GX_MZ);
        mkl_free(ZT_GT);
        mkl_free(DUM1);
        mkl_free(DUM3);
        mkl_free(DBINV_DSIGMA);
        mkl_free(DSINV_DSIGMA);
        mkl_free(DSINV_DTAU);
        mkl_free(PICOMP);
        exit(0);
    }
    else {
//        printf ("Successfully allocated memories for gradient calculation.\n");
    }


    sigma2 = sigma * sigma;
	sigma3 = sigma2 * sigma;
    tau2 = tau * tau;

    mu_grad = 0.0;
    sigma_grad = 0.0;
    tau_grad = 0.0;

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
        }

        for ( i = 0; i < (nsample*nsample); i++ ) {
            S_INV[i] = SZINV[ (unsigned long)z*nsample*nsample + i ];
            DSINV_DSIGMA[i] = 0.0;
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
        for ( i = 0; i < nz; i++ ) {
            zpos = ZARR[zindx + i];
            dlogdetS_dsigma   += sigma2   - B_INV[ zpos*nsnps+zpos ];
            for ( j = 0; j < nsnps; j++ ) {
                for ( k = 0; k < nsnps; k++ ) {
                    DBINV_DSIGMA  [ j*nsnps+k ] += 2.0 * B_INV[ j*nsnps+zpos ] * B_INV[ zpos*nsnps+k ] / sigma3;
                }
            }
        }
        dlogdetS_dsigma   = 2.0 * dlogdetS_dsigma   / sigma3;
        a_matT_matb_mat ( nsnps, nsample, (-tau*tau), GT, DBINV_DSIGMA,   DSINV_DSIGMA,   DUM3 );
        sigmacomp     = - 0.5 * ( dlogdetS_dsigma   + vecAT_smat_vecB( nsample, GX_MZ, DSINV_DSIGMA,   GX_MZ, DUM1 ) );
        sigma_grad   += ZCOMPS[z] * sigmacomp;

        dlogdetS_dtau = - mat_trace( nsample, S_INV ) / tau2;
        a_mat_matT ( nsample, nsample, (1 / tau2), S_INV, DSINV_DTAU );
        taucomp = - 0.5 * ( dlogdetS_dtau + vecAT_smat_vecB( nsample, GX_MZ, DSINV_DTAU, GX_MZ, DUM1 ) );
        tau_grad += ZCOMPS[z] * taucomp;

        zindx += nz;
    }

    GRAD[ nfeat + 0 ] = mu_grad;
    GRAD[ nfeat + 1 ] = sigma_grad;
    GRAD[ nfeat + 2 ] = 0; //sigmabg_grad;
    GRAD[ nfeat + 3 ] = tau_grad;

    mkl_free(B_INV);
    mkl_free(S_INV);
    mkl_free(GX_MZ);
    mkl_free(ZT_GT);
    mkl_free(DUM1);
    mkl_free(DUM3);
    mkl_free(PICOMP);
    mkl_free(DBINV_DSIGMA);
    mkl_free(DSINV_DSIGMA);
    mkl_free(DSINV_DTAU);

}       /* -----  end of function get_grad  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  get_zexp
 *  Description:  Calculate m_vz (eq. 3.29)
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
        FACT0[i] = GT_GX[i];
    }

    zindx = 0;
    for ( z = 0; z < zlen; z++ ) {

        nz = ZNORM[z];

        for ( i = 0; i < nsnps; i++ ) {
            FACTZ[i] = FACT0[i];
        }
        for ( i = 0; i < nz; i++ ) {
            zpos = ZARR[zindx + i];
            FACTZ[zpos] += (mu / sigma2);
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

}       /* -----  end of function get_zexp  ----- */


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

    debug = true;

    //if (zlen > 10000) {
    //    debug = true;
    //}

    if (debug) {
        printf ("%d zstates, %d SNPs and %d samples\n", zlen, nsnps, nsample);
        printf ("No. of features: %d\n", nfeat);
        printf ("Size of double: %lu bytes\n", sizeof( double ));
        printf ("Size of float:  %lu bytes\n", sizeof( float ));
        printf ("Size required:  %f Gb\n", (double)((unsigned long)zlen * nsnps   * nsnps * sizeof(double)) / (1024 * 1024 * 1024));
    }
    
    BZINV = (double *)mkl_malloc( (unsigned long)zlen * nsnps   * nsnps   * sizeof( double ), 64 );
    if (BZINV == NULL) {success = false; goto cleanup_main_BZINV;}

    SZINV = (double *)mkl_malloc( (unsigned long)zlen * nsample * nsample * sizeof( double ), 64 );
    if (SZINV == NULL) {success = false; goto cleanup_main_SZINV;}
    
    success = get_zcomps ( nsnps, nsample, zlen, PI, mu, sigma, tau, ZARR, ZNORM, GT, GX, ZCOMPS, BZINV, SZINV, logmL, debug );
    if (debug) {
        printf ( "ZCOMPS calculated.\n" );
    }

    if (success == false) {
        logmL[0] = 0.0;
        printf( "C error: Zcomps calculation did not succeed.\n" );
        goto cleanup_main;
    } 


        
    if (get_gradient) {
        get_grad ( nsnps, nsample, zlen, nfeat, PI, mu, sigma, tau, ZARR, ZNORM, GT, GX, FEAT, ZCOMPS, BZINV, SZINV, GRAD );
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

}       /* -----  end of function logmarglik  ----- */

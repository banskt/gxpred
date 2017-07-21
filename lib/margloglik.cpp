int main() {


    int info;
    char *lower = "L";
    char *upper = "U";

    int morder = 100;

    /* Cholesky factorization of a symmetric (Hermitian) positive-definite matrix */
    dpotrf( lower, &morder, C, &leadC, &info );
    if (info != 0) {
        cout << "c++ error: Cholesky failed" << endl;
    }


    /* Compute the inverse of a symmetric positive semi-definite matrix */
    dpotri( uplo, n, a, lda, info )
}

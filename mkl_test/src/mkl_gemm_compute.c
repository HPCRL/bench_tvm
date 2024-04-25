#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include "mkl_types.h"
#include "mkl_cblas.h"
#include <time.h>
#include <omp.h>

#ifndef IFLOAT
#define IFLOAT float
#endif
#define FLOAT float
#define GEMM cblas_sgemm

#define MAX_NUM 1.0

int main(int argc, char *argv[]) {
    int sizes[][3] = {
        {512, 64, 1024},    // Bert large BMATmul
        {512, 4096, 1024},  // Bert large MLP1
        {512, 1024, 4096},  // Bert large MLP2
        {512, 64, 768},     // Bert basic BMATmul
        {512, 3072, 768},   // Bert basic MLP1
        {512, 768, 3072}    // Bert basic MLP2
    };
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    long nT = mkl_get_max_threads();
    printf("Number of threads: %ld\n", nT);
    mkl_set_num_threads(nT);

    IFLOAT *a, *b;
    FLOAT *c;
    FLOAT alpha = 2.0;
    FLOAT beta = 0.0;
    int m, n, k, lda, ldb, ldc;
    
    int loops = 100; 

    for (int index = 0; index < numSizes; index++) {
        m = sizes[index][0];
        n = sizes[index][1];
        k = sizes[index][2];

        lda = k;  // Since matrix A is not transposed
        ldb = n;  // Since matrix B is not transposed
        ldc = n;  // Standard leading dimension for matrix C

        a = (IFLOAT *)malloc(sizeof(IFLOAT) * m * k);
        b = (IFLOAT *)malloc(sizeof(IFLOAT) * k * n);
        c = (FLOAT *)malloc(sizeof(FLOAT) * m * n);

        if (a == NULL || b == NULL || c == NULL) {
            fprintf(stderr, "Out of Memory!!\n");
            exit(1);
        }

        for (int i = 0; i < m * k; i++) a[i] = (IFLOAT)rand() / (IFLOAT)RAND_MAX;
        for (int i = 0; i < k * n; i++) b[i] = (IFLOAT)rand() / (IFLOAT)RAND_MAX;
        for (int i = 0; i < m * n; i++) c[i] = 0;

        double start, end, time1, timeg;
        start = omp_get_wtime();
        for (int ite = 0; ite < loops; ite++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        end = omp_get_wtime();

        time1 = end - start;
        timeg = time1 / loops;


        double gflops = (2.0 * m * n * k) / (timeg * 1e9); // 2*m*n*k because each multiplication and addition counts as one operation
        

        printf("Matmul testcase no, %d, gflpos, %f, time, %f\n", index, gflops, timeg);

        free(a);
        free(b);
        free(c);
    }

    return 0;
}

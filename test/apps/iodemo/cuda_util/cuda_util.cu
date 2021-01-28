// int main() { return 1; }

#include <stdio.h>
#include "cuda_util.h"


__global__
void LEO_add(int *x)
{
    printf("LEO %d\n", *x);
    *x += 1;
    printf("LEO %d\n", *x);
    // x += y;
}

void LEO_add2(int *x)
    {
        LEO_add<<<1, 1>>>(x);
        cudaDeviceSynchronize();
    }

#include <iostream>
#include <random>
#include <ctime>
#include <cmath>
#define N 5
__global__ void softmax(float *x , float *out,int col){
    int row_start = threadIdx.x * N;
    int end = row_start + col; 
    float sum = 0.0;
    for(int i=row_start;i<end;i++){
        sum += exp(x[i]);
    }
    for(int i=row_start;i<end;i++){
        out[i] = exp(x[i]) / sum;
    }
}


int main(){
    std::default_random_engine e;
    std::normal_distribution<float> u(0,1); // 均值为0，标准差为1
    e.seed(time(0));

    float X[N][N],O[N][N];
    float *x,*out;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            X[i][j] = u(e);
            printf("%f ",X[i][j]);
        }
        printf("\n");
    }
    int bytes_size = sizeof(float)*N*N;
    cudaMalloc(&x,bytes_size);
    cudaMalloc(&out,bytes_size);
    cudaMemcpy(x,X,bytes_size,cudaMemcpyHostToDevice);

    softmax<<<1,N>>>(x,out,N);

    cudaMemcpy(O,out,bytes_size,cudaMemcpyDeviceToHost);
    printf("softmax....\n");
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            printf("%f ",O[i][j]);
        }
        printf("\n");
    }
    cudaFree(out);
    return 0;
}
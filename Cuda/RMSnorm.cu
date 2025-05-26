#include <iostream>
#include <random>
#include <ctime>
#include <cmath>
#define N 5

__global__ void RMSNorm(float*x,float* out){
    int row_start = threadIdx.x * N;
    int end = row_start + N; 
    float eps = 1e-6;
    float rms_x = 0.0;
    for(int i=row_start;i<end;i++){
        rms_x += x[i]*x[i];
    }
    rms_x = sqrt(rms_x / N);
    for(int i=row_start;i<end;i++){
        out[i] = x[i] / (rms_x + eps);
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

    RMSNorm<<<1, N>>>(x,out);
    cudaMemcpy(O,out,bytes_size,cudaMemcpyDeviceToHost);
    printf("RMSNorm....\n");
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            printf("%f ",O[i][j]);
        }
        printf("\n");
    }
    cudaFree(out);


    return 0;
}
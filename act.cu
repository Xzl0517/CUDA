#include <iostream>
#include <random>
#include <ctime>
#include <cmath>
#define N 5
__global__ void Leaky_Relu(float *x, float alpha){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("id:%d\n", id);
    printf("threadidx:%d\n", threadIdx.x);
    if(x[id] < 0){
        x[id] *= alpha;
    }
}

__global__ void Relu(float *x){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(x[id]<0) x[id] = 0;
}

__global__ void Sigmoid(float *x){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    x[id] = 1.0 / (1.0 + exp(x[id]));
}

__global__ void Tanh(float *x){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    x[id] = (exp(x[id]) - exp(-x[id])) / (exp(x[id]) + exp(-x[id]));
}

__global__ void Gelu(float *x){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float tanh_x = (exp(x[id]) - exp(-x[id])) / (exp(x[id]) + exp(-x[id]));
    float x_p = exp(tanh_x);
    float x_n = exp(-tanh_x);
    x[id] = 0.5 * x[id] *(1.0 + (x_p - x_n)/(x_p + x_n));
}

__global__ void Silu(float *x){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    x[id] = x[id] / (1.0 + exp(x[id]));
}


int main(){
    std::default_random_engine e;
    std::normal_distribution<float> u(0,1); // 均值为0，标准差为1
    e.seed(time(0));
    float A[N][N],B[N][N];
    float *A_cuda;

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            A[i][j]=u(e);
            printf("%f ",A[i][j]);
        }
        printf("\n");
    }
    cudaMalloc((void **)&A_cuda,sizeof(float)*N*N);
    cudaMemcpy(A_cuda,A,sizeof(float)*N*N,cudaMemcpyHostToDevice);

    //Leaky_Relu<<<N,N>>>(A_cuda, 0.01);
    //Relu<<<N,N>>>(A_cuda);
    //Sigmoid<<<N,N>>>(A_cuda);
    //Tanh<<<N,N>>>(A_cuda);
    Gelu<<<N,N>>>(A_cuda);

    cudaMemcpy(B,A_cuda,sizeof(float)*N*N,cudaMemcpyDeviceToHost);

    printf("act......\n");
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            printf("%f ",B[i][j]);
        }
        printf("\n");
    }

    cudaFree(A_cuda);
    return 0;
}


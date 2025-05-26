#include <iostream>
#define  N 20
//1 个block 1个thread
__global__ void vector_add_1(int *a,int *b,int *c,int n){
    int id = 0;
    while(id<n){
        c[id] = a[id] + b[id];
        id+=1;
    }
}
// 1个block 多个thread
__global__ void vector_add_2(int *a,int *b,int *c,int n){
    int tid = threadIdx.x;
    int offs = blockDim.x;
    while(tid<n){
        c[tid] = a[tid] + b[tid];
        tid+=offs;
    }
}
// 多个block 多个thread
__global__ void vector_add_3(int *a,int *b,int *c,int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offs = gridDim.x * blockDim.x;
    while(tid<n){
        c[tid] = a[tid] + b[tid];
        tid += offs;
    }
}
int main(){
    int A[N],B[N],C[N];
    int *a_cuda,*b_cuda,*c_cuda;

    for(int i=0;i<N;i++){
        A[i]=i;
        B[i]=i;
    }

    cudaMalloc(&a_cuda,sizeof(int) * N);
    cudaMemcpy(a_cuda, A, sizeof(int) * N , cudaMemcpyHostToDevice);
    
    cudaMalloc(&b_cuda,sizeof(int) * N);
    cudaMemcpy(b_cuda, B, sizeof(int) * N , cudaMemcpyHostToDevice);

    cudaMalloc(&c_cuda,sizeof(int) * N);
    cudaMemcpy(c_cuda, C, sizeof(int) * N , cudaMemcpyHostToDevice);

    vector_add_1<<<1,1>>>(a_cuda,b_cuda,c_cuda,N);
    cudaMemcpy(C, c_cuda, sizeof(int) * N , cudaMemcpyDeviceToHost);
    for (int i=0;i<N;i++){
        printf("%d ",C[i]);
    }
    printf("\n");
    vector_add_2<<<1,5>>>(a_cuda,b_cuda,c_cuda,N);
    cudaMemcpy(C, c_cuda, sizeof(int) * N , cudaMemcpyDeviceToHost);
    for (int i=0;i<N;i++){
        printf("%d ",C[i]);
    }
    printf("\n");
    vector_add_3<<<2,5>>>(a_cuda,b_cuda,c_cuda,N);
    cudaMemcpy(C, c_cuda, sizeof(int) * N , cudaMemcpyDeviceToHost);
    for (int i=0;i<N;i++){
        printf("%d ",C[i]);
    }
    printf("\n");
    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(c_cuda);
    return 0;
}
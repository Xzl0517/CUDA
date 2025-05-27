#include <iostream>
#define N 10
__global__ void matmul(int* a,int* b,int* c,int stride_AK){
    int row_id = blockIdx.x ;
    int col_id = threadIdx.x;
    int sum = 0;
    for (int i=0;i<stride_AK;i++){
        sum += a[row_id * blockDim.x + i] * b[i * blockDim.x + col_id];     
    }
    c[row_id * blockDim.x + col_id] = sum;

}


int main(){
    int A[N][N],B[N][N],C[N][N];
    int *a_cuda,*b_cuda,*c_cuda;

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            A[i][j] = i;
            B[i][j] = j;
        }
    }
    
    int bytes_size = sizeof(int)*N*N;
    cudaMalloc(&a_cuda,bytes_size);
    cudaMalloc(&b_cuda,bytes_size);
    cudaMalloc(&c_cuda,bytes_size);

    cudaMemcpy(a_cuda,A,bytes_size,cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda,B,bytes_size,cudaMemcpyHostToDevice);


    matmul<<<N,N>>>(a_cuda,b_cuda,c_cuda, N);
    
    cudaMemcpy(C,c_cuda,bytes_size,cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            printf("%d ",C[i][j]);
        }
        printf("\n");
    }

    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(c_cuda);
    return 0;
}
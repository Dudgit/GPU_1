// Include Cuda libraries, 'cause I use Visual Studio
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//Refference functions
#include "vector_mx_mul.h"
#include "cpp_functions.h"

//Constant values


__global__ void mx_vec_gpu(float* result_mx, float* input_vector, float* input_mx, int N)
{

    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float Atmp[MBS * MBS];
    __shared__ float Btmp[MBS*MBS];


    float sum = 0.f;

    for (int K = 0; K < N / MBS; ++K)
    {
        Atmp[threadIdx.y * MBS + threadIdx.x] = input_mx[y * N + (K * MBS + threadIdx.x)];
        Btmp[threadIdx.y * MBS+threadIdx.x] = input_vector[(K * MBS + threadIdx.y)];
        
        __syncthreads();
        for (int k = 0; k < MBS; ++k)
        {
            sum += Atmp[threadIdx.y * MBS + k] * Btmp[k * MBS + threadIdx.x];
        }
        __syncthreads();
    }
    result_mx[y] = sum;
}

float do_Cuda(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C0, std::vector<float>& C1, std::vector<float>& C2)
{

    float* pA = nullptr;
    float* pB = nullptr;
    float* pC2 = nullptr;

    cudaEvent_t evt[2];
    for (auto& e : evt) { cudaEventCreate(&e); }

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&pA, N * sizeof(float));
    if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaMalloc((void**)&pB, N * N * sizeof(float));
    if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaMalloc((void**)&pC2, N * sizeof(float));
    if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaMemcpy(pA, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaMemcpy(pB, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

    {
        dim3 dimGrid(n_blocks, n_blocks);
        dim3 dimBlock(block_sz, block_sz);
        cudaEventRecord(evt[0]);
        mx_vec_gpu << <dimGrid, dimBlock >> > (pC2, pA, pB, N);
        err = cudaGetLastError();
        if (err != cudaSuccess) { std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
        cudaEventRecord(evt[1]);
    }
    err = cudaMemcpy(C2.data(), pC2, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree(pA);
    if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree(pB);
    if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree(pC2);
    if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    cudaEventSynchronize(evt[1]);
    float dt = 0.0f;//milliseconds
    cudaEventElapsedTime(&dt, evt[0], evt[1]);
    for (auto& e : evt) { cudaEventDestroy(e); }
    return dt;
}

int main()
{
    std::vector<float> A(N);
    std::vector<float> B(N * N);
    std::vector<float> C0(N);
    std::vector<float> C1(N);
    std::vector<float> C2(N);
    fill_values(A, B, C0, C1, C2);
    float cuda_run_time = do_Cuda(A, B, C0, C1, C2);

    auto t0 = std::chrono::high_resolution_clock::now();
    vector_mx_naive(C0, A, B, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    vector_mx_alg(C1, A, B, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    checker(C0, C1, "C0 vs C1","CPU Naive","CPU improved");
    checker(C0, C2, "C0 vs C2", "CPU Naive", "GPU Naive");
  

    std::cout << "CPU naive    Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f << " ms\n";
    std::cout << "CPU improved Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f << " ms\n";
    std::cout << "GPU improved Computation took: " << cuda_run_time << " ms.\n";
 
    return 0;
}

#include "run_cuda.cuh"

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
    std::cout << "CPU Algorithm Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f << " ms\n";
    std::cout << "GPU improved Computation took: " << cuda_run_time << " ms.\n";
 
    return 0;
}

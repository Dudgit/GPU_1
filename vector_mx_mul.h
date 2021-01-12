#include <vector>
#include <numeric>

void vector_mx_naive(std::vector<float>& C, std::vector<float> const& vec, std::vector<float> const& mx, int N)
{
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            C[y] += mx[y * N + x] * vec[x];
        }
    }
}

void vector_mx_alg(std::vector<float>& C, std::vector<float>& vec, std::vector<float>& mx, int N)
{
    /*
    A is the Matrix
    B is the Vectr
    */
    for (int i = 0; i < N; ++i)
    {
        C[i] = std::inner_product(mx.begin() + (i * N), mx.begin() + ((i + 1) * N), vec.begin(), .0f);
    }
}
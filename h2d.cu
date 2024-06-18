#include <cuda_runtime.h>
#include <numa.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include <string>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error in " << __FILE__ << " at line " <<  \
                      __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(err);                                                   \
        }                                                                \
    } while (0)

void measureBandwidth(size_t size, int iterations, int warmup, int gpuDevice, int numaNode, cudaMemcpyKind kind) {
    int *d_data;
    int *h_data = (int *)numa_alloc_onnode(size, numaNode); 

    if (h_data == nullptr) {
        std::cerr << "Memory allocation failed on NUMA node " << numaNode << " for host memory.\n";
        exit(1);
    }

    CUDA_CHECK(cudaSetDevice(gpuDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_data, size));

    cudaStream_t stream;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(kind == cudaMemcpyHostToDevice ? d_data : h_data, 
                                   kind == cudaMemcpyHostToDevice ? h_data : d_data, 
                                   size, kind, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<float> times(iterations);
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        CUDA_CHECK(cudaMemcpyAsync(kind == cudaMemcpyHostToDevice ? d_data : h_data, 
                                   kind == cudaMemcpyHostToDevice ? h_data : d_data, 
                                   size, kind, stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        times[i] = milliseconds / 1000.0f; // Convert to seconds
    }

    float avgTime = std::accumulate(times.begin(), times.end(), 0.0f) / iterations;
    float bandwidth = size / avgTime / 1e9; // GB/s

    std::string src = (kind == cudaMemcpyHostToDevice) ? "M" + std::to_string(numaNode) : "G" + std::to_string(gpuDevice);
    std::string dst = (kind == cudaMemcpyDeviceToHost) ? "M" + std::to_string(numaNode) : "G" + std::to_string(gpuDevice);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << src << " -> " << dst << ":\n";
    std::cout << "  Avg. Time: " << avgTime << " s\n";
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n\n";

    CUDA_CHECK(cudaFree(d_data));
    numa_free(h_data, size);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    const int numGPUs = 4;
    const int numNUMANodes = 2;
    size_t size = 1024 * 1024; // 1 GB
    int iterations = 10;
    int warmup = 5;

    for (int gpu = 0; gpu < numGPUs; ++gpu) {
        for (int numa = 0; numa < numNUMANodes; ++numa) {
            measureBandwidth(size, iterations, warmup, gpu, numa, cudaMemcpyHostToDevice); 
            measureBandwidth(size, iterations, warmup, gpu, numa, cudaMemcpyDeviceToHost); 
        }
    }

    return 0;
}

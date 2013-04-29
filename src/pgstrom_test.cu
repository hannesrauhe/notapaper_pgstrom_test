/**
 * Copyright 2013 Hannes Rauhe
 */
#include "CUDAHelper.hpp"
#include "pgstrom_test.h"

__global__ void filter(bool* res, float* x, float* y, size_t t_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<t_size) {
        res[idx] = (sqrt( pow((x[idx]-25.6),2) + pow((y[idx]-12.8),2)) < 15);
    }
}

__global__ void filter(int* res, float* x, float* y, size_t t_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<t_size) {
        res[idx] = (sqrt( pow((x[idx]-25.6),2) + pow((y[idx]-12.8),2)) < 15);
    }
}

double test_gpu_naive(const float* x, const float* y, const size_t t_size, int& res) {
//    CUDA_SAFE_CALL( cudaSetDevice(1) );
    float* dev_x;
    float* dev_y;
    bool* dev_result_v;
    bool* result_v;
    

//    int p_size = 1;
    int nblocks = t_size / NUMBER_OF_THREADS;

    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_x, sizeof(float)*t_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_y, sizeof(float)*t_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_result_v, sizeof(bool)*t_size) );
    CUDA_SAFE_CALL( cudaMallocHost( (void**)&result_v, sizeof(int)*t_size) );

//#ifdef _DEBUG
    CUDA_SAFE_CALL( cudaMemset( dev_result_v, 0, sizeof(bool)*t_size) );
//#endif
    CUDA_SAFE_CALL( cudaMemcpy( 		dev_x,			x,			t_size*sizeof(float),			cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy( 		dev_y,			y,			t_size*sizeof(float),			cudaMemcpyHostToDevice) );

    double t_start = omp_get_wtime();

    filter<<<nblocks,NUMBER_OF_THREADS>>>(dev_result_v, dev_x, dev_y, t_size);
    CUDA_SAFE_CALL( cudaMemcpy(result_v, dev_result_v, t_size*sizeof(bool), cudaMemcpyDeviceToHost) );

    CUDA_SAFE_CALL( cudaThreadSynchronize());

    double elapsedKernelTime = (omp_get_wtime()-t_start);

    for(int idx=0; idx<t_size; idx++) {
        res += result_v[idx];
    }

    CUDA_SAFE_CALL( cudaFree( dev_x) );
    CUDA_SAFE_CALL( cudaFree( dev_y) );
    CUDA_SAFE_CALL( cudaFree( dev_result_v) );
    CUDA_SAFE_CALL( cudaFreeHost( result_v) );
    
    return elapsedKernelTime;
}

double test_gpu_stream(const float* x, const float* y, const size_t t_size, int& res) {
//    CUDA_SAFE_CALL( cudaSetDevice(1) );
    float* dev_src_x[3];
    float* dev_src_y[3];
    int* dev_dest[3];
    float* src_x[3];
    float* src_y[3];
    int* result_v;

    CUDA_SAFE_CALL( cudaMallocHost( (void**)&result_v, sizeof(int)*t_size) );

    int buf_size = STREAM_BUF_SIZE;
    if(buf_size>t_size) {
        buf_size=t_size;
    }
    const int nblocks = buf_size / NUMBER_OF_THREADS;

    cudaStream_t streams[3];
    CUDA_SAFE_CALL( cudaMallocHost( (void**)&src_x[0], sizeof(float)*buf_size) );
    CUDA_SAFE_CALL( cudaMallocHost( (void**)&src_x[1], sizeof(float)*buf_size) );
    CUDA_SAFE_CALL( cudaMallocHost( (void**)&src_x[2], sizeof(float)*buf_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_src_x[0],sizeof(float)*buf_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_src_x[1],sizeof(float)*buf_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_src_x[2],sizeof(float)*buf_size) );
    CUDA_SAFE_CALL( cudaMallocHost( (void**)&src_y[0], sizeof(float)*buf_size) );
    CUDA_SAFE_CALL( cudaMallocHost( (void**)&src_y[1], sizeof(float)*buf_size) );
    CUDA_SAFE_CALL( cudaMallocHost( (void**)&src_y[2], sizeof(float)*buf_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_src_y[0],sizeof(float)*buf_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_src_y[1],sizeof(float)*buf_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_src_y[2],sizeof(float)*buf_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_dest[0],sizeof(int)*buf_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_dest[1],sizeof(int)*buf_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_dest[2],sizeof(int)*buf_size) );
    CUDA_SAFE_CALL( cudaStreamCreate(&streams[0]) );
    CUDA_SAFE_CALL( cudaStreamCreate(&streams[1]) );
    CUDA_SAFE_CALL( cudaStreamCreate(&streams[2]) );


    double tk_start = time_in_seconds();
    int str_runs = t_size/buf_size+2;
    for( int buf_pos=0; buf_pos<str_runs; buf_pos++)
    {
        if(buf_pos<str_runs-1 && buf_pos>0) {
            CUDA_SAFE_CALL( cudaMemcpyAsync(
                    dev_src_x[(buf_pos)%3],
                    src_x[(buf_pos-1)%3],
                    buf_size*sizeof(float),
                    cudaMemcpyHostToDevice,
                    streams[0] ) );
            CUDA_SAFE_CALL( cudaMemcpyAsync(
                    dev_src_y[(buf_pos)%3],
                    src_y[(buf_pos-1)%3],
                    buf_size*sizeof(float),
                    cudaMemcpyHostToDevice,
                    streams[0] ) );
        }
        if(buf_pos>2)
            CUDA_SAFE_CALL( cudaMemcpyAsync(
                    result_v + ((buf_pos-3)*buf_size),
                    dev_dest[(buf_pos-2)%3],
                    buf_size*sizeof(int),
                    cudaMemcpyDeviceToHost,
                    streams[1] ) );

        if(buf_pos>1 && buf_pos<str_runs-1)
            filter<<<nblocks,NUMBER_OF_THREADS,0,streams[2]>>>(
                    dev_dest[(buf_pos-1)%3],
                    dev_src_x[(buf_pos-1)%3],
                    dev_src_y[(buf_pos-1)%3],
                    buf_size);

        if(buf_pos<str_runs-2) {
                memcpy(src_x[buf_pos%3],x+(buf_pos*buf_size),buf_size*sizeof(int));
                memcpy(src_y[buf_pos%3],y+(buf_pos*buf_size),buf_size*sizeof(int));
        }

        CUDA_SAFE_CALL( cudaThreadSynchronize());
    }

    double elapsedKernelTime = (time_in_seconds()-tk_start);

    /* stoped the clock before the CPU work because this could be implemented
     * in a way, so that it overlaps the streaming - it's fair to assume that
     * almost no time is needed for this*/

//    #pragma omp parallel for reduction(+:res)
    for(int idx=0; idx<t_size; idx++) {
        res += result_v[idx];
    }


    CUDA_SAFE_CALL( cudaFreeHost( src_x[0]) );
    CUDA_SAFE_CALL( cudaFreeHost( src_x[1]) );
    CUDA_SAFE_CALL( cudaFreeHost( src_x[2]) );
    CUDA_SAFE_CALL( cudaFreeHost( src_y[0]) );
    CUDA_SAFE_CALL( cudaFreeHost( src_y[1]) );
    CUDA_SAFE_CALL( cudaFreeHost( src_y[2]) );
    CUDA_SAFE_CALL( cudaFreeHost( result_v) );
    CUDA_SAFE_CALL( cudaFree( dev_src_x[0]) );
    CUDA_SAFE_CALL( cudaFree( dev_src_x[1]) );
    CUDA_SAFE_CALL( cudaFree( dev_src_x[2]) );
    CUDA_SAFE_CALL( cudaFree( dev_src_y[0]) );
    CUDA_SAFE_CALL( cudaFree( dev_src_y[1]) );
    CUDA_SAFE_CALL( cudaFree( dev_src_y[2]) );
    CUDA_SAFE_CALL( cudaFree( dev_dest[0]) );
    CUDA_SAFE_CALL( cudaFree( dev_dest[1]) );
    CUDA_SAFE_CALL( cudaFree( dev_dest[2]) );
    CUDA_SAFE_CALL( cudaStreamDestroy(streams[0]) );
    CUDA_SAFE_CALL( cudaStreamDestroy(streams[1]) );
    CUDA_SAFE_CALL( cudaStreamDestroy(streams[2]) );

    return elapsedKernelTime;
}





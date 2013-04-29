/**
 * Copyright 2013 Hannes Rauhe
 */
#include <omp.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

static const int STREAM_BUF_SIZE = 64*1024; //in number of integers
#define NUMBER_OF_THREADS 256


static double time_in_seconds (void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

double test_gpu_naive(const float* x, const float* y, const size_t t_size, int& res);
double test_gpu_stream(const float* x, const float* y, const size_t t_size, int& res);

/**
 * Copyright 2013 Hannes Rauhe
 */

#include "pgstrom_test.h"

#if __GNUC_MINOR__>4
#include <random>
#endif

#define NUM_RECORDS 10*1000*1000
#define NUMBER_OF_RUNS 5
//#define NUMBER_OF_WARMUP_RUNS 1

int main(int argc, char* argv[]) {
    size_t t_size=NUM_RECORDS;
    if(argc>1) {
        t_size = atoi(argv[1]);
    }
    std::vector<float> x(t_size);
    std::vector<float> y(t_size);

    printf("Size is %.3f MB\n",static_cast<float>(2*t_size*sizeof(float))/1024/1024);

#if __GNUC_MINOR__ > 4
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0,1000);

	#pragma omp parallel for
    for(int i=0; i<(int)t_size; i++) {
        x[i] = distribution(generator);
        y[i] = distribution(generator);
    }
#else
	srand(time(NULL));

	#pragma omp parallel for
    for(int i=0; i<(int)t_size; i++) {
        x[i] = 1000.0*(float)rand();
        y[i] = 1000.0*(float)rand();
    }
#endif


	double cpu_t=0,cpu_pt=0,gpu_nt=0,gpu_st=0,gpu_nt2=0,gpu_st2=0;
	int result,result_p,res_gn,res_gs;
	for(int run=0;run<NUMBER_OF_RUNS;++run) {
	    result=0;result_p=0;res_gn=0;res_gs=0;
        double t1 = time_in_seconds();
        for(int i=0; i<(int)t_size; i++) {
            result += (sqrt( pow((x[i]-25.6),2) + pow((y[i]-12.8),2)) < 15);
        }
        cpu_t = time_in_seconds()-t1;

        double t2 = time_in_seconds();
    #pragma omp parallel for reduction(+:result_p)
        for(int i=0; i<(int)t_size; i++) {
            result_p += (sqrt( pow((x[i]-25.6),2) + pow((y[i]-12.8),2)) < 15);
        }
        cpu_pt = time_in_seconds()-t2;


        double t3 = time_in_seconds();
        gpu_nt = test_gpu_naive(x.data(),y.data(),t_size,res_gn);
        gpu_nt2 = time_in_seconds()-t3;


        double t4 = time_in_seconds();
        gpu_st = test_gpu_stream(x.data(),y.data(),t_size,res_gs);
        gpu_st2 = time_in_seconds()-t4;
	}

	printf("Result: %d; CPU %.3f sec\n",result,cpu_t);
	printf("Result: %d; p CPU %.3f sec\n",result_p,cpu_pt);
    printf("Result: %d; GPU naive %.3f (%.3f) sec\n",res_gn,gpu_nt,gpu_nt2);
    printf("Result: %d; GPU stream %.3f (%.3f) sec\n",res_gs,gpu_st,gpu_st2);

	return 0;
}

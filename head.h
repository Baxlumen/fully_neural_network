#ifndef SETTING_H_
#define SETTING_H_

#include <bits/stdc++.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<curand.h>
#include <cufft.h>
#include <cuda_fp16.h>
#define IPNNUM 784
#define HDNNUM 128
#define OPNNUM 10

using namespace std;
__global__ void activate(double *d_A,int b,int a);
__global__ void Vector2_Multiply_By_Elements (const double* a, const double* b, int n, double* out);
__global__ void Vector1_Multiply_By_Elements (const double* a, double* b, int n);
void printTensor(float *d_des,long m,long n,long l);
void for_cuda(double *input,double *W1,double *outh,double *W2,double *outo,int in,int hid,int out,cublasHandle_t handle);
void printTensor(double *d_des,long m,long n,long l);
void back_cuda(double *Y,double *Y_hat,double *outh,double *W2,double *input,double *W1,int in,int hid,int out,double rate,cublasHandle_t handle);
double loss_gpu(double *A,double *B,int n,cublasHandle_t handle);
#endif /* SETTING_H_ */
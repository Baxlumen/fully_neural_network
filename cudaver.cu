#include "head.h"

__global__ void Vector2_Multiply_By_Elements (const double* a, const double* b, int n, double* out){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<n)
	{
		out[tid]=a[tid]*(1.0-a[tid])*(b[tid]-a[tid]);
		tid+=temp;
	}
	__syncthreads();
}
__global__ void Vector1_Multiply_By_Elements (const double* a, double* b, int n){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<n)
	{
		b[tid]=a[tid]*(1.0-a[tid])*b[tid];
		tid+=temp;
	}
	__syncthreads();
}

__global__ void activate(double *d_A,int b,int a)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<a)
	{
		d_A[tid] = 1/(1+exp(-d_A[tid]+b));
		tid+=temp;
	}
	__syncthreads();
}
void printTensor(double *d_des,long m,long n,long l){
	double *des = new double[m*n*l]();
	cudaMemcpy(des,d_des,sizeof(double)*m*n*l,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for(long k = 0;k<l;k++){
		for(long i = 0;i<n;i++){
			for(long j = 0;j<m;j++){
				cout<<des[k*m*n+i*m+j]<<" ";
			}
			cout<<endl;
		}
		cout<<"~~~~~~~~~~~~~~~~"<<endl;
	}
	delete[] des;des=nullptr;
}

void for_cuda(double *input,double *W1,double *outh,double *W2,double *outo,int in,int hid,int out,cublasHandle_t handle)
{
	//hid 行 in列 W1 ,out行，hid列 W2

	//printTensor(input,5,1,1);
	//cout<<"weigh matrix is :"<<endl;printTensor(W1,4,4,1);

	//double *outo;
	//cudaMalloc((void**)&outo,sizeof(double)*out);
	int b= (rand() % 100) / (double)100; //偏置值
	double alpha=1.0, beta=0.0;


	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,hid,1,in,&alpha,W1,hid,input,in,&beta,outh,hid);

	//激活函数
	activate<<<1,1024>>>(outh,b,hid);
	//printTensor(outh,3,3,1);

	cudaDeviceSynchronize();

	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,out,1,hid,&alpha,W2,out,outh,hid,&beta,outo,out);	
	activate<<<1,1024>>>(outo,b,out);
	cudaDeviceSynchronize();
	//cout<<"model output:"<<endl;printTensor(outo,2,2,1);
}

void back_cuda(double *Y,double *Y_hat,double *outh,double *W2,double *input,double *W1,int in,int hid,int out,double rate,cublasHandle_t handle)
{
	//cout<<"yu ce value:"<<endl;printTensor(Y_hat,10,1,1);
	//cout<<"bp zhong de W1"<<endl;printTensor(W1,4,4,1);
	double *d_thta3,*d_thta2;
	cudaMalloc((void**)&d_thta3,sizeof(double)*out);
	cudaMalloc((void**)&d_thta2,sizeof(double)*hid);
	Vector2_Multiply_By_Elements<<<1,512>>>(Y_hat, Y, out, d_thta3);
	double alpha=1.0, beta=0.0;
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,hid,1,out,&alpha,W2,out,d_thta3,out,&beta,d_thta2,hid);
	Vector1_Multiply_By_Elements<<<1,512>>>(outh, d_thta2, hid);

	alpha=rate; beta=1.0;
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,hid,in,1,&alpha,d_thta2,hid,input,1,&beta,W1,hid);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,out,hid,1,&alpha,d_thta3,out,outh,1,&beta,W2,out);

	//printTensor(W1,4,4,1);

	cudaFree(d_thta2);
	cudaFree(d_thta3);
}	
double loss_gpu(double *A,double *B,int n,cublasHandle_t handle)
{	
	//A 实际值  B预测值
	//printTensor(A,2,2,1);
	//printTensor(B,2,2,1);
	double alpha1 = -1.0,loss;
	double *tmp;
	cudaMalloc((void**)&tmp,sizeof(double)*n);

	cublasDcopy(handle,n,B,1,tmp,1);	
	cublasDaxpy(handle,n,&alpha1,A,1,tmp,1);
	cublasDnrm2(handle,n,tmp,1,&loss); 
	cudaDeviceSynchronize();
	cudaFree(tmp);
	return loss;
}


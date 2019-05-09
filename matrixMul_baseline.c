/*************************************************************************
	> File Name: matrixMul_baseline.c
	> Author: logos
	> Mail: 838341114@qq.com 
	> Created Time: 2019年04月18日 星期四 14时22分44秒
 ************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>


float rand_float(float s){
		return 4*s*(1-s);
}

void matrix_gen(float *a,float *b,int N,float seed){
	float s=seed;
	for(int i=0;i<N*N;i++){
		s=rand_float(s);
		a[i]=s;
		s=rand_float(s);
		b[i]=s;
	}
}

float cal_trace(float *a,int N){
	float result=0;
	for(int i=0;i<N;i++){
		result+=a[i*(N+1)];
	}
}

void print_matrix(float *a, int N){
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			printf("%f ",a[i*N+j]);
		}
		printf("\n");
	}
	printf("\n");
}

/*
 * inputs: N, seed
*/

int main(int argc, char *argv[]){

	//parameter initiation
	const int N= atoi(argv[1]); //the size of matrix
	float seed=atof(argv[2]);

	//matrix init
	float *a;
	float *b;
	float *c;

	a=(float*)malloc(N*N*sizeof(float));
	b=(float*)malloc(N*N*sizeof(float));
	c=(float*)malloc(N*N*sizeof(float));

	//matrix generation
	matrix_gen(a,b,N,seed);

	//run time calculation
	struct timeval start;
	struct timeval end;

	gettimeofday(&start,NULL);

	//matrix c calculation

	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			int temp=i*N;
			int index=temp+j;
			c[index]=0;
			for(int k=0;k<N;k++){
				c[index]+=a[temp+k]*b[k*N+j];
			}
		}

	}

	gettimeofday(&end,NULL);

	float duration=end.tv_sec-start.tv_sec;
	
	float trace=0;
	//trace calculation
	trace=cal_trace(c,N);
	
	//print_matrix(a,N);
	//print_matrix(b,N);
	print_matrix(c,N);
	printf("%f %f",trace,duration);
		
}

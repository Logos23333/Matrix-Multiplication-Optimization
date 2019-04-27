/*************************************************************************
	> File Name: multiThreads.c
	> Author: logos
	> Mail: 838341114@qq.com 
	> Created Time: 2019年04月18日 Friday 13时52分44秒
 ************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<pthread.h>

#define NUM_THREADS 16
#define M 8

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

void print_matrix(float *a,int N){
	for(int i=0;i<N;i++){
		if(i%M==0)
			printf("\n");
		for(int j=0;j<N;j++){
			if(j%M==0)
				printf(" ");
			printf("%f ",a[i*N+j]);
		}
		printf("\n");
	}
}

float cal_trace(float *a,int N){
	float result=0;
	for(int i=0;i<N;i++){
		result+=a[i*(N+1)];
	}
	return result;
}

struct parameter{
	int N;
	float *a;
	float *b;
	float *c;
	int number;
};

void* matrix_mul(void *arg){
	struct parameter *p;
	p=(struct parameter *) arg;

	int N=p->N;
	float *a=p->a;
	float *b=p->b;
	float *c=p->c;
	int number=p->number;

	int num=N/NUM_THREADS;
	int i_start=number*num;
	int i_finish=i_start+num;

	for(int i=i_start;i<i_finish;i++){
		int temp=i*N; //optimization
		for(int j=0;j<N;j++){
			int index=temp+j; //optimization
			c[index]=0;
			for(int k=0;k<N;k++){
				c[index]+=a[temp+k]*b[k*N+j];
				
			}
		}
	}
	
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

	//init thread parameters
	pthread_t threads[NUM_THREADS];
	struct parameter parameters[NUM_THREADS];

	gettimeofday(&start,NULL);

	//matrix c calculation
	//parallel optimization
	for(int cnt=0;cnt<NUM_THREADS;cnt++){


		//parameters set value
		parameters[cnt].N=N;
		parameters[cnt].a=a;
		parameters[cnt].b=b;
		parameters[cnt].c=c;
		parameters[cnt].number=cnt;
		
		pthread_create(&threads[cnt],NULL,matrix_mul,&parameters[cnt]); //create thread
	}
	
	for(int cnt=0;cnt<NUM_THREADS;cnt++){
		pthread_join(threads[cnt],NULL);
	}
	
	gettimeofday(&end,NULL);

	float duration=end.tv_sec-start.tv_sec;

	
	float trace=0;
	//trace calculation
	trace=cal_trace(c,N);

	printf("%f %f\n",trace,duration);

	/*
	 * code for test
	 */

	//print_matrix(a,N);
	//print_matrix(b,N);
	//print_matrix(c,N);
	
}

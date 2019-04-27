/*************************************************************************
  > File Name: parallel_cache_SSE.c
	> Author: logos
	> Mail: 838341114@qq.com 
	> Created Time: 2019年04月23日 Tuesday 18时01分44秒
 ************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<pthread.h>
#include<string.h>
#include<xmmintrin.h>

#define NUM_THREADS 16
#define M 64
#define SSE_SIZE 4
#define AVX_SIZE 8

float rand_float(float s);
void matrix_gen(float *a,float *b,int N,float seed);
void print_matrix(float *a,int N);
float cal_trace(float *a,int N);
void matrix_mul_blocks(float a[M*M], float b[M*M], float c[M*M]);
void* matrix_mul(void *arg);
void matrix_c_add(float *record,int len, float *c, int N, int r, int l);
void set_small_matrix(float *record,int len,float *a,int N, int r,int l);

float rand_float(float s){// to produce a random float, 0<seed<1
	return 4*s*(1-s);
}

void matrix_gen(float *a,float *b,int N,float seed){//to generate two N*N(float) matrixs a & b
	float s=seed;
	for(int i=0;i<N*N;i++){
		s=rand_float(s);
		a[i]=s;
		s=rand_float(s);
		b[i]=s;
	}
}

void print_matrix(float *a,int N){// to print a N*N matrix a
	for(int i=0;i<N;i++){
		if(i%M==0){
			printf("\n");
		}
		for(int j=0;j<N;j++){
			if(j%M==0){
				printf(" ");
			}
			printf("%f ",a[i*N+j]);
		}
		printf("\n");
	}
	printf("\n");
}

float cal_trace(float *a,int N){// to calculate the trace of N*N matrix a
	float result=0;
	for(int i=0;i<N;i++){
		result+=a[i*(N+1)];
	}
	return result;
}

struct parameter{//to creata a struce which will be passed to the threads as argument
	int N;
	float *a;
	float *b;
	float *c;
	int number;
};


void matrix_mul_blocks(float a[M*M],float b[M*M], float c[M*M]){ // c= a*b
	int num= M/SSE_SIZE; // the number of blocks in a row, now the a,b,c is a num*num matrix, SSE*SSE size each block 
	for(int i=0;i<num;i++){
		int op1=i*SSE_SIZE*M;
		for(int j=0;j<num;j++){//calculating c[i][j] = sum(a[i][k]*b[k][j]);i
			int op2=j*SSE_SIZE*M;
			int op3=j*SSE_SIZE;
			float *record_c=(float *)malloc(SSE_SIZE*SSE_SIZE*sizeof(float)); //record of c
			
			for(int k=0;k<num;k++){//calculating a[i][k]*b[k][j], multiplication of SSE_SIZE*SSE_SIZE matrix			

				int op4=k*SSE_SIZE;
				// data initiating
				float *a0=a+(op1+op4); //the first address of block a[i][k];
				float *b0=b+(op4*M+op3); //the first address of block b[k][j];

				__m128 row0=_mm_load_ps(b0); b0+=M; //the first row of b
				__m128 row1=_mm_load_ps(b0); b0+=M;// the second row of b
				__m128 row2=_mm_load_ps(b0); b0+=M; // the third row of b
				__m128 row3=_mm_load_ps(b0); //the fourth row of b

				
				for(int cnt=0;cnt<SSE_SIZE;cnt++){
					__m128 c0= _mm_set1_ps(a0[M*cnt+0]); //a[cnt][0]	
					__m128 c1= _mm_set1_ps(a0[M*cnt+1]); //a[cnt][1]
					__m128 c2= _mm_set1_ps(a0[M*cnt+2]); //a[cnt][2]
					__m128 c3= _mm_set1_ps(a0[M*cnt+3]); //a[cnt][3]

					__m128 row =_mm_add_ps(
							_mm_add_ps(_mm_mul_ps(c0,row0),_mm_mul_ps(c1,row1)),
							_mm_add_ps(_mm_mul_ps(c2,row2),_mm_mul_ps(c3,row3))
							);

					_mm_store_ps(&record_c[SSE_SIZE*cnt],row);	
				}

				matrix_c_add(record_c,SSE_SIZE,c,M,i,j);
			}

			//print_matrix(record_c,SSE_SIZE);
			//print_matrix(c,M);
			free(record_c);	

		}
	}
}


void matrix_c_add(float *record,int len, float *c, int N, int r, int l){
	for(int i=0;i<len;i++){
		for(int j=0;j<len;j++){
			c[(r*len+i)*N+l*len+j]+=record[i*len+j];
		}
	}
}

void set_small_matrix(float *record,int len,float *a,int N, int r,int l){
	for(int i=0;i<len;i++){
		for(int j=0;j<len;j++){
			//printf("record[%d][%d]=a[%d][%d]\n",i,j,r*M+i,l*M+j);
			//printf("record[%d][%d]=%f\n",i,j,a[(r*M+i)*N+(l*M+j)]);
			record[i*len+j]=a[(r*len+i)*N+(l*len+j)];
		}
	}
}

void* matrix_mul(void *arg){
	struct parameter *p;
	p=(struct parameter *) arg;

	const int N=p->N;
	float *a=p->a;
	float *b=p->b;
	float *c=p->c;
	int number=p->number;

	const int P= N/M; //the number of a blocks in a row, divide the N*N matrix into P*P matrix

	int size_block_thread=P/NUM_THREADS; //the number of a blocks per thread
	int i_start=number*size_block_thread; //the cnt thread is for the cnt row
	int i_finish=i_start+size_block_thread;
	int j_start=0;
	int j_finish=P;

	for(int i=i_start;i<i_finish;i++){
		for(int j=j_start;j<j_finish;j++){
			for(int k=0; k<P; k++){
				//c[i][j]+=a[i][k]+b[k][j];
				
				float *a_copy=(float*)malloc(M*M*sizeof(float)); // block in a matrix
				float *b_copy=(float*)malloc(M*M*sizeof(float)); // block in a matrix
				float *record=(float*)malloc(M*M*sizeof(float)); // block in a matrix

				//init a_copy & b_copy
				//a_copy = a[i][k]

				/*
				printf("a=\n");
				print_matrix(a,N);
				*/

				set_small_matrix(a_copy,M,a,N,i,k);
				//b_copy = b[k][j]
				set_small_matrix(b_copy,M,b,N,k,j);

				//block mul
				matrix_mul_blocks(a_copy,b_copy,record);
				
				/*
				printf("a_copy=\n");
				print_matrix(a_copy,M);
				printf("b_copy=\n");
				print_matrix(b_copy,M);
				printf("record=\n");
				print_matrix(record,M);
				*/
				matrix_c_add(record,M,c,N,i,j);
				/*
				free(a_copy);
				free(b_copy);
				free(record);
				*/
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
	
//	print_matrix(a,N);
//	print_matrix(b,N);
//	print_matrix(c,N);

	free(a);
	free(b);
	free(c);
}

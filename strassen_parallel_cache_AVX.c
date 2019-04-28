/*************************************************************************
  > File Name: strassen_parallel_cache_AVX.c
	> Author: logos
	> Mail: 838341114@qq.com 
	> Created Time: 2019年04月28日 Sunday 16时36分44秒
 ************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<pthread.h>
#include<string.h>
#include<immintrin.h>

#define NUM_THREADS 32
#define M 64
#define SSE_SIZE 4
#define AVX_SIZE 8


float rand_float(float s);
void matrix_gen(float *a,float *b,int N,float seed);
void print_matrix(float *a,int N);
float cal_trace(float *a,int N);
void* matrix_mul(void *arg);
void small_matrix_mul(float *a, float *b, float *c);
void matrix_add(float *a, float *b, float *c, int N);

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
			printf("%6.4f ",a[i*N+j]);
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

void small_matrix_mul(float *a, float *b, float *c){

	float *a0=a;
	float *b0=b;
	float *c0=c;

	__m256 row0=_mm256_loadu_ps(b0); b0+=AVX_SIZE; //the first row of b
	__m256 row1=_mm256_loadu_ps(b0); b0+=AVX_SIZE;// the second row of b
	__m256 row2=_mm256_loadu_ps(b0); b0+=AVX_SIZE; // the third row of b
	__m256 row3=_mm256_loadu_ps(b0); b0+=AVX_SIZE;//the fourth row of b
	__m256 row4=_mm256_loadu_ps(b0); b0+=AVX_SIZE;
	__m256 row5=_mm256_loadu_ps(b0); b0+=AVX_SIZE;
	__m256 row6=_mm256_loadu_ps(b0); b0+=AVX_SIZE;
	__m256 row7=_mm256_loadu_ps(b0);

	for(int cnt=0;cnt<AVX_SIZE;cnt++){
		int temp=AVX_SIZE*cnt;

		__m256 c0= _mm256_set1_ps(a0[temp+0]); //a[cnt][0]	
		__m256 c1= _mm256_set1_ps(a0[temp+1]); //a[cnt][1]
		__m256 c2= _mm256_set1_ps(a0[temp+2]); //a[cnt][2]
		__m256 c3= _mm256_set1_ps(a0[temp+3]); //a[cnt][3]
		__m256 c4= _mm256_set1_ps(a0[temp+4]);
		__m256 c5= _mm256_set1_ps(a0[temp+5]);
		__m256 c6= _mm256_set1_ps(a0[temp+6]);
		__m256 c7= _mm256_set1_ps(a0[temp+7]);

		__m256 temp1=_mm256_add_ps(
				_mm256_add_ps(_mm256_mul_ps(c0,row0),_mm256_mul_ps(c1,row1)),
				_mm256_add_ps(_mm256_mul_ps(c2,row2),_mm256_mul_ps(c3,row3))
				);
		__m256 temp2=_mm256_add_ps(
				_mm256_add_ps(_mm256_mul_ps(c4,row4),_mm256_mul_ps(c5,row5)),
				_mm256_add_ps(_mm256_mul_ps(c6,row6),_mm256_mul_ps(c7,row7))
				);
		__m256 row =_mm256_add_ps(temp1,temp2);

		_mm256_storeu_ps(c0+AVX_SIZE*cnt,row);	
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

	const int P=N/M; //the number of blocks in a row
	int size_block_thread=P/NUM_THREADS; //the number of a blocks per thread
	int i_start=number*size_block_thread; //the cnt thread is for the cnt row
	int i_finish=i_start+size_block_thread;
	int j_start=0;
	int j_finish=P;

	for(int i=i_start;i<i_finish;i++){
		for(int j=j_start;j<j_finish;j++){ //c[i][j]=sum(a[i][k]*b[j][k])
			
			for(int k=0; k<P; k++){//calculating a[i][k]*b[k][j]

				int a_os= i*M*N+k*M; //the address offset of a
				int b_os= k*M*N+j*M; //the address offset of b
				int c_os= i*M*N+j*M;

				int num= M/AVX_SIZE/2; // the number of blocks in a row, now the a,b,c is a num*num matrix, AVX*AVX size each block 
				for(int ii=0;ii<num;ii++){
					for(int jj=0;jj<num;jj++){//calculating c[ii][jj]. A 16*16 matrix

						//A11 B11 C11 init 8*8 matrix
						for(int iii=0;iii<8;iii++)
						//A12 B12 C12 init 8*8 matrix
						
						//A21 B21 C21 init

						//A22 B22 C22 init

						//P1=(A11+A22)(B11+B22)
						//P2=(A21+A22)(B11)
						//P3=(A1)(B12-B22)
						//P4=A22(B21-B11)
						//P5=(A11+A12)(B22)
						//P6=(A21-A11)(B11+B12)
						//P7=(A12-A22)(B21+B22)

						//C11=P1+P4-P5+P7
						//C12=P3+P5
						//C21=P2+P4
						//C22=P1+P3-P2+P6
						
					}
				}
			}			
		}
	}
}	
/*
 * inputs: N, seed
*/

int main(int argc, char *argv[]){

	//parameter initiation
	const int N = atoi(argv[1]); //the size of matrix
	float seed = atof(argv[2]);

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

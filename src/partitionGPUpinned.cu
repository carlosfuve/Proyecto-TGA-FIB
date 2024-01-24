#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <sys/times.h>
#include <sys/resource.h>
#include <cmath>
#include <stack>

using namespace std;


#define N 254
#define nThreads 24
#define PINNED 1

__global__ void partkernel(int pivot, int *array,int *arraysorted, int start, int end,int *gbeg, int *lbeg) {
    __shared__ int lt[nThreads];
    __shared__ int gt[nThreads];
    __shared__ int lfrom;
    __shared__ int gfrom;


    int idThread = threadIdx.x;
    lt[idThread] = 0;
    gt[idThread] = 0;

    //gridDim es cuantos bloques hay
    //blockDim es cuantos threads hay
    int blockT = (end-start+(gridDim.x*blockDim.x-1))/(gridDim.x*blockDim.x);
    int i_start = start + (blockIdx.x * blockDim.x * blockT) + threadIdx.x * blockT;  
    int i_end = i_start + blockT;
    if ((blockIdx.x==gridDim.x-1) && (threadIdx.x==blockDim.x-1)) i_end=end;

    //Cada thread cuenta cuantos mayores y menos al bloque asociado
    for (int i=i_start; i<i_end; i++){
        if (array[i] > pivot) gt[idThread] = gt[idThread]+1;
        if (array[i] < pivot) lt[idThread] = lt[idThread]+1;
    }

    __syncthreads();

    if (threadIdx.x == 0){
        int ltsum = 0;
        int gtsum = 0;
        for(int k = 0; k < nThreads; k++){
            ltsum += lt[k];
            gtsum += gt[k];
        }

        // Se guarda el valor de la suma para saber donde se empezará a indexar el bloque
        int idblock = blockIdx.x;
        lbeg[idblock] = ltsum;
        gbeg[idblock] = gtsum;

        // Se calculan los indices de donde empieza el bloque
        lfrom = 0;
        gfrom = 0;
        for(int j = 0; j < blockIdx.x; ++j){
	        lfrom += lbeg[j];
            gfrom += gbeg[j];
        }
    }

    __syncthreads();

    //Se calcula los indices de cada bloque asignado a un thread para sumarselo al indice del bloque e indexar
    int sumlt = 0;
    int sumgt = 0;
    for(int k = 0; k < idThread; k++){
	    sumlt += lt[k];
	    sumgt +=  gt[k];
    }
    lt[idThread] = sumlt;
    gt[idThread] = sumgt;

    //El thread 0 empieza a indexar al incio del bloque
    if(threadIdx.x ==  0){ 
	    lt[0] = 0;
	    gt[0] = 0;
    }

    __syncthreads();

    int idlt = lfrom + lt[threadIdx.x];
    int idgt = end-1 - gfrom - gt[threadIdx.x];

    //Se asignan los valores del vector original al vector que va a tener los mayores a la derecha y los menores a la izquierda
    for (int i=i_start; i<i_end; i++){
        if(array[i] < pivot) {
            arraysorted[idlt] = array[i];
            ++idlt;
        }
        if (array[i] > pivot) {
            arraysorted[idgt] = array[i];
            --idgt;
        }
    }

    __syncthreads();
    
    //Por último se añaden los pivotes que habian en el vector en las posiciones correspondientes
    //Al principio del total de los menores hasta N-número de mayores que el pivote
    if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
        int ltsumt = 0;
        int gtsumt = 0;
        for(int n = 0; n < gridDim.x; n++){
            ltsumt += lbeg[n];
            gtsumt += gbeg[n];
        }

        int endp = end - gtsumt;
        for (int j = ltsumt; j < endp; j++){
            arraysorted[j] = pivot;
        }
    }
}


void CheckCudaError(char sms[], int line);
float GetTime(void);
void quicksortSeq(int *array, int left, int rigth);
void InitRandom(int *v);
void InitCeros(int *v, int size);
int median(int v1, int v2, int v3);


int main(int argc, char** argv){

    int nBlocksV = (N+nThreads-1)/nThreads; //Funciona bien en cualquier caso

    int *gbeg, *lbeg;
    int *array, *arraysorted;
    int *darray, *darraysorted, *dlbeg, *dgbeg;

    float time1, time2, time3, time4;
	float tempTotal, tempKernel;
  	float tempHD, tempDH;

	cudaEvent_t E0, E1, E2, E3;
	cudaEventCreate(&E0);
  	cudaEventCreate(&E1);
  	cudaEventCreate(&E2);
  	cudaEventCreate(&E3);

    //Obtener memoria en el host
    int nBlocks = N * sizeof(int);
    int nBlocksT = nBlocksV * sizeof(int);


    if (PINNED) {
    	// Obtiene Memoria [pinned] en el host
    	cudaMallocHost((int**)&array, nBlocks); 
    	cudaMallocHost((int**)&arraysorted, nBlocks); 
        cudaMallocHost((int**)&gbeg, nBlocksT); 
    	cudaMallocHost((int**)&lbeg, nBlocksT); 
  	}
  	else {
    	// Obtener Memoria en el host
        array = (int*) malloc(nBlocks);
        arraysorted = (int*) malloc(nBlocks);
        gbeg = (int*) malloc(nBlocksT);
        lbeg = (int*) malloc(nBlocksT);
  	}

    InitRandom(array);
    InitCeros(arraysorted,N);
    InitCeros(lbeg,nBlocksV);
    InitCeros(gbeg,nBlocksV);

   int pivote = median(array[0], array[N/2], array[N-1]);
   for(int j = 0; j < N; j++) printf("%d ",array[j]);
   printf("\n");

   //Obtener memoria en el device
   cudaMalloc((int**)&darraysorted,nBlocks);
   CheckCudaError((char *) "Obtener memoria en el device", __LINE__);
   cudaMalloc ((int**)&darray,nBlocks);
   CheckCudaError((char *) "Obtener memoria en el device", __LINE__);
   cudaMalloc((int**)&dlbeg,nBlocksT);
   CheckCudaError((char *) "Obtener memoria en el device", __LINE__);
   cudaMalloc((int**)&dgbeg,nBlocksT);
   CheckCudaError((char *) "Obtener memoria en el device", __LINE__);

   	cudaEventRecord(E0, 0);
  	cudaEventSynchronize(E0);

   //Copiar datos del host al device
   cudaMemcpy(darray,array,nBlocks,cudaMemcpyHostToDevice);
   CheckCudaError((char *) "Copiar datos Host to Device", __LINE__);
   cudaMemcpy(darraysorted,arraysorted,nBlocks,cudaMemcpyHostToDevice);
   CheckCudaError((char *) "Copiar datos Host to Device", __LINE__);
   cudaMemcpy(dlbeg,lbeg,nBlocksT,cudaMemcpyHostToDevice);
   CheckCudaError((char *) "Copiar datos Host to Device", __LINE__);
   cudaMemcpy(dgbeg,gbeg,nBlocksT,cudaMemcpyHostToDevice);
   CheckCudaError((char *) "Copiar datos Host to Device", __LINE__);

   	cudaEventRecord(E1, 0);
  	cudaEventSynchronize(E1);
   
   //Llamada al kernel
   partkernel<<<nBlocksV, nThreads>>>(pivote,darray,darraysorted,0,N,dgbeg,dlbeg);
   CheckCudaError((char *) "Invocar al kernel", __LINE__);

   	cudaEventRecord(E2, 0);
  	cudaEventSynchronize(E2);
   
   //Obtener resultado desde el host
   cudaMemcpy(arraysorted,darraysorted,nBlocks,cudaMemcpyDeviceToHost);
   CheckCudaError((char *) "Copiar datos Device to Host", __LINE__);
   cudaMemcpy(lbeg,dlbeg,nBlocksT,cudaMemcpyDeviceToHost);
   CheckCudaError((char *) "Copiar datos Device to Host", __LINE__);
   cudaMemcpy(gbeg,dgbeg,nBlocksT,cudaMemcpyDeviceToHost);
   CheckCudaError((char *) "Copiar datos Device to Host", __LINE__);

   cudaEventRecord(E3, 0);
   cudaEventSynchronize(E3);

   //Liberar memoria en el device
   cudaFree(darraysorted); cudaFree(darray); cudaFree(dlbeg); cudaFree(dgbeg);
   //Sincronizar device
   cudaDeviceSynchronize();



    //Eventos de cuda
	cudaEventElapsedTime(&tempTotal,  E0, E3);
  	cudaEventElapsedTime(&tempKernel, E1, E2);
  	cudaEventElapsedTime(&tempHD, E0, E1);
  	cudaEventElapsedTime(&tempDH, E2, E3);

    printf("\n");
	printf("Tiempo Global: %4.6f milseg\n", tempTotal);
  	printf("Tiempo Kernel: %4.6f milseg\n", tempKernel);

   //TEST
   bool test = true;

   int ltsumt = 0;
   int gtsumt = 0;
   for(int n = 0; n < nBlocksV; n++){
        ltsumt += lbeg[n];
        gtsumt += gbeg[n];
    }

    time1 = GetTime();
    quicksortSeq(arraysorted, 0, ltsumt-1);
    quicksortSeq(arraysorted, N-gtsumt, N-1);
    time2 = GetTime();
    printf("Tiempo Ordenación: %4.6f milseg\n", time2-time1);

    time3 = GetTime();
    quicksortSeq(array, 0, N-1);
    time4 = GetTime();
    printf("Tiempo Secuencial: %4.6f milseg\n", time4-time3);

    printf("Tiempo Host to Device: %4.6f milseg\n", tempHD);
  	printf("Tiempo Device to Host: %4.6f milseg\n", tempDH);

    for(int m = 1; m < N; m++){
        if (arraysorted[m-1] > arraysorted[m]) {
            test = false;
            break;
        }
    }

    printf("Pinned?: %d\n", PINNED);
    printf ("N Elementos: %d\n", N);
    printf("nThreads: %d\n", nThreads);
    printf("Bloques: %d\n",nBlocksV);
    printf("Pivote: %d\n",pivote);
    printf("\n");
    for(int j = 0; j < N; j++) printf("%d ",arraysorted[j]);
    printf("\n");

    if (!test) printf("TEST FAILED\n");
    else printf("TEST PASS\n");


    cudaEventDestroy(E0); 
  	cudaEventDestroy(E1); 
  	cudaEventDestroy(E2); 
  	cudaEventDestroy(E3);

    //Liberar memoria en el host
	if (PINNED) {
    	cudaFreeHost(array);cudaFreeHost(arraysorted); cudaFreeHost(lbeg); cudaFreeHost(gbeg);
  	}
  	else {
		free(array);free(arraysorted); free(lbeg); free(gbeg);
  	}
}




void swap(int *xp, int *yp){
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

int partition(int *array, int left, int right){
    int x = array[left];

    //Los primeros valores no dan segmentation fault porque se ejecuta el do antes de acceder al vector
    int i = left-1;
    int j = right+1;

    for(;;){
        do {
            i++;
        }while(array[i] < x);

        do{
            j--;
        }while(array[j] > x);
        //Cuando ya no encuentra más grandes a la izquierda del pivote, acaba
        if(i >= j) return j;
        swap(&array[i],&array[j]);
    }
}

void quicksortSeq(int *array, int left, int rigth){
	if (left < rigth){
		int pi = partition(array,left,rigth);

		quicksortSeq(array,left,pi);
		quicksortSeq(array,pi+1,rigth);
	}
}


void InitRandom(int *v) {
	for (int i = 0; i < N; i++) {
		v[i] = rand() % (2*N); //Genera números entre 0 y 2N
	}
}

void InitCeros(int *v, int size){
	for(int i = 0; i < size; i++) v[i] = 0;
}

int median(int v1, int v2, int v3){
    if (v1 >= v2){
        if (v1 >= v3) return v1;
        return v3;
    }
    else{
        if (v2 >= v3) return v2;
        return v3;
    }
}


void CheckCudaError(char sms[], int line){
	cudaError_t error;
	error = cudaGetLastError();
	if (error){
		printf("(ERROR) %s - %s in %s at line %d\n",sms,cudaGetErrorString(error), __FILE__, line);
	}
}

float GetTime(void){
	struct timeval tim;
	struct rusage ru;
	getrusage(RUSAGE_SELF,&ru);
	tim = ru.ru_utime;
	return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
}

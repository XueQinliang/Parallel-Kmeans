#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
using namespace std;
#define blocknum 64
struct matrix{
    float *data;
    int *statusno;
    int x;
    int y;
};

double timestamp(){
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

void CreateData(float* data, int x, int y){
    srand((unsigned int)time(NULL));
    for(int i = 0; i < y; i +=5){
        for(int j = 0; j < x; j++){
            if(j%2==1)
                data[i*x+j] = rand()%30+50;
            else
                data[i*x+j] = rand()%30-50;
            data[i*x+j+x] = rand()%30+100;
            data[i*x+j+2*x] = rand()%30-50;
            if(j%2==0){
                data[i*x+j+3*x] = rand()%30+50;
            }else{
                data[i*x+j+3*x] = rand()%30-50;
            }
            data[i*x+j+4*x] = rand()%100-20;
        }
    }
}
void CreateParam(float* data, int x, int y){
    srand((unsigned int)time(NULL));
    for(int i = 0; i < y; i ++){
        for(int j = 0; j < x; j++){
            data[i*x+j] = rand()%180-50;
        }
    }
}
float SqrtDistance(float* a, float* b,int size){
    float dis = 0;
    float temp = 0;
    for(int k = 0; k < size ; k ++){
        temp = a[k] - b[k];
        dis += temp*temp;
    }
    dis = sqrt(dis);
    return dis;
}
 
void Estep(struct matrix m, struct matrix para){
    for(int i = 0; i < m.y; i++){
        float mindis = (float)0x7fffffff;
        for(int j = 0; j < para.y; j++){
            float dis = 0;
            float temp = 0;
            for(int k = 0; k < m.x; k ++){
                temp = m.data[i*m.x+k] - para.data[j*m.x+k];
                dis += temp*temp;
            }
            dis = sqrt(dis);
            if(dis < mindis){
                m.statusno[i] = para.statusno[j];
                mindis = dis;
            }
        }
    }
}
__global__ void Estep_gpu(int datay, int paray, int size, float* data, float* para, int* datastatus, int* parastatus){
    int i = blockIdx.x * blockDim.x;
    int x = threadIdx.x;
    if(i+x<datay){
        // __shared__ float tile[blocknum][16];
        // for(int k = 0; k < size; k ++){
        //     tile[x][k] = data[(i+x)*size+k];
        // }
        // __syncthreads();
        float mindis = (float)0x7fffffff;
        for(int j = 0; j < paray; j++){
            float dis = 0;
            float temp = 0;
            for(int k = 0; k < size; k ++){
                temp = data[(i+x)*size+k] - para[j*size+k];
                dis += temp*temp;
            }
            dis = sqrt(dis);
            if(dis < mindis){
                datastatus[i] = parastatus[j];
                mindis = dis;
            }
        }
    }
}
// __global__ void Estep_gpu(int datay, int paray, int size, float* data, float* para, int* datastatus, int* parastatus){
//     int i = blockIdx.x * blockDim.x;
//     int x = threadIdx.x;
//     if(i+x<datay){
//         __shared__ float tile[blocknum][16];
//         for(int k = 0; k < size; k ++){
//             tile[x][k] = data[(i+x)*size+k];
//         }
//         __syncthreads();
//         float mindis = (float)0x7fffffff;
//         for(int j = 0; j < paray; j++){
//             float dis = 0;
//             float temp = 0;
//             for(int k = 0; k < size; k ++){
//                 temp = tile[x][k] - para[j*size+k];
//                 dis += temp*temp;
//             }
//             dis = sqrt(dis);
//             if(dis < mindis){
//                 datastatus[i] = parastatus[j];
//                 mindis = dis;
//             }
//         }
//     }
// }
void Mstep(struct matrix m, struct matrix para, int* num){
    for(int i = 0 ; i < para.y ; i++){
        num[i] = 0;
        for(int j = 0; j < para.x; j++){
            para.data[i*m.x+j] = 0;
        }
    }
    for(int i = 0; i < m.y; i++){
        #pragma unroll
        for(int j = 0; j < m.x; j++){
            para.data[m.statusno[i]*m.x+j] += m.data[i*m.x+j];
        }
        num[m.statusno[i]]++;
    }
    for(int i = 0; i < para.y; i++){
        if(num[i]!=0){
            for(int j = 0; j < para.x; j++){
                para.data[i*para.x+j] /= num[i];
            }
        }
    }
}
int main(int argc,char** argv) {
    // initial data
    struct matrix m;
    m.x = atoi(argv[3]);
    m.y = atoi(argv[1]);
    m.data = new float[m.y*m.x];
    m.statusno = new int[m.y];
    CreateData(m.data,m.x,m.y);
    // initial parameter
    struct matrix para;
    para.x = atoi(argv[3]);
    para.y = atoi(argv[2]);
    para.data = new float[para.y*para.x];
    para.statusno = new int[para.y];
    CreateParam(para.data,para.x,para.y);
    for(int i = 0 ; i < para.y ; i ++){
        para.statusno[i] = i;
    }
    float* gpudata;
    float* gpupara;
    int* datastatus;
    int* parastatus;
    cudaMalloc((void**)&gpudata, sizeof(float)*m.y*m.x);
    cudaMemcpy(gpudata, m.data, sizeof(float)*m.y*m.x, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpupara, sizeof(float)*para.y*para.x);
    cudaMemcpy(gpupara, para.data, sizeof(float)*para.y*para.x, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&parastatus,sizeof(int)*para.y);
    cudaMemcpy(parastatus, para.statusno, sizeof(int)*para.y, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&datastatus,sizeof(int)*m.y);
    // EM-step
    double time1 = timestamp();
    double timee = 0;
    double timem = 0;
    int* num = new int[para.y];
    for(int i = 0; i < 10000; i++){
        double time3= timestamp();
        cudaMemcpy(datastatus, m.statusno, sizeof(int)*m.y, cudaMemcpyHostToDevice);
        //Estep(m,para);
        Estep_gpu<<<(m.y+blocknum-1)/blocknum,blocknum>>>(m.y,para.y,m.x,gpudata,gpupara,datastatus,parastatus);
        cudaMemcpy(m.statusno, datastatus, sizeof(int)*m.y, cudaMemcpyDeviceToHost);
        double time4= timestamp();
        Mstep(m,para,num);
        double time5= timestamp();
        timee += time4-time3;
        timem += time5-time4;
    }
    double time2 = timestamp();
    cout<<argv[1]<<","<<argv[2]<<","<<argv[3]<<","<<time2-time1<<endl;
    // cout<<timee<<" "<<timem<<endl;
    FILE* d = fopen("result.csv","w");
    for(int i = 0 ; i < m.y ; i++){
        for(int j = 0; j < m.x; j++){
            fprintf(d,"%f,",m.data[i*m.x+j]);
        }
        fprintf(d,"%d\n",m.statusno[i]);
    }
    fclose(d);
    FILE* p = fopen("point.csv","w");
    for(int i = 0 ; i < para.y ; i++){
        if(num[i]!=0){
            for(int j = 0; j < para.x; j++){
                if(j!=para.x-1)
                    fprintf(d,"%f,",para.data[i*para.x+j]);
                else
                    fprintf(d,"%f\n",para.data[i*para.x+j]);
            }
        }
    }
    fclose(p);
    free(m.data);
    free(para.data);
    free(m.statusno);
    free(para.statusno);
    cudaFree(gpudata);
    cudaFree(gpupara);
    cudaFree(datastatus);
    cudaFree(parastatus);
    return 0;
}

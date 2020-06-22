#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
using namespace std;
struct matrix{
    float **data;
    int *statusno;
    int x;
    int y;
};

double timestamp(){
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

void CreateData(float** data, int x, int y){
    srand((unsigned int)time(NULL));
    for(int i = 0; i < y; i +=5){
        for(int j = 0; j < x; j++){
            if(j%2==1)
                data[i][j] = rand()%30+50;
            else
                data[i][j] = rand()%30-50;
            data[i+1][j] = rand()%30+100;
            data[i+2][j] = rand()%30-50;
            if(j%2==0){
                data[i+3][j] = rand()%30+50;
            }else{
                data[i+3][j] = rand()%30-50;
            }
            data[i+4][j] = rand()%200-100;
        }
    }
}
void CreateParam(float** data, int x, int y){
    srand((unsigned int)time(NULL));
    for(int i = 0; i < y; i ++){
        for(int j = 0; j < x; j++){
            data[i][j] = rand()%180-50;
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
void EMstep(struct matrix m, struct matrix para){
    #pragma omp parallel
    {
        #pragma omp for schedule(guided)
        for(int i = 0; i < m.y; i++){
            float mindis = (float)0x7fffffff;
            for(int j = 0; j < para.y; j++){
                //float dis = SqrtDistance(m.data[i],para.data[j],m.x);
                float dis = 0;
                float temp = 0;
                for(int k = 0; k < m.x; k ++){
                    temp = m.data[k] - para.data[k];
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
        for(int i = 0 ; i < para.y ; i++){
            for(int j = 0; j < para.x; j++){
                para.data[i][j] = 0;
            }
        }
        for(int i = 0; i < m.y; i++){
            for(int j = 0; j < m.x; j++){
                para.data[m.statusno[i]][j] += m.data[i][j]/m.y;
            }
        }
}
void Estep(struct matrix m, struct matrix para){
    #pragma omp parallel for schedule(guided)
    for(int i = 0; i < m.y; i++){
        float mindis = (float)0x7fffffff;
        for(int j = 0; j < para.y; j++){
            //float dis = SqrtDistance(m.data[i],para.data[j],m.x);
            float dis = 0;
            float temp = 0;
            for(int k = 0; k < m.x; k ++){
                temp = m.data[i][k] - para.data[j][k];
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
void Mstep(struct matrix m, struct matrix para, int* num){
    for(int i = 0 ; i < para.y ; i++){
        num[i] = 0;
        for(int j = 0; j < para.x; j++){
            para.data[i][j] = 0;
        }
    }
    for(int i = 0; i < m.y; i++){
        #pragma unroll
        for(int j = 0; j < m.x; j++){
            para.data[m.statusno[i]][j] += m.data[i][j];
        }
        num[m.statusno[i]]++;
    }
    for(int i = 0; i < para.y; i++){
        if(num[i]!=0){
            for(int j = 0; j < para.x; j++){
                para.data[i][j] /= num[i];
            }
        }
    }
}
int main(int argc, char* argv[]){
    // initial data
    struct matrix m;
    m.x = atoi(argv[3]);
    m.y = atoi(argv[1]);
    m.data = new float*[m.y];
    m.statusno = new int[m.y];
    for(int i = 0 ; i < m.y ; i ++){
        m.data[i] = new float[m.x];
    }
    CreateData(m.data,m.x,m.y);
    // initial parameter
    struct matrix para;
    para.x = atoi(argv[3]);
    para.y = atoi(argv[2]);
    para.data = new float*[para.y];
    para.statusno = new int[para.y];
    for(int i = 0 ; i < para.y ; i ++){
        para.data[i] = new float[para.x];
    }
    CreateParam(para.data,para.x,para.y);
    for(int i = 0 ; i < para.y ; i ++){
        para.statusno[i] = i;
    }
    // EM-step
    double time1 = timestamp();
    double timee = 0;
    double timem = 0;
    int* num = new int[para.y];
    for(int i = 0; i < 10000; i++){
        double time3= timestamp();
        Estep(m,para);
        double time4= timestamp();
        Mstep(m,para,num);
        double time5= timestamp();
        timee += time4-time3;
        timem += time5-time4;
    }
    double time2 = timestamp();
    cout<<argv[1]<<","<<argv[2]<<","<<argv[3]<<","<<time2-time1<<endl;
    //cout<<timee<<" "<<timem<<endl;
    // for(int i = 0 ; i < m.y ; i ++){
    //    cout<<m.data[i][0]<<","<<m.data[i][1]<<","<<m.statusno[i]<<endl;
    // }
    for(int i = 0 ; i < m.y ; i ++){
        delete [] m.data[i];
    }
    for(int i = 0 ; i < para.y ; i ++){
        delete [] para.data[i];
    }
    delete [] m.data;
    delete [] para.data;
    delete [] m.statusno;
    delete [] para.statusno;
    delete [] num;
    return 0;
}

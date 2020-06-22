#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include<sys/time.h>

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
            data[i+4][j] = rand()%100-20;
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
    for(int k = 0; k < size ; k ++){
        float d = a[k] - b[k];
        dis += d*d;
    }
    dis = sqrt(dis);
    return dis;
}
void Estep(struct matrix m, struct matrix para){
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
    /*for(int i = 0; i < m.y; i++){
        cout<<m.statusno[i]<<" ";
    }
    cout<<endl;*/
}
void Mstep(struct matrix m, struct matrix para, int* num){
    for(int i = 0 ; i < para.y ; i++){
        num[i] = 0;
        for(int j = 0; j < para.x; j++){
            para.data[i][j] = 0;
        }
    }
    for(int i = 0; i < m.y; i++){
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
    int* num = new int[para.y];
    // EM-step
    double time1 = timestamp();
    for(int i = 0; i < 10000; i++){
        Estep(m,para);
        Mstep(m,para,num);
    }
    double time2 = timestamp();
    cout<<time2-time1<<endl;
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

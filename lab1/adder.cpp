#include <iostream>
using namespace std;
#include<chrono>
//n个数求和的平凡算法
#define N 2048
void cal_adder(float *a,float &sum)
{
    sum = 0.0;
    for(int i=0;i<N;i++)
    {
        sum+=a[i];
    }
}
int main()
{
    //初始化a[N]和sum
    float* a =new float[N];

    for(int i=0;i<N;i++)
    {
        a[i] = (i+1)*1.0 ;
    }
    float sum = 0.0;

    //调用函数部分
    auto start = std::chrono::high_resolution_clock::now();
    cal_adder(a,sum);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "function running time: " << duration.count() << "ns" << std::endl;

    delete[] a;
    return 0;
}
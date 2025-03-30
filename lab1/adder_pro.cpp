#include <iostream>
using namespace std;
//n个数求和的优化算法
#define N 2048
void cal_adder_pro1(float *a,float &sum)
{
    float sum1=0.0;
    float sum2=0.0;
    sum = 0.0;
    for (int i = 0;i < N; i += 2)
    {
         sum1 += a[i];
         sum2 += a[i + 1];
    }
    sum = sum1 + sum2;
}
void cal_adder_pro2(float *a,float &sum)
{
    int n=N;
    for (int m = n; m > 1; m /= 2) // log(n)个步骤
       {
        for (int i = 0; i < m / 2; i++)
         { 
            a[i ] = a[i * 2] + a[i * 2 + 1];// 相邻元素相加连续存储到数组最前面
         }
       }
    sum = a[0];//最终结果为a[0]
}
int main()
{
    //初始化a[N]和sum
    float *a =new float[N];

    for(int i=0;i<N;i++)
    {
        a[i] = (i+1)*1.0 ;
    }
    float sum1=0.0;
    float sum2=0.0;
    float sum = 0.0;

    //算法部分
    // 多链路式
    cal_adder_pro1(a,sum);
    sum = 0.0;
    cal_adder_pro2(a,sum);

    delete[] a;
    return 0;
}
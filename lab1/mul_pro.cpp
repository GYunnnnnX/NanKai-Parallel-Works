#include <iostream>
using namespace std;
//矩阵与向量内积的优化算法
#define N 4096
void cal_mul_pro(float **b, float *a, float *sum)
{
     // 改为逐行访问矩阵元素：一步外层循环计算不出任何一个内积，只是向每个内积累加一个乘法结果
     for (int j = 0; j < N; j++)//特定的行
     {
          for (int i = 0; i < N; i++)//一行中，每列往后遍历
          {
              sum[i] += b[j][i] * a[j];//sum只有在最后一行运行完后才会出现全部的结果。
          }
     }
}
int main()
{
    float** b = new float* [N];
    for (int i = 0; i < N; i++) {
        b[i] = new float[N];
    }
    float* a = new float[N];
    float* sum = new float[N];
    //统一的初始化b[N][N]
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float value = i * j + i + j;
            b[i][j] = value;
        }
    }
    //统一的初始化a[N]
    for (int i = 0; i < N; i++)
    {
        a[i] = (i + 1) * 1.0;
    }
    //调用计算函数
    cal_mul_pro(b, a, sum);
  
 
    for (int i = 0; i < N; i++) delete[] b[i];
    delete[] b;
    delete[] a;
    delete[] sum;
    return 0;
}


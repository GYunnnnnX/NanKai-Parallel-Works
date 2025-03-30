#include <iostream>
using namespace std;
//矩阵与向量内积的平凡算法
#define N 4096
void cal_mul(float **b, float *a, float *sum)
{
    //逐列访问矩阵元素：一步外层循环（内存循环一次完整执行）计算出一个内积结果
    for (int i = 0; i < N; i++)
    {
        sum[i] = 0.0;
        for (int j = 0; j < N; j++)
        {
            sum[i] += b[j][i] * a[j];
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
    cal_mul(b, a, sum);
  
 
    for (int i = 0; i < N; i++) delete[] b[i];
    delete[] b;
    delete[] a;
    delete[] sum;
    return 0;
}


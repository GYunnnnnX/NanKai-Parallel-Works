#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 多优先队列编译指令如下
// g++ main2.cpp train.cpp guessing2.cpp md5.cpp -o main
// g++ main2.cpp train.cpp guessing2.cpp md5.cpp -o main -O1
// g++ main2.cpp train.cpp guessing2.cpp md5.cpp -o main -O2

int main()
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init(q.m.ordered_pts);
    //八个线程
    int num_threads = 8;
    vector<PriorityQueue> queues(num_threads);
    vector<vector<PT>> pts_per_thread(num_threads);


    vector<vector<string>> local_guesses(num_threads);
    //为了负载均衡，采用模的方式分配任务
    for (int i = 0; i < q.m.ordered_pts.size(); ++i) {
        pts_per_thread[i % num_threads].push_back(q.m.ordered_pts[i]);
    }
    omp_set_num_threads(8);
    #pragma omp parallel for
    for (int i = 0; i < num_threads; ++i) {
        queues[i].m = q.m;
        queues[i].init(pts_per_thread[i]);  // 只传每个线程对应的Pt子集
    }

    cout << "here" << endl;
    int curr_num = 0;
    q.total_guesses=0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./files/results.txt");
    while (true)
    {
        bool flag = true;//是否可以停止了
        for (int i = 0; i < num_threads; ++i) {
            if(!queues[i].priority.empty())
            {
                flag =false;
            }
        }
        if(flag)break;

        int local_sum = 0;
        #pragma omp parallel for reduction(+:local_sum)
        for (int i = 0; i < num_threads; ++i) {
            queues[i].PopNext();
            local_guesses[i] = queues[i].guesses;
            local_sum += queues[i].guesses.size();
            queues[i].guesses.clear();
        }
        q.total_guesses += local_sum;


        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " <<history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n=10000000;
            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
                cout << "Hash time:" << time_hash << "seconds"<<endl;
                cout << "Train time:" << time_train <<"seconds"<<endl;
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            // 在主线程中合并结果
            for (int i = 0; i < num_threads; ++i) {
                q.guesses.insert(q.guesses.end(), local_guesses[i].begin(), local_guesses[i].end());
            }
            auto start_hash = system_clock::now();
            bit32 state[4];
            /*for (string pw : q.guesses)
            {
                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
               // MD5Hash(pw, state);
                MD5Hash(pw, state);

                // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                // a<<pw<<"\t";
                // for (int i1 = 0; i1 < 4; i1 += 1)
                // {
                //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
                // }
                // a << endl;
            }*/
            
            for (int i = 0; i < q.guesses.size(); i += 4) {
                //每次跑四个
                string batch[4];
                bit32 states[4][4];
            
                // 初始化batch,填充不满的部分（最后几个）
                for (int j = 0; j < 4; ++j) {
                    batch[j] = (i + j < q.guesses.size()) ? q.guesses[i + j] : "";
                }
            
                MD5Hash_neon_parallel(batch, states);
            }
            
            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
}

#include "PCFG.h"
#include "md5.h"
#include <mpi.h>
#include <chrono>
#include <fstream>
#include <unordered_set>
using namespace std;
using namespace chrono;

//编译指令如下
//mpic++ -o correctness_guess correctness_mpi.cpp guessing.cpp train.cpp md5.cpp -O2


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Train阶段
    double wall_train_start = 0, wall_train_end = 0, wall_train_time = 0;
    if (rank == 0) wall_train_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    PriorityQueue q;
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        wall_train_end = MPI_Wtime();
        wall_train_time = wall_train_end - wall_train_start;
    }

    //加载测试集
    unordered_set<string> test_set;
    ifstream file("/guessdata/Rockyou-singleLined-full.txt");
    string pw;
    int count = 0;
    while (file >> pw && count < 1000000) {
        test_set.insert(pw);
        count++;
    }

    //Guess阶段
    double wall_guess_start = 0, wall_guess_end = 0, wall_guess_time = 0;
    if (rank == 0) wall_guess_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    //猜测
    int local_cracked = 0, local_total_guesses = 0;
    q.init();

    //Hash阶段
    double wall_hash_start = 0, wall_hash_end = 0, wall_hash_time = 0;
    if (rank == 0) wall_hash_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    while (!q.priority.empty()) {
        q.PopNext();  // PopNext内部按rank分配任务

        bit32 state[4];
        for (const string& pw : q.guesses) {
            if (test_set.count(pw)) local_cracked++;
            MD5Hash(pw, state);
        }
        local_total_guesses += q.guesses.size();
        q.guesses.clear();

        int global_guesses = 0;
        MPI_Allreduce(&local_total_guesses, &global_guesses, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (global_guesses > 10000000) break;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        wall_hash_end = MPI_Wtime();
        wall_hash_time = wall_hash_end - wall_hash_start;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        wall_guess_end = MPI_Wtime();
        wall_guess_time = wall_guess_end - wall_guess_start;
    }

    //汇总cracked数目
    int total_cracked = 0;
    MPI_Reduce(&local_cracked, &total_cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    //输出
    if (rank == 0) {
        cout << "Guess time:" << wall_guess_time << "seconds" << endl;
        cout << "Hash time:" << wall_hash_time << "seconds" << endl;
        cout << "Train time:" << wall_train_time << "seconds" << endl;
        cout << "Cracked:" << total_cracked << endl;
    }

    MPI_Finalize();
    return 0;
}
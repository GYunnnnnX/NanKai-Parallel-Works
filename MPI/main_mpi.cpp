#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <sstream>
using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 所有进程都需要训练模型
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    double time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    // 只有主进程初始化优先队列
    if (rank == 0) {
        //std::cout << "start " << q.priority.size() << std::endl;
        q.init();
        std::cout << "[rank 0] Priority queue initialized with size: " << q.priority.size() << std::endl;
    }

    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);
    //std::cout << "[rank " << rank << "] model trained and initialized" << std::endl;

    double time_hash = 0;
    double time_guess = 0;
    int curr_num = 0;
    int history = 0;
    auto start = system_clock::now();
    
    if (rank == 0) {
        // 主进程逻辑
        while (!q.priority.empty()) {
            std::vector<int> dispatched_ranks;
            // 分发任务给工作进程
            for (int i = 1; i < size && !q.priority.empty(); i++) {
                PT pt = q.priority.front();
                q.priority.erase(q.priority.begin());
                
                // 序列化PT
                std::string pt_str = pt.serialize();
                int str_size = pt_str.size();
                
                // 发送字符串大小
                MPI_Send(&str_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                // 发送字符串数据
                MPI_Send(pt_str.c_str(), str_size, MPI_CHAR, i, 1, MPI_COMM_WORLD);
                dispatched_ranks.push_back(i); // 记录实际分发的进程号
            }
            
            // 收集结果
            for (int rank_id : dispatched_ranks) {
                //if (q.priority.empty()) break;
                
                // 接收生成的口令数量
                int guess_count;
                MPI_Recv(&guess_count, 1, MPI_INT, rank_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //std::cout << "[rank 0] received " << guess_count << " guesses from rank " << rank_id << std::endl;
                
                // 接收所有口令
                /*for (int j = 0; j < guess_count; j++) {
                        if (j % 10000 == 0) {
                            std::cout << "[rank 0] Receiving guess " << j << "/" << guess_count 
                                    << " from rank " << rank_id << std::endl;}
                    int str_len;
                    MPI_Recv(&str_len, 1, MPI_INT, rank_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    char* buffer = new char[str_len + 1];
                    MPI_Recv(buffer, str_len, MPI_CHAR, rank_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    buffer[str_len] = '\0';
                    q.guesses.push_back(string(buffer));
                    delete[] buffer;
                    q.total_guesses++;
                }*/
                q.total_guesses += guess_count;
                
                // 接收新PT的数量
                int new_pt_count;
                MPI_Recv(&new_pt_count, 1, MPI_INT, rank_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // 接收所有新PT
                for (int j = 0; j < new_pt_count; j++) {
                    int str_size;
                    MPI_Recv(&str_size, 1, MPI_INT, rank_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    char* buffer = new char[str_size + 1];
                    MPI_Recv(buffer, str_size, MPI_CHAR,rank_id, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    buffer[str_size] = '\0';
                    
                    PT new_pt;
                    new_pt.deserialize(string(buffer));
                    delete[] buffer;
                    
                    // 重新计算概率并插入队列
                    q.CalProb(new_pt);
                    auto it = q.priority.begin();
                    while (it != q.priority.end() && it->prob > new_pt.prob) {
                        it++;
                    }
                    q.priority.insert(it, new_pt);
                }
            }
            
            // 更新统计信息
            curr_num = q.guesses.size();
            int total_generated = q.total_guesses;//history + curr_num;
            
            // 输出进度
            if (total_generated % 100000 == 0) {
                cout << "Guesses generated: " << total_generated << endl;
            }
            
            // 检查是否达到生成上限
            if (total_generated > 10000000) {
                
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                //cout << "Hash time: " << time_hash << " seconds" << endl;
                cout << "Train time: " << time_train << " seconds" << endl;
                break;
            }
            
            // 定期清空猜测队列
            /*if (curr_num > 1000000) {
                auto start_hash = system_clock::now();
                bit32 state[4];
                
                // 使用SIMD MD5哈希
                for (int i = 0; i < q.guesses.size(); i += 4) {
                    string batch[4];
                    bit32 states[4][4];
                
                    for (int j = 0; j < 4; ++j) {
                        batch[j] = (i + j < q.guesses.size()) ? q.guesses[i + j] : "";
                    }
                
                    MD5Hash_neon_parallel(batch, states);
                }
                
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }*/
        }
        
        // 发送结束信号给所有工作进程
        for (int i = 1; i < size; i++) {
            int end_signal = -1;
            MPI_Send(&end_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            //std::cout << "[rank 0] sent end_signal to rank " << i << std::endl;
        }
        
        // 最终统计

        
    } else {
        // 工作进程逻辑
        while (true) {
            int data_size;
            MPI_Status status;
            
            // 接收数据大小（标签0）
            MPI_Recv(&data_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            
            // 检查结束信号
            if (data_size == -1) 
            {
                //std::cout << "[rank " << rank << "] received end_signal, exiting." << std::endl;
                break;}
            
            // 接收PT数据（标签1）
            char* buffer = new char[data_size + 1];
            MPI_Recv(buffer, data_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &status);
            buffer[data_size] = '\0';
            std::string pt_str(buffer);
            delete[] buffer;
            
            // 重建PT对象
            PT pt;
            pt.deserialize(pt_str);
            
            // 生成口令
            vector<string> local_guesses;
            if (pt.content.size() == 1) {
                segment* a = nullptr;
                if (pt.content[0].type == 1) a = &q.m.letters[q.m.FindLetter(pt.content[0])];
                else if (pt.content[0].type == 2) a = &q.m.digits[q.m.FindDigit(pt.content[0])];
                else if (pt.content[0].type == 3) a = &q.m.symbols[q.m.FindSymbol(pt.content[0])];
                
                if (a) {
                    for (int i = 0; i < pt.max_indices[0]; i++) {
                        local_guesses.push_back(a->ordered_values[i]);
                    }
                }
            } else {
                string prefix;
                int seg_idx = 0;
                for (int idx : pt.curr_indices) {
                    if (pt.content[seg_idx].type == 1) 
                        prefix += q.m.letters[q.m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
                    else if (pt.content[seg_idx].type == 2)
                        prefix += q.m.digits[q.m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
                    else if (pt.content[seg_idx].type == 3)
                        prefix += q.m.symbols[q.m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
                    
                    if (++seg_idx == pt.content.size() - 1) break;
                }
                
                segment* a = nullptr;
                segment& last = pt.content.back();
                if (last.type == 1) a = &q.m.letters[q.m.FindLetter(last)];
                else if (last.type == 2) a = &q.m.digits[q.m.FindDigit(last)];
                else if (last.type == 3) a = &q.m.symbols[q.m.FindSymbol(last)];
                
                if (a) {
                    for (int i = 0; i < pt.max_indices.back(); i++) {
                        local_guesses.push_back(prefix + a->ordered_values[i]);
                    }
                }
            }
            
            // 发送生成的口令数量
            int guess_count = local_guesses.size();
            //std::cout << "[rank " << rank << "] generated " << local_guesses.size() << " guesses" << std::endl;

            MPI_Send(&guess_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            //std::cout << "[rank " << rank << "] sent guess_count to rank 0" << std::endl;
            // 发送所有口令
            /*for (const string& guess : local_guesses) {
                int len = guess.length();
                MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(guess.c_str(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }*/
            

            
                auto start_hash = system_clock::now();
                bit32 state[4];
                
                // 使用SIMD MD5哈希
                for (int i = 0; i < local_guesses.size(); i += 4) {
                    string batch[4];
                    bit32 states[4][4];
                
                    for (int j = 0; j < 4; ++j) {
                        batch[j] = (i + j <  local_guesses.size()) ?  local_guesses[i + j] : "";
                    }
                
                    MD5Hash_neon_parallel(batch, states);
                }
                
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
            std::cout<<"Hash time "<<time_hash<<std::endl;




            // 生成新PT并发送
            vector<PT> new_pts = pt.NewPTs();
            int new_pt_count = new_pts.size();
            MPI_Send(&new_pt_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            
            for (PT& new_pt : new_pts) {
                std::string new_pt_str = new_pt.serialize();
                int str_size = new_pt_str.size();
                MPI_Send(&str_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(new_pt_str.c_str(), str_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
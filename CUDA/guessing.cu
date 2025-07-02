#include "PCFG.h"
using namespace std;
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <thread>


void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}









































#define BATCH_SIZE 64  // 根据显存和任务量调整

//constexpr int MAX_PWD_LENGTH = 100;

struct GpuPTInfo {
    int prefix_len;
    int guess_offset; // output中起始位置
};

__global__ void kernelBatchGenerate(char *output, const char *prefixes, const int *prefix_lens,
                                    const char *suffixes, const int *suffix_lens,
                                    const int *pt_guess_offset, int total_guesses, int max_length, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_guesses) return;

    //查找当前idx所属的PT
    int pt_id = 0;
    while (pt_id < batch - 1 && idx >= pt_guess_offset[pt_id + 1]) {
        pt_id++;
    }
    //int local_idx = idx - pt_guess_offset[pt_id];
    int prefix_len = prefix_lens[pt_id];
    const char *prefix = prefixes + pt_id * max_length;

    // 直接使用全局索引访问后缀
    const char *suffix = suffixes + idx * max_length;
    int suffix_len = suffix_lens[idx];

    char *pwd = output + idx * max_length;
    // 复制前缀
    int pos = 0;
    for (int i = 0; i < prefix_len; ++i) {
        pwd[pos++] = prefix[i];
    }
    // 复制后缀
    for (int i = 0; i < suffix_len; ++i) {
        pwd[pos++] = suffix[i];
    }
    pwd[pos] = '\0';
}


/*void PriorityQueue::PopNext() {
    // 批量收集PT
    int batch = std::min(BATCH_SIZE, (int)priority.size());
    std::vector<PT> batch_pts(priority.begin(), priority.begin() + batch);

    // 统计总猜测数
    int total_guesses = 0;
    std::vector<int> pt_guess_offset(batch + 1, 0);
    for (size_t i = 0; i < batch_pts.size(); ++i) {
        int n = (batch_pts[i].content.size() == 1)
            ? batch_pts[i].max_indices[0]
            : batch_pts[i].max_indices[batch_pts[i].content.size() - 1];
        total_guesses += n;
        pt_guess_offset[i + 1] = total_guesses;
    }

    // 确保GPU内存足够
    EnsureGPUMemory(batch, total_guesses);

    // 准备主机缓冲区
    std::vector<char> h_prefixes(batch * MAX_PWD_LENGTH, 0);
    std::vector<int> h_prefix_lens(batch, 0);
    std::vector<char> h_suffixes(total_guesses * MAX_PWD_LENGTH, 0);
    std::vector<int> h_suffix_lens(total_guesses, 0);

    // 填充数据
    for (size_t i = 0; i < batch_pts.size(); ++i) {
        const PT &pt = batch_pts[i];
        std::string prefix = "";

        // 构建前缀（多段PT）
        if (pt.content.size() > 1) {
            for (size_t j = 0; j < pt.content.size() - 1; ++j) {
                int idx_val = pt.curr_indices[j];
                const segment &seg = pt.content[j];
                if (seg.type == 1) prefix += m.letters[m.FindLetter(seg)].ordered_values[idx_val];
                else if (seg.type == 2) prefix += m.digits[m.FindDigit(seg)].ordered_values[idx_val];
                else if (seg.type == 3) prefix += m.symbols[m.FindSymbol(seg)].ordered_values[idx_val];
            }
        }

        // 处理前缀
        h_prefix_lens[i] = prefix.length();
        if (!prefix.empty()) {
            memcpy(&h_prefixes[i * MAX_PWD_LENGTH], prefix.c_str(), prefix.length());
        }
        // 处理后缀
        segment *last_seg = nullptr;
        const segment &last = pt.content.back();
        int n = (pt.content.size() == 1) ? pt.max_indices[0] : pt.max_indices.back();
        
        if (last.type == 1) last_seg = &m.letters[m.FindLetter(last)];
        else if (last.type == 2) last_seg = &m.digits[m.FindDigit(last)];
        else if (last.type == 3) last_seg = &m.symbols[m.FindSymbol(last)];
        
        int start_idx = pt_guess_offset[i];
        for (int j = 0; j < n; ++j) {
            std::string val = last_seg->ordered_values[j];
            int global_idx = start_idx + j;
            h_suffix_lens[global_idx] = val.length();
            if (!val.empty()) {
                memcpy(&h_suffixes[global_idx * MAX_PWD_LENGTH], val.c_str(), val.length());
            }
        }
    }

    // GPU内存分配和拷贝
    char *d_output, *d_prefixes, *d_suffixes;
    int *d_prefix_lens, *d_suffix_lens, *d_pt_offsets;

    
    cudaMemcpy(d_prefixes, h_prefixes.data(), batch * MAX_PWD_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_lens, h_prefix_lens.data(), batch * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffixes, h_suffixes.data(), total_guesses * MAX_PWD_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffix_lens, h_suffix_lens.data(), total_guesses * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt_offsets, pt_guess_offset.data(), (batch + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // [6] 启动核函数
    int blockSize = 256;
    int gridSize = (total_guesses + blockSize - 1) / blockSize;
    kernelBatchGenerate<<<gridSize, blockSize>>>(
        d_output, d_prefixes, d_prefix_lens,
        d_suffixes, d_suffix_lens, d_pt_offsets,
        total_guesses, MAX_PWD_LENGTH, batch  // 添加batch参数
    );

    // [7] 复制结果回主机
    std::vector<char> h_output(total_guesses * MAX_PWD_LENGTH);
    cudaMemcpy(h_output.data(), d_output, total_guesses * MAX_PWD_LENGTH, cudaMemcpyDeviceToHost);

    // [8] 添加猜测到队列（修复双重计数）
    for (int i = 0; i < total_guesses; ++i) {
        guesses.emplace_back(h_output.data() + i * MAX_PWD_LENGTH);
    }
    this->total_guesses += total_guesses;  // 正确更新总计数


    // 处理优先队列和新PT
    for (int i = 0; i < batch; ++i) {
        vector<PT> new_pts = priority[i].NewPTs();
        for (PT pt : new_pts) {
            CalProb(pt);
            // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
            for (auto iter = priority.begin(); iter != priority.end(); iter++)
            {
                // 对于非队首和队尾的特殊情况
                if (iter != priority.end() - 1 && iter != priority.begin())
                {
                    // 判定概率
                    if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                    {
                        priority.emplace(iter + 1, pt);
                        break;
                    }
                }
                if (iter == priority.end() - 1)
                {
                    priority.emplace_back(pt);
                    break;
                }
                if (iter == priority.begin() && iter->prob < pt.prob)
                {
                    priority.emplace(iter, pt);
                    break;
                }
            }
        }
    }
    priority.erase(priority.begin(), priority.begin() + batch);
}
*/
void PriorityQueue::PopNext() {
    int batch = std::min(BATCH_SIZE, static_cast<int>(priority.size()));
    std::vector<PT> batch_pts(priority.begin(), priority.begin() + batch);

    // 计算总猜测数 + 偏移
    int total_guesses = 0;
    std::vector<int> pt_guess_offset(batch + 1, 0);
    for (size_t i = 0; i < batch_pts.size(); ++i) {
        int n = (batch_pts[i].content.size() == 1)
                    ? batch_pts[i].max_indices[0]
                    : batch_pts[i].max_indices.back();
        total_guesses += n;
        pt_guess_offset[i + 1] = total_guesses;
    }

    EnsureGPUMemory(batch, total_guesses);

    std::vector<char> h_prefixes(batch * MAX_PWD_LENGTH, 0);
    std::vector<int> h_prefix_lens(batch, 0);
    std::vector<char> h_suffixes(total_guesses * MAX_PWD_LENGTH, 0);
    std::vector<int> h_suffix_lens(total_guesses, 0);

    for (size_t i = 0; i < batch_pts.size(); ++i) {
        const PT &pt = batch_pts[i];
        std::string prefix = "";

        if (pt.content.size() > 1) {
            for (size_t j = 0; j < pt.content.size() - 1; ++j) {
                int idx_val = pt.curr_indices[j];
                const segment &seg = pt.content[j];
                if (seg.type == 1) prefix += m.letters[m.FindLetter(seg)].ordered_values[idx_val];
                else if (seg.type == 2) prefix += m.digits[m.FindDigit(seg)].ordered_values[idx_val];
                else if (seg.type == 3) prefix += m.symbols[m.FindSymbol(seg)].ordered_values[idx_val];
            }
        }

        h_prefix_lens[i] = prefix.length();
        if (!prefix.empty()) {
            memcpy(&h_prefixes[i * MAX_PWD_LENGTH], prefix.c_str(), prefix.length());
        }

        segment *last_seg = nullptr;
        const segment &last = pt.content.back();
        int n = (pt.content.size() == 1) ? pt.max_indices[0] : pt.max_indices.back();
        if (last.type == 1) last_seg = &m.letters[m.FindLetter(last)];
        else if (last.type == 2) last_seg = &m.digits[m.FindDigit(last)];
        else if (last.type == 3) last_seg = &m.symbols[m.FindSymbol(last)];

        int start_idx = pt_guess_offset[i];
        for (int j = 0; j < n; ++j) {
            std::string val = last_seg->ordered_values[j];
            int global_idx = start_idx + j;
            h_suffix_lens[global_idx] = val.length();
            if (!val.empty()) {
                memcpy(&h_suffixes[global_idx * MAX_PWD_LENGTH], val.c_str(), val.length());
            }
        }
    }

    // ==== GPU拷贝 & 异步核函数 ====
    char *d_output, *d_prefixes, *d_suffixes;
    int *d_prefix_lens, *d_suffix_lens, *d_pt_offsets;

    cudaMemcpy(d_prefixes, h_prefixes.data(), batch * MAX_PWD_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_lens, h_prefix_lens.data(), batch * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffixes, h_suffixes.data(), total_guesses * MAX_PWD_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffix_lens, h_suffix_lens.data(), total_guesses * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt_offsets, pt_guess_offset.data(), (batch + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int blockSize = 256;
    int gridSize = (total_guesses + blockSize - 1) / blockSize;
    kernelBatchGenerate<<<gridSize, blockSize, 0, stream>>>(
        d_output, d_prefixes, d_prefix_lens,
        d_suffixes, d_suffix_lens, d_pt_offsets,
        total_guesses, MAX_PWD_LENGTH, batch
    );

    // ==== CPU并行处理新PT ====
    std::thread cpu_thread([&]() {
        for (int i = 0; i < batch; ++i) {
            std::vector<PT> new_pts = priority[i].NewPTs();
            for (PT &pt : new_pts) {
                CalProb(pt);
                InsertByProb(pt);  // 👇封装插入逻辑为函数
            }
        }
    });

    // ==== 等待GPU完成 ====
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    std::vector<char> h_output(total_guesses * MAX_PWD_LENGTH);
    cudaMemcpy(h_output.data(), d_output, total_guesses * MAX_PWD_LENGTH, cudaMemcpyDeviceToHost);

    for (int i = 0; i < total_guesses; ++i) {
        guesses.emplace_back(h_output.data() + i * MAX_PWD_LENGTH);
    }
    this->total_guesses += total_guesses;

    // 等待CPU线程
    cpu_thread.join();

    // 清理旧的 batch
    priority.erase(priority.begin(), priority.begin() + batch);
}

void PriorityQueue::InsertByProb(const PT &pt) {
    for (auto iter = priority.begin(); iter != priority.end(); iter++) {
        if (iter != priority.begin() && iter != priority.end() - 1) {
            if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                priority.emplace(iter + 1, pt);
                return;
            }
        }
        if (iter == priority.end() - 1) {
            priority.emplace_back(pt);
            return;
        }
        if (iter == priority.begin() && iter->prob > iter->prob) {
            priority.emplace(iter, pt);
            return;
        }
    }
}





































/*void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}*/

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}


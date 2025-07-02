
/*// 定义最大密码长度
constexpr int MAX_PWD_LENGTH = 100;

//CUDA内核：处理单段PT
__global__ void kernelSingleSegment(char* output, const char* segment_values, 
                                   const int* lengths, int num_guesses, 
                                   int max_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_guesses) return;
    
    char* pwd_ptr = output + idx * max_length;
    const char* src_ptr = segment_values + (idx == 0 ? 0 : lengths[idx-1]);
    int len = lengths[idx] - (idx == 0 ? 0 : lengths[idx-1]);
    
    for (int i = 0; i < len; i++) {
        pwd_ptr[i] = src_ptr[i];
    }
    pwd_ptr[len] = '\0';
}

//CUDA内核：处理多段PT
__global__ void kernelMultiSegment(char* output, const char* prefix, 
                                  const char* suffix_values, 
                                  const int* lengths, int num_guesses, 
                                  int prefix_len, int max_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_guesses) return;
    
    char* pwd_ptr = output + idx * max_length;
    
    // 复制前缀
    for (int i = 0; i < prefix_len; i++) {
        pwd_ptr[i] = prefix[i];
    }
    
    // 添加后缀
    const char* src_ptr = suffix_values + (idx == 0 ? 0 : lengths[idx-1]);
    int suffix_len = lengths[idx] - (idx == 0 ? 0 : lengths[idx-1]);
    
    for (int i = 0; i < suffix_len; i++) {
        pwd_ptr[prefix_len + i] = src_ptr[i];
    }
    pwd_ptr[prefix_len + suffix_len] = '\0';
}

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a = nullptr;
        // 获取 segment 数据 [与原始代码相同]
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
        

        int num_guesses = pt.max_indices[0];
        if (num_guesses == 0) return;

        // 准备展平数据
        std::vector<int> lengths(num_guesses);
        std::vector<char> flat_values;
        int total_size = 0;
        
        for (int i = 0; i < num_guesses; i++) {
            const std::string& val = a->ordered_values[i];
            lengths[i] = total_size + val.size();
            total_size += val.size();
            flat_values.insert(flat_values.end(), val.begin(), val.end());
        }

        // 设备内存分配
        char* d_output;
        char* d_values;
        int* d_lengths;
        
        cudaMalloc(&d_output, num_guesses * MAX_PWD_LENGTH * sizeof(char));
        cudaMalloc(&d_values, flat_values.size() * sizeof(char));
        cudaMalloc(&d_lengths, num_guesses * sizeof(int));
        
        // 复制数据到设备
        cudaMemcpy(d_values, flat_values.data(), flat_values.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lengths, lengths.data(), num_guesses * sizeof(int), cudaMemcpyHostToDevice);

        // 启动内核
        int blockSize = 256;
        int gridSize = (num_guesses + blockSize - 1) / blockSize;
        kernelSingleSegment<<<gridSize, blockSize>>>(d_output, d_values, d_lengths, 
                                                    num_guesses, MAX_PWD_LENGTH);
        
        // 复制结果回主机
        std::vector<char> host_output(num_guesses * MAX_PWD_LENGTH);
        cudaMemcpy(host_output.data(), d_output, 
                  num_guesses * MAX_PWD_LENGTH, cudaMemcpyDeviceToHost);
        
        // 处理结果
        for (int i = 0; i < num_guesses; i++) {
            guesses.emplace_back(host_output.data() + i * MAX_PWD_LENGTH);
            total_guesses++;
        }

        // 清理设备内存
        cudaFree(d_output);
        cudaFree(d_values);
        cudaFree(d_lengths);
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

        int num_guesses = pt.max_indices[pt.content.size() - 1];
        if (num_guesses == 0) return;

        // 准备展平后缀数据
        std::vector<int> lengths(num_guesses);
        std::vector<char> flat_values;
        int total_size = 0;
        
        for (int i = 0; i < num_guesses; i++) {
            const std::string& val = a->ordered_values[i];
            lengths[i] = total_size + val.size();
            total_size += val.size();
            flat_values.insert(flat_values.end(), val.begin(), val.end());
        }

        // 设备内存分配
        char* d_output;
        char* d_prefix;
        char* d_values;
        int* d_lengths;
        
        cudaMalloc(&d_output, num_guesses * MAX_PWD_LENGTH * sizeof(char));
        cudaMalloc(&d_prefix, guess.size() + 1);
        cudaMalloc(&d_values, flat_values.size() * sizeof(char));
        cudaMalloc(&d_lengths, num_guesses * sizeof(int));
        
        // 复制数据到设备
        cudaMemcpy(d_prefix, guess.c_str(), guess.size() + 1, cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, flat_values.data(), flat_values.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lengths, lengths.data(), num_guesses * sizeof(int), cudaMemcpyHostToDevice);

        // 启动内核
        int blockSize = 256;
        int gridSize = (num_guesses + blockSize - 1) / blockSize;
        kernelMultiSegment<<<gridSize, blockSize>>>(d_output, d_prefix, d_values, d_lengths, 
                                                  num_guesses, guess.size(), MAX_PWD_LENGTH);
        
        // 复制结果回主机
        std::vector<char> host_output(num_guesses * MAX_PWD_LENGTH);
        cudaMemcpy(host_output.data(), d_output, 
                  num_guesses * MAX_PWD_LENGTH, cudaMemcpyDeviceToHost);
        
        // 处理结果
        for (int i = 0; i < num_guesses; i++) {
            guesses.emplace_back(host_output.data() + i * MAX_PWD_LENGTH);
            total_guesses++;
        }

        // 清理设备内存
        cudaFree(d_output);
        cudaFree(d_prefix);
        cudaFree(d_values);
        cudaFree(d_lengths);
    }
}*/

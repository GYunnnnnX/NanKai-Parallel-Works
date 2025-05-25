#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>

using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcess(string input, int *n_byte)
{
	// 将输入的字符串转换为Byte为单位的数组
	Byte *blocks = (Byte *)input.c_str();
	int length = input.length();

	// 计算原始消息长度（以比特为单位）
	int bitLength = length * 8;

	// paddingBits: 原始消息需要的padding长度（以bit为单位）
	// 对于给定的消息，将其补齐至length%512==448为止
	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
	int paddingBits = bitLength % 512;
	if (paddingBits > 448)
	{
		paddingBits = 512 - (paddingBits - 448);
	}
	else if (paddingBits < 448)
	{
		paddingBits = 448 - paddingBits;
	}
	else if (paddingBits == 448)
	{
		paddingBits = 512;
	}

	// 原始消息需要的padding长度（以Byte为单位）
	int paddingBytes = paddingBits / 8;
	// 创建最终的字节数组
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte *paddedMessage = new Byte[paddedLength];

	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

	// 添加消息长度（64比特，小端格式）
	for (int i = 0; i < 8; ++i)
	{
		// 特别注意此处应当将bitLength转换为uint64_t
		// 这里的length是原始消息的长度
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;
	return paddedMessage;
}


/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash(string input, bit32 *state)
{

	Byte *paddedMessage;
	int *messageLength = new int[1];
	for (int i = 0; i < 1; i += 1)
	{
		paddedMessage = StringProcess(input, &messageLength[i]);
		// cout<<messageLength[i]<<endl;
		assert(messageLength[i] == messageLength[0]);
	}
	int n_blocks = messageLength[0] / 64;

	// bit32* state= new bit32[4];
	state[0] = 0x67452301;
	state[1] = 0xefcdab89;
	state[2] = 0x98badcfe;
	state[3] = 0x10325476;

	// 逐block地更新state
	for (int i = 0; i < n_blocks; i += 1)
	{
		bit32 x[16];

		// 下面的处理，在理解上较为复杂
		for (int i1 = 0; i1 < 16; ++i1)
		{
			x[i1] = (paddedMessage[4 * i1 + i * 64]) |
					(paddedMessage[4 * i1 + 1 + i * 64] << 8) |
					(paddedMessage[4 * i1 + 2 + i * 64] << 16) |
					(paddedMessage[4 * i1 + 3 + i * 64] << 24);
		}

		bit32 a = state[0], b = state[1], c = state[2], d = state[3];

		auto start = system_clock::now();
		/* Round 1 */
		FF(a, b, c, d, x[0], s11, 0xd76aa478);
		FF(d, a, b, c, x[1], s12, 0xe8c7b756);
		FF(c, d, a, b, x[2], s13, 0x242070db);
		FF(b, c, d, a, x[3], s14, 0xc1bdceee);
		FF(a, b, c, d, x[4], s11, 0xf57c0faf);
		FF(d, a, b, c, x[5], s12, 0x4787c62a);
		FF(c, d, a, b, x[6], s13, 0xa8304613);
		FF(b, c, d, a, x[7], s14, 0xfd469501);
		FF(a, b, c, d, x[8], s11, 0x698098d8);
		FF(d, a, b, c, x[9], s12, 0x8b44f7af);
		FF(c, d, a, b, x[10], s13, 0xffff5bb1);
		FF(b, c, d, a, x[11], s14, 0x895cd7be);
		FF(a, b, c, d, x[12], s11, 0x6b901122);
		FF(d, a, b, c, x[13], s12, 0xfd987193);
		FF(c, d, a, b, x[14], s13, 0xa679438e);
		FF(b, c, d, a, x[15], s14, 0x49b40821);

		/* Round 2 */
		GG(a, b, c, d, x[1], s21, 0xf61e2562);
		GG(d, a, b, c, x[6], s22, 0xc040b340);
		GG(c, d, a, b, x[11], s23, 0x265e5a51);
		GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GG(a, b, c, d, x[5], s21, 0xd62f105d);
		GG(d, a, b, c, x[10], s22, 0x2441453);
		GG(c, d, a, b, x[15], s23, 0xd8a1e681);
		GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GG(a, b, c, d, x[9], s21, 0x21e1cde6);
		GG(d, a, b, c, x[14], s22, 0xc33707d6);
		GG(c, d, a, b, x[3], s23, 0xf4d50d87);
		GG(b, c, d, a, x[8], s24, 0x455a14ed);
		GG(a, b, c, d, x[13], s21, 0xa9e3e905);
		GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GG(c, d, a, b, x[7], s23, 0x676f02d9);
		GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		/* Round 3 */
		HH(a, b, c, d, x[5], s31, 0xfffa3942);
		HH(d, a, b, c, x[8], s32, 0x8771f681);
		HH(c, d, a, b, x[11], s33, 0x6d9d6122);
		HH(b, c, d, a, x[14], s34, 0xfde5380c);
		HH(a, b, c, d, x[1], s31, 0xa4beea44);
		HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HH(b, c, d, a, x[10], s34, 0xbebfbc70);
		HH(a, b, c, d, x[13], s31, 0x289b7ec6);
		HH(d, a, b, c, x[0], s32, 0xeaa127fa);
		HH(c, d, a, b, x[3], s33, 0xd4ef3085);
		HH(b, c, d, a, x[6], s34, 0x4881d05);
		HH(a, b, c, d, x[9], s31, 0xd9d4d039);
		HH(d, a, b, c, x[12], s32, 0xe6db99e5);
		HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HH(b, c, d, a, x[2], s34, 0xc4ac5665);

		/* Round 4 */
		II(a, b, c, d, x[0], s41, 0xf4292244);
		II(d, a, b, c, x[7], s42, 0x432aff97);
		II(c, d, a, b, x[14], s43, 0xab9423a7);
		II(b, c, d, a, x[5], s44, 0xfc93a039);
		II(a, b, c, d, x[12], s41, 0x655b59c3);
		II(d, a, b, c, x[3], s42, 0x8f0ccc92);
		II(c, d, a, b, x[10], s43, 0xffeff47d);
		II(b, c, d, a, x[1], s44, 0x85845dd1);
		II(a, b, c, d, x[8], s41, 0x6fa87e4f);
		II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		II(c, d, a, b, x[6], s43, 0xa3014314);
		II(b, c, d, a, x[13], s44, 0x4e0811a1);
		II(a, b, c, d, x[4], s41, 0xf7537e82);
		II(d, a, b, c, x[11], s42, 0xbd3af235);
		II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		II(b, c, d, a, x[9], s44, 0xeb86d391);

		state[0] += a;
		state[1] += b;
		state[2] += c;
		state[3] += d;
	}

	// 下面的处理，在理解上较为复杂
	for (int i = 0; i < 4; i++)
	{
		uint32_t value = state[i];
		state[i] = ((value & 0xff) << 24) |		 // 将最低字节移到最高位
				   ((value & 0xff00) << 8) |	 // 将次低字节左移
				   ((value & 0xff0000) >> 8) |	 // 将次高字节右移
				   ((value & 0xff000000) >> 24); // 将最高字节移到最低位
	}

	// 输出最终的hash结果
	// for (int i1 = 0; i1 < 4; i1 += 1)
	// {
	// 	cout << std::setw(8) << std::setfill('0') << hex << state[i1];
	// }
	// cout << endl;

	// 释放动态分配的内存
	// 实现SIMD并行算法的时候，也请记得及时回收内存！
	delete[] paddedMessage;
	delete[] messageLength;
}



//neon并行化后的MD5函数，处理四个输入
void MD5Hash_neon_parallel(string inputs[4], bit32 states[4][4]) {
    //首先，需要将输入改成四个消息的数组input[4],存储哈希值的states也要开成[4][4]大小的，一次性输出多个消息的哈希值。
    Byte* paddedMessages[4];
	int *messageLength = new int[4];
    int max_blocks = 0;//获得四个消息中，分割后的块数
    //其实并行处理要求消息数是一样的，否则达不到应有的效率和运行正确性

	//同时处理四个消息的填充
    for (int i = 0; i < 4; i++) {
		paddedMessages[i] = StringProcess(inputs[i], &messageLength[i]);
		int blocks = messageLength[i] / 64;
		if (blocks > max_blocks) max_blocks = blocks;
	}

    // 初始化四个并行状态（每个通道处理一个消息）
    uint32x4_t state[4] = {
        vdupq_n_u32(0x67452301),  // A0-A3
        vdupq_n_u32(0xefcdab89),   // B0-B3
        vdupq_n_u32(0x98badcfe),   // C0-C3
        vdupq_n_u32(0x10325476)    // D0-D3
    };

    // 并行处理所有块
    for (int i = 0; i < max_blocks; i++) {
        uint32x4_t x[16];

        // 加载四个消息的当前块到SIMD寄存器
        for (int i1 = 0; i1 < 16; i1++) {
            uint32_t words[4] = {0};

            for (int msg = 0; msg < 4; msg++) {
                if (i < messageLength[msg] / 64) {
                    Byte* block_ptr = paddedMessages[msg] + i * 64;
                    words[msg] = (block_ptr[i1*4 + 0])      |
                                (block_ptr[i1*4 + 1] << 8)  |
                                (block_ptr[i1*4 + 2] << 16) |
                                (block_ptr[i1*4 + 3] << 24);
                }
            }
            //x[0...15]中16个寄存器每个寄存器 (uint32x4\_t) 包含4个32位字,用来并行处理四条消息。
            x[i1] = vld1q_u32(words);
        }

        // 初始化工作变量
        uint32x4_t a = state[0];
        uint32x4_t b = state[1];
        uint32x4_t c = state[2];
        uint32x4_t d = state[3];

        /* Round 1 - 四路并行计算 */
        FF_neon(a, b, c, d, x[0], s11, vdupq_n_u32(0xd76aa478));
        FF_neon(d, a, b, c, x[1], s12, vdupq_n_u32(0xe8c7b756));
        FF_neon(c, d, a, b, x[2], s13, vdupq_n_u32(0x242070db));
        FF_neon(b, c, d, a, x[3], s14, vdupq_n_u32(0xc1bdceee));
        FF_neon(a, b, c, d, x[4], s11, vdupq_n_u32(0xf57c0faf));
        FF_neon(d, a, b, c, x[5], s12, vdupq_n_u32(0x4787c62a));
        FF_neon(c, d, a, b, x[6], s13, vdupq_n_u32(0xa8304613));
        FF_neon(b, c, d, a, x[7], s14, vdupq_n_u32(0xfd469501));
        FF_neon(a, b, c, d, x[8], s11, vdupq_n_u32(0x698098d8));
        FF_neon(d, a, b, c, x[9], s12, vdupq_n_u32(0x8b44f7af));
        FF_neon(c, d, a, b, x[10], s13, vdupq_n_u32(0xffff5bb1));
        FF_neon(b, c, d, a, x[11], s14, vdupq_n_u32(0x895cd7be));
        FF_neon(a, b, c, d, x[12], s11, vdupq_n_u32(0x6b901122));
        FF_neon(d, a, b, c, x[13], s12, vdupq_n_u32(0xfd987193));
        FF_neon(c, d, a, b, x[14], s13, vdupq_n_u32(0xa679438e));
        FF_neon(b, c, d, a, x[15], s14, vdupq_n_u32(0x49b40821));

        /* Round 2 */
        GG_neon(a, b, c, d, x[1], s21, vdupq_n_u32(0xf61e2562));
        GG_neon(d, a, b, c, x[6], s22, vdupq_n_u32(0xc040b340));
        GG_neon(c, d, a, b, x[11], s23, vdupq_n_u32(0x265e5a51));
        GG_neon(b, c, d, a, x[0], s24, vdupq_n_u32(0xe9b6c7aa));
        GG_neon(a, b, c, d, x[5], s21, vdupq_n_u32(0xd62f105d));
        GG_neon(d, a, b, c, x[10], s22, vdupq_n_u32(0x02441453));
        GG_neon(c, d, a, b, x[15], s23, vdupq_n_u32(0xd8a1e681));
        GG_neon(b, c, d, a, x[4], s24, vdupq_n_u32(0xe7d3fbc8));
        GG_neon(a, b, c, d, x[9], s21, vdupq_n_u32(0x21e1cde6));
        GG_neon(d, a, b, c, x[14], s22, vdupq_n_u32(0xc33707d6));
        GG_neon(c, d, a, b, x[3], s23, vdupq_n_u32(0xf4d50d87));
        GG_neon(b, c, d, a, x[8], s24, vdupq_n_u32(0x455a14ed));
        GG_neon(a, b, c, d, x[13], s21, vdupq_n_u32(0xa9e3e905));
        GG_neon(d, a, b, c, x[2], s22, vdupq_n_u32(0xfcefa3f8));
        GG_neon(c, d, a, b, x[7], s23, vdupq_n_u32(0x676f02d9));
        GG_neon(b, c, d, a, x[12], s24, vdupq_n_u32(0x8d2a4c8a));

        /* Round 3 */
        HH_neon(a, b, c, d, x[5], s31, vdupq_n_u32(0xfffa3942));
        HH_neon(d, a, b, c, x[8], s32, vdupq_n_u32(0x8771f681));
        HH_neon(c, d, a, b, x[11], s33, vdupq_n_u32(0x6d9d6122));
        HH_neon(b, c, d, a, x[14], s34, vdupq_n_u32(0xfde5380c));
        HH_neon(a, b, c, d, x[1], s31, vdupq_n_u32(0xa4beea44));
        HH_neon(d, a, b, c, x[4], s32, vdupq_n_u32(0x4bdecfa9));
        HH_neon(c, d, a, b, x[7], s33, vdupq_n_u32(0xf6bb4b60));
        HH_neon(b, c, d, a, x[10], s34, vdupq_n_u32(0xbebfbc70));
        HH_neon(a, b, c, d, x[13], s31, vdupq_n_u32(0x289b7ec6));
        HH_neon(d, a, b, c, x[0], s32, vdupq_n_u32(0xeaa127fa));
        HH_neon(c, d, a, b, x[3], s33, vdupq_n_u32(0xd4ef3085));
        HH_neon(b, c, d, a, x[6], s34, vdupq_n_u32(0x04881d05));
        HH_neon(a, b, c, d, x[9], s31, vdupq_n_u32(0xd9d4d039));
        HH_neon(d, a, b, c, x[12], s32, vdupq_n_u32(0xe6db99e5));
        HH_neon(c, d, a, b, x[15], s33, vdupq_n_u32(0x1fa27cf8));
        HH_neon(b, c, d, a, x[2], s34, vdupq_n_u32(0xc4ac5665));

        /* Round 4 */
        II_neon(a, b, c, d, x[0], s41, vdupq_n_u32(0xf4292244));
        II_neon(d, a, b, c, x[7], s42, vdupq_n_u32(0x432aff97));
        II_neon(c, d, a, b, x[14], s43, vdupq_n_u32(0xab9423a7));
        II_neon(b, c, d, a, x[5], s44, vdupq_n_u32(0xfc93a039));
        II_neon(a, b, c, d, x[12], s41, vdupq_n_u32(0x655b59c3));
        II_neon(d, a, b, c, x[3], s42, vdupq_n_u32(0x8f0ccc92));
        II_neon(c, d, a, b, x[10], s43, vdupq_n_u32(0xffeff47d));
        II_neon(b, c, d, a, x[1], s44, vdupq_n_u32(0x85845dd1));
        II_neon(a, b, c, d, x[8], s41, vdupq_n_u32(0x6fa87e4f));
        II_neon(d, a, b, c, x[15], s42, vdupq_n_u32(0xfe2ce6e0));
        II_neon(c, d, a, b, x[6], s43, vdupq_n_u32(0xa3014314));
        II_neon(b, c, d, a, x[13], s44, vdupq_n_u32(0x4e0811a1));
        II_neon(a, b, c, d, x[4], s41, vdupq_n_u32(0xf7537e82));
        II_neon(d, a, b, c, x[11], s42, vdupq_n_u32(0xbd3af235));
        II_neon(c, d, a, b, x[2], s43, vdupq_n_u32(0x2ad7d2bb));
        II_neon(b, c, d, a, x[9], s44, vdupq_n_u32(0xeb86d391));

        // 更新四个并行状态
        state[0] = vaddq_u32(state[0], a);
        state[1] = vaddq_u32(state[1], b);
        state[2] = vaddq_u32(state[2], c);
        state[3] = vaddq_u32(state[3], d);
    }

    // 提取并处理四个结果
	 for (int msg = 0; msg < 4; msg++) {
        // 正确提取每个消息的最终状态
		uint32_t final_state[4];
        //使用vgetq\_lane\_u32（参考Neon功能文档）将state寄存器中的向量提取出来，存放在final\_state[4]中
		switch (msg) {
			case 0:
				final_state[0] = vgetq_lane_u32(state[0], 0); // A0
				final_state[1] = vgetq_lane_u32(state[1], 0); // B0
				final_state[2] = vgetq_lane_u32(state[2], 0); // C0
				final_state[3] = vgetq_lane_u32(state[3], 0); // D0
				break;
			case 1:
				final_state[0] = vgetq_lane_u32(state[0], 1); // A1
				final_state[1] = vgetq_lane_u32(state[1], 1); // B1
				final_state[2] = vgetq_lane_u32(state[2], 1); // C1
				final_state[3] = vgetq_lane_u32(state[3], 1); // D1
				break;
			case 2:
				final_state[0] = vgetq_lane_u32(state[0], 2); // A2
				final_state[1] = vgetq_lane_u32(state[1], 2); // B2
				final_state[2] = vgetq_lane_u32(state[2], 2); // C2
				final_state[3] = vgetq_lane_u32(state[3], 2); // D2
				break;
			case 3:
				final_state[0] = vgetq_lane_u32(state[0], 3); // A3
				final_state[1] = vgetq_lane_u32(state[1], 3); // B3
				final_state[2] = vgetq_lane_u32(state[2], 3); // C3
				final_state[3] = vgetq_lane_u32(state[3], 3); // D3
				break;
		}
            
        // 字节序转换，大端序小端序
        for (int i = 0; i < 4; i++) {
            final_state[i] = ((final_state[i] & 0xff) << 24) |
                           ((final_state[i] & 0xff00) << 8)  |
                           ((final_state[i] & 0xff0000) >> 8) |
                           ((final_state[i] & 0xff000000) >> 24);
        }

        // 存储结果
        memcpy(states[msg], final_state, sizeof(final_state));
    }

    // 清理内存
    for (int i = 0; i < 4; i++) {
        delete[] paddedMessages[i];
    }
    delete[] messageLength;
}
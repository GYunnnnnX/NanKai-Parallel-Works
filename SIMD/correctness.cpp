#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o test.exe


// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    bit32 state[4];
    bit32 state_neon[4][4];
    //四个消息
   // string myinput1 ="bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva";
    string myinput1 = "correctness";
    string myinput2 = "guoyunxuansixsixsix";
    string myinput3 = "ilovethis";
    string myinput4 = "openeuler";
    string inputs[4] = {myinput1, myinput2, myinput3, myinput4};
    //串行结果输出
    for (int i = 0; i < 4; i += 1)
    {
        cout << "Hash " << i << ": ";
        MD5Hash(inputs[i], state);
        for(int i1 = 0;i1<4;i1++){
            cout << std::setw(8) << std::setfill('0') << hex << state[i1];
        }
        cout << endl;
    }
    cout<<endl;
    //并行结果输出
    MD5Hash_neon_parallel(inputs, state_neon);
    for (int i = 0; i < 4; i++) {
        cout << "Neon Hash " << i << ": ";
        for (int i1 = 0; i1 < 4; i1++) {
            cout << setw(8) << setfill('0') << hex << state_neon[i][i1];
        }
        cout << endl;
    }

    cout << endl;
}
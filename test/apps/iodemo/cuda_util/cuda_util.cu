#include <stdio.h>
#include <assert.h>
#include <iostream>
#include "cuda_util.h"


/**
 * Linear congruential generator (LCG):
 * n[i + 1] = (n[i] * A + C) % M
 * where A, C, M used as in glibc
 */

 IoDemoRandom & IoDemoRandom::get() {
    static IoDemoRandom instance;
    return instance;
}
    void IoDemoRandom::srand(unsigned seed) {
        _seed = seed & _M;
    }

    void IoDemoRandom::fill(unsigned &seed, void *buffer, size_t size, bool flag) {
        size_t body_count = size / sizeof(uint64_t);
        size_t tail_count = size & (sizeof(uint64_t) - 1);
        uint64_t *body    = reinterpret_cast<uint64_t*>(buffer);
        uint8_t *tail     = reinterpret_cast<uint8_t*>(body + body_count);

        fill(seed, body, body_count, flag);
        fill(seed, tail, tail_count, flag);
    }

    size_t IoDemoRandom::validate(unsigned &seed, const void *buffer,
                                  size_t size) {
        size_t body_count    = size / sizeof(uint64_t);
        size_t tail_count    = size & (sizeof(uint64_t) - 1);
        const uint64_t *body = reinterpret_cast<const uint64_t*>(buffer);
        const uint8_t *tail  = reinterpret_cast<const uint8_t*>(body + body_count);

        size_t err_pos = validate(seed, body, body_count);
        if (err_pos < body_count) {
            return err_pos * sizeof(body[0]);
        }

        err_pos = validate(seed, tail, tail_count);
        if (err_pos < tail_count) {
            return (body_count * sizeof(body[0])) + (err_pos * sizeof(tail[0]));
        }

        return size;
    }

    void IoDemoRandom::setMemoryType(ucs_memory_type_t memory_type)
    {
        _memory_type = memory_type;
    }

    template <typename T>
    void IoDemoRandom::fill(unsigned &seed, T *buffer, size_t count, bool flag) {
// #ifdef HAVE_CUDA
//         T temp;
// #endif

    unsigned *seed2;
    cudaMalloc(&seed2, sizeof(unsigned));
    // printf("LEO seed before: %d (count=%d)\n", seed, count);
    cudaMemcpy(seed2, &seed, sizeof(unsigned), cudaMemcpyDefault);
    
    ucs_memory_type_t mt = flag ? _memory_type : UCS_MEMORY_TYPE_HOST;
    for (size_t i = 0; i < count; ++i) {
        switch (mt) {
            #ifdef HAVE_CUDA
            case UCS_MEMORY_TYPE_CUDA:
                cuda_fill<<<1, 1>>>(&buffer[i], seed2);
                // cudaMemcpy(&temp, &buffer[i], sizeof(T), cudaMemcpyDefault);
                // LOG << ""
                // temp = rand<T>(seed);
                // assert(buffer != NULL);
                // LEO_add2((uint64_t*)buffer);
                // cudaMemcpy(&buffer[i], &temp, sizeof(T), cudaMemcpyDefault);
                break;
            case UCS_MEMORY_TYPE_CUDA_MANAGED:
#endif
            case UCS_MEMORY_TYPE_HOST:
                buffer[i] = rand<T>(seed);
                break;
            default:
                /* Unreachable - would fail in ctor */
                abort();
            }
        }

        if (_memory_type == UCS_MEMORY_TYPE_CUDA && flag) {
            cudaMemcpy(&seed, seed2, sizeof(unsigned), cudaMemcpyDefault);

        }

        cudaFree(seed2);
        // printf("LEO seed after: %d\n", seed);
    }

    template <typename T>
    size_t IoDemoRandom::validate(unsigned &seed, const T *buffer, size_t count) {
#ifdef HAVE_CUDA
        T expected_value, actual_value;
#endif
        // TODO: TEMP!!
        // return count;

        for (size_t i = 0; i < count; ++i) {
            switch (_memory_type) {
#ifdef HAVE_CUDA
            case UCS_MEMORY_TYPE_CUDA:
                expected_value = rand<T>(seed);
                cudaMemcpy(&actual_value, &buffer[i], sizeof(T),
                           cudaMemcpyDefault);
                if (actual_value != expected_value) {
                    return i;
                }
                break;
            case UCS_MEMORY_TYPE_CUDA_MANAGED:
#endif
            case UCS_MEMORY_TYPE_HOST:
                if (buffer[i] != rand<T>(seed)) {
                    return i;
                }
                break;

            default:
                /* Unreachable - would fail in ctor */
                abort();
            }
        }

        return count;
    }

//     static       unsigned     _seed;
//     static const unsigned     _A;
//     static const unsigned     _C;
//     static const unsigned     _M;
//     static ucs_memory_type_t _memory_type;
// };
unsigned IoDemoRandom::_seed    = 0;
const unsigned IoDemoRandom::_A = 1103515245U;
const unsigned IoDemoRandom::_C = 12345U;
const unsigned IoDemoRandom::_M = 0x7fffffffU;
ucs_memory_type_t IoDemoRandom::_memory_type = UCS_MEMORY_TYPE_HOST;



// const static int cuda_util_blocks_num = 1;
// const static int cuda_util_threads_per_block = 1;

__global__
void LEO_add(uint64_t *x)
{
    *x += 1;
    // x += y;
}

void LEO_add2(uint64_t *x)
{
    uint64_t tmp;
    cudaMemcpy(&tmp, x, sizeof(uint64_t), cudaMemcpyDefault);
    printf("LEO1 %lu\n", tmp);
    LEO_add<<<1, 1>>>(x);
    cudaDeviceSynchronize();
    cudaMemcpy(&tmp, x, sizeof(uint64_t), cudaMemcpyDefault);
    printf("LEO1 %lu\n", tmp);
}

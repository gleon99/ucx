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
                                  size_t size, bool flag) {
        size_t body_count    = size / sizeof(uint64_t);
        size_t tail_count    = size & (sizeof(uint64_t) - 1);
        const uint64_t *body = reinterpret_cast<const uint64_t*>(buffer);
        const uint8_t *tail  = reinterpret_cast<const uint8_t*>(body + body_count);

        size_t err_pos = validate(seed, body, body_count, flag);
        if (err_pos < body_count) {
            return err_pos * sizeof(body[0]);
        }

        err_pos = validate(seed, tail, tail_count, flag);
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
#ifdef HAVE_CUDA
        T temp;
#endif

    ucs_memory_type_t mt = flag ? _memory_type : UCS_MEMORY_TYPE_HOST;
    printf("LEO mt=%d, seed before: %d (count=%d)\n", mt, seed, count);
   
   
    if (flag && (_memory_type == UCS_MEMORY_TYPE_CUDA)) {
        // unsigned *seed2;
        // cudaMalloc(&seed2, sizeof(unsigned));
        cudaMemcpy(_seed_p, &seed, sizeof(unsigned), cudaMemcpyDefault);
        cuda_fill<<<1, 1>>>(buffer, _seed_p, count);
        cudaMemcpy(&seed, _seed_p, sizeof(unsigned), cudaMemcpyDefault);
        // cudaFree(seed2);

        printf("LEO seed after: %d\n", seed);
        {
            T first_char, last_char;
            cudaMemcpy(&first_char, buffer, sizeof(T), cudaMemcpyDefault);
            cudaMemcpy(&last_char, &buffer[count > 0 ? count - 1: 0], sizeof(T), cudaMemcpyDefault);
            printf("First char: %d, last_char = %d\n", first_char, last_char);
        }
        
        return;
  }
    
    for (size_t i = 0; i < count; ++i) {
        switch (mt) {
            #ifdef HAVE_CUDA
            case UCS_MEMORY_TYPE_CUDA:
                // temp = rand<T>(seed);
                // cudaMemcpy(&buffer[i], &temp, sizeof(T), cudaMemcpyDefault);


            // cuda_fill<<<1, 1>>>(&buffer[i], seed2);
            // cudaMemcpy(&temp, &buffer[i], sizeof(T), cudaMemcpyDefault);
            // LOG << ""
            // temp = rand<T>(seed);
            // assert(buffer != NULL);
            // LEO_add2((uint64_t*)buffer);
            // cudaMemcpy(&buffer[i], &temp, sizeof(T), cudaMemcpyDefault);
            // abort();
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

        // if (_memory_type == UCS_MEMORY_TYPE_CUDA && flag) {
        //     cudaMemcpy(&seed, seed2, sizeof(unsigned), cudaMemcpyDefault);

        // }

        // cudaFree(seed2);
        printf("LEO seed after: %d\n", seed);
        {
            T first_char, last_char;
            cudaMemcpy(&last_char, &buffer[count > 0 ? count - 1: 0], sizeof(T), cudaMemcpyDefault);
            cudaMemcpy(&first_char, buffer, sizeof(T), cudaMemcpyDefault);
            printf("First char: %d, last_char: %d\n", first_char, last_char);
        }
    }

    template <typename T>
    size_t IoDemoRandom::validate(unsigned &seed, const T *buffer, size_t count, bool flag) {
#ifdef HAVE_CUDA
        T expected_value, actual_value;
#endif
        // TODO: TEMP!!
        // return count;

        ucs_memory_type_t mt = flag ? _memory_type : UCS_MEMORY_TYPE_HOST;
        printf("LEO validate mt=%d, seed before: %d (count=%d)\n", mt, seed, count);
        for (size_t i = 0; i < count; ++i) {
            switch (mt) {
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
unsigned *IoDemoRandom::_seed_p = NULL;
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

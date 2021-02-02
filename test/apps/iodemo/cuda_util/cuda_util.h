
#include <stddef.h>
#include <stdint.h>
#include <ucs/memory/memory_type.h>
#include <limits>
#include <cuda.h>
#include <cuda_runtime.h>


class IoDemoRandom {
public:
    IoDemoRandom() { printf("CTOR\n"); }
    ~IoDemoRandom() { printf("DTOR\n"); }
    static void srand(unsigned seed);
  
    template <typename T>
    static inline T rand(T min = std::numeric_limits<T>::min(),
                         T max = std::numeric_limits<T>::max() - 1) {
        return rand(_seed, min, max);
    }

    template <typename T>
    static T rand(unsigned &seed, T min = std::numeric_limits<T>::min(),
                         T max = std::numeric_limits<T>::max() - 1) {
        seed = (seed * _A + _C) & _M;
        /* To resolve that LCG returns alternating even/odd values */
        if (max - min == 1) {
            return (seed & 0x100) ? max : min;
        } else {
            return T(seed) % (max - min + 1) + min;
        }
    }

    static void fill(unsigned &seed, void *buffer, size_t size, bool flag = true);

    static size_t validate(unsigned &seed, const void *buffer,
                                  size_t size);

    static void setMemoryType(ucs_memory_type_t memory_type);
    
    static IoDemoRandom & get(); 

private:
    template <typename T>
    static inline void fill(unsigned &seed, T *buffer, size_t count, bool flag = true);

    template <typename T>
    static inline size_t validate(unsigned &seed, const T *buffer, size_t count);

    static       unsigned     _seed;
    static const unsigned     _A;
    static const unsigned     _C;
    static const unsigned     _M;
    static ucs_memory_type_t _memory_type;
};
/* 
template<>
uint64_t IoDemoRandom::rand(uint64_t min, uint64_t max);

template<>
unsigned long IoDemoRandom::rand(unsigned long min, unsigned long max);

template<>
uint32_t IoDemoRandom::rand(uint32_t min, uint32_t max); */

__global__ 
void LEO_add(uint64_t *x);

template <typename T>
__device__
    static void cuda_rand(unsigned &seed, T *dst, T min = std::numeric_limits<T>::min(),
                         T max = std::numeric_limits<T>::max() - 1) {
        
        /* TODO: Cleanup */
        seed = (seed * 1103515245U + 12345U) & 0x7fffffffU;
        
        /* To resolve that LCG returns alternating even/odd values */
        if (max - min == 1) {
            *dst = (seed & 0x100) ? max : min;
        } else {
            *dst = T(seed) % (max - min + 1) + min;
        }
        
    }

template<typename T>
__global__
void cuda_fill(T *dest, unsigned *seed)
{
    // unsigned seed2;

    // cudaMemcpy(&seed2, &seed, sizeof(seed), cudaMemcpyDefault);
     cuda_rand<T>(*seed, dest);
    //  cudaMemcpy(&seed2, &seed, sizeof(seed), cudaMemcpyDefault);
                // assert(buffer != NULL);
                // LEO_add2((uint64_t*)buffer);
                // cudaMemcpy(&temp, &temp, sizeof(T), cudaMemcpyDefault);
}

void LEO_add2(uint64_t *x);

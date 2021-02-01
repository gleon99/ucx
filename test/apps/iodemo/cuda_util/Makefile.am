NVCC = nvcc
NVCCFLAGS = -DHAVE_CUDA
# NVCCFLAGS = -DHAVE_CUDA
# if !IS_NON_POWER_ARCH
# NVCCFLAGS += $(cuda_CFLAGS_FP128)
# endif
NV_ARCH_FLAGS = 

# if IS_CUDA_COMPUTE_80
# NV_ARCH_FLAGS += -gencode=arch=compute_80,code=compute_80
# endif

LINK = $(LIBTOOL) --mode=link $(CC) -o $@

.cu.o:
	$(NVCC) -c $< -o $@  $(NVCCFLAGS) $(NV_ARCH_FLAGS)

.cu.lo:
	bash $(top_srcdir)/cuda_lt.sh "$(LIBTOOL)" $@ $(NVCC) -c  $< $(NVCCFLAGS) $(NV_ARCH_FLAGS)



comp_noinst = libcuda_util.la
libcuda_util_la_SOURCES  = cuda_util.cu
noinst_HEADERS  = cuda_util.h
# if HAVE_CUDA
# libcuda_util_la_CPPFLAGS = $(CUDA_CPPFLAGS) -DHAVE_CUDA
# libcuda_util_la_CFLAGS = 
# else
# libcuda_util_la_CPPFLAGS =
# endif

noinst_LTLIBRARIES = $(comp_noinst)


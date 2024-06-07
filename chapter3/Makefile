.PHONY : build clean

HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 sbsa ppc64le armv7l))
    ifneq ($(TARGET_ARCH),$(HOST_ARCH))
        ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 sbsa ppc64le))
            TARGET_SIZE := 64
        else ifneq (,$(filter $(TARGET_ARCH),armv7l))
            TARGET_SIZE := 32
        endif
    else
        TARGET_SIZE := $(shell getconf LONG_BIT)
    endif
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif

HOST_COMPILER ?= clang-17

# NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCC := $(CUDA_PATH)/bin/nvcc

NVCCFLAGS   := -m${TARGET_SIZE}

ALL_CCFLAGS := 
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += --threads 0
# variable <variable> was set but never used
ALL_CCFLAGS += -diag-suppress 550

GENCODE_FLAGS := 
GENCODE_FLAGS += -gencode arch=compute_50,code=sm_50
GENCODE_FLAGS += -gencode arch=compute_52,code=sm_52 
GENCODE_FLAGS += -gencode arch=compute_60,code=sm_60
GENCODE_FLAGS += -gencode arch=compute_61,code=sm_61 
GENCODE_FLAGS += -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS += -gencode arch=compute_75,code=sm_75
GENCODE_FLAGS += -gencode arch=compute_80,code=sm_80
GENCODE_FLAGS += -gencode arch=compute_86,code=sm_86 
GENCODE_FLAGS += -gencode arch=compute_89,code=sm_89
GENCODE_FLAGS += -gencode arch=compute_90,code=sm_90 
GENCODE_FLAGS += -gencode arch=compute_90,code=compute_90

all: build

build: grayScale blurGray blurRgb matrixMul

grayScale.o: grayScale.cu
	$(NVCC) $(ALL_CCFLAGS) -DGPU $(GENCODE_FLAGS) -o $@ -c $<

grayScale: grayScale.o
	$(NVCC) $(ALL_CCFLAGS) -DGPU $(GENCODE_FLAGS) -o $@ $+

blurGray.o: blur.cu
	$(NVCC) $(ALL_CCFLAGS) -DCHANNEL_NUM=1 $(GENCODE_FLAGS) -o $@ -c $<

blurGray: blurGray.o
	$(NVCC) $(ALL_CCFLAGS) -DCHANNEL_NUM=1 $(GENCODE_FLAGS) -o $@ $+

blurRgb.o: blur.cu
	$(NVCC) $(ALL_CCFLAGS) -DCHANNEL_NUM=3 $(GENCODE_FLAGS) -o $@ -c $<

blurRgb: blurRgb.o
	$(NVCC) $(ALL_CCFLAGS) -DCHANNEL_NUM=3 $(GENCODE_FLAGS) -o $@ $+

matrixMul.o: matrixMul.cu
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

matrixMul: matrixMul.o
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

run: build
	./grayScale
	./blurGray
	./blurRgb
	./matrixMul

clean:
	rm -f *.o  *.png grayScale blurGray blurRgb matrixMul

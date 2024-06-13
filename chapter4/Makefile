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

all: build

build: deviceQuery

deviceQuery.o: deviceQuery.cu
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

deviceQuery: deviceQuery.o
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

run: build
	./deviceQuery

clean:
	rm -f *.o deviceQuery

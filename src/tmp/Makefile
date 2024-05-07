# Compilers
HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
CC := g++

# Compiler flags
NVCCFLAGS := -Xcompiler -Wall -Xcompiler -Wextra -O2
CFLAGS := -Wall -Wextra -O2

# Include directories
INCLUDES := -I$(CUDA_PATH)/include

# CUDA library directory
CUDA_LIB_DIR := -L$(CUDA_PATH)/lib64
# CUDA linking libraries
CUDA_LINK_LIBS := -lcudart
# Architecture for NVCC
ARCH := -arch=sm_50

# Object file directory
OBJ_DIR := bin

# Target executable name
TARGET := BFS

# Source files
CU_SOURCES := $(wildcard *.cu)
C_SOURCES := $(wildcard *.c)

# Object files
CU_OBJECTS := $(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(CU_SOURCES))
C_OBJECTS := $(patsubst %.c,$(OBJ_DIR)/%.c.o,$(C_SOURCES))

# Build rule for the target executable
$(TARGET): $(CU_OBJECTS) $(C_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $^ -o $@

# Build rule for CUDA source files
$(OBJ_DIR)/%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@ $(ARCH)

# Build rule for C source files
$(OBJ_DIR)/%.c.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	rm -f $(TARGET) $(CU_OBJECTS) $(C_OBJECTS)

# Specify phony rules
.PHONY: clean

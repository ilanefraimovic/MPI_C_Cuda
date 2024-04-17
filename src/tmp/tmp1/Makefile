###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=gcc
CC_FLAGS=
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=-arch=sm_50

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = ../../../Documents/cuda-samples/Common

##########################################################

## Make variables ##

# Target executable name:
EXE = run_test

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/BFS_Cuda.o $(OBJ_DIR)/BFS_serial_c.o

##########################################################

## Compile ##

# Link c and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .c file to object files:
$(OBJ_DIR)/%.o : %.c
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -I$(INC_DIR)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)

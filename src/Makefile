# Compilers
NVCC = nvcc
CC = gcc

# Compiler flags
NVCCFLAGS = -arch=sm_35 -Xcompiler -Wall -Xcompiler -Wextra -O2
CFLAGS = -Wall -Wextra -O2

# Include directories
INCLUDES = -I/usr/local/cuda/include

# Libraries
LIBRARIES = -lcudart -lcublas

# Target executable name
TARGET = myapp

# Source files
CU_SOURCES = $(wildcard *.cu)
C_SOURCES = $(wildcard *.c)

# Object files
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)

# Build rule for the target executable
$(TARGET): $(CU_OBJECTS) $(C_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBRARIES)

# Build rule for CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Build rule for C source files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	rm -f $(TARGET) $(CU_OBJECTS) $(C_OBJECTS)

# Specify phony rules
.PHONY: clean


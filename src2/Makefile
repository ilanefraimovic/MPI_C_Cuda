# Compilers
HOST_COMPILER ?= mpic++
CC := mpic++

# Compiler flags
CFLAGS := -Wall -Wextra -O2 -DOMPI_SKIP_MPICXX

# Include directories
INCLUDES := -I../../Common

# Object file directory
OBJ_DIR := bin

# Target executable name
TARGET := BFS

# Source files
C_SOURCES := $(wildcard ../src/*.c) ../src2/main.c

# Object files
C_OBJECTS := $(OBJDIR)/BFS_serial_c.c.o.o $(OBJDIR)/partition.c.o.o $(OBJDIR)/graph_create.c.o $(OBJDIR)/main.c.o

# Build rule for the target executable
$(TARGET): $(C_OBJECTS)
	$(CC) $^ -o $@

# Build rule for C source files in src directory
$(OBJ_DIR)/BFS_serial_c.c.o: ../src/BFS_serial_c.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
$(OBJ_DIR)/partition.c.o: ../src/partition.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
$(OBJ_DIR)/graph_create.c.o: ../src/graph_create.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Build rule for main.c in src2 directory
$(OBJ_DIR)/main.c.o: main.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	rm -f $(TARGET) $(CU_OBJECTS) $(filter-out $(OBJ_DIR)/main.c.o, $(C_OBJECTS))

# Specify phony rules
.PHONY: clean

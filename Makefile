
# Compilers
NVCC := nvcc
CFLAGS := -O3 --std=c++20 
CUDAFLAGS := -arch=sm_86 -lcusparse

# Directories
BUILD_DIR := build
OBJECTS_DIR := $(BUILD_DIR)/objects
SRC_DIR := src
CUDA_DIR := /usr/local/cuda/include

# Files
SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
HEADER_FILES := $(wildcard $(SRC_DIR)/*.h)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu, $(OBJECTS_DIR)/%.o, $(SRC_FILES))
TARGET := $(BUILD_DIR)/run.out

# Rules
all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(CUDAFLAGS) -o $@ $^

$(OBJECTS_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADER_FILES)
	@mkdir -p $(OBJECTS_DIR)
	$(NVCC) $(CFLAGS) $(CUDAFLAGS) -I$(CUDA_DIR) -I$(SRC_DIR) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean



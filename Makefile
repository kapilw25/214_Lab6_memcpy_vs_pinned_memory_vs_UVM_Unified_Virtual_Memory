# Define the CUDA compiler to use
NVCC = nvcc

# Define any compile-time flags
CFLAGS = -O3

# Define the executable file names
EXECUTABLES = vector_add_Memcpy vector_add_Pinned vector_add_UVM

# Data sizes (32B to 2GB)
DATA_SIZES = 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912 1073741824 2147483648

.PHONY: all clean run

all: $(EXECUTABLES)

$(EXECUTABLES):
	$(NVCC) $(CFLAGS) $@.cu -o $@

run:
	@for exec in $(EXECUTABLES); do \
		for size in $(DATA_SIZES); do \
			echo Running $$exec with data size $$size; \
			./$$exec $$size; \
		done; \
	done

clean:
	rm -f $(EXECUTABLES)

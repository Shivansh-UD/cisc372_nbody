FLAGS = -DDEBUG
LIBS  = -lm
ALWAYS_REBUILD = makefile

# CUDA settings
NVCC = nvcc
CUDAFLAGS = -arch=sm_70  # Adjust if needed (sm_60, sm_75, etc.)

# -------------------------------
# CPU Version
# -------------------------------
nbody: nbody.o compute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<

compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<

# -------------------------------
# CUDA Version
# -------------------------------
nbody_cuda: nbody_cuda.o compute.cu.o
	$(NVCC) $(CUDAFLAGS) $^ -o $@ $(LIBS)

nbody_cuda.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

compute.cu.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

clean:
	rm -f *.o nbody nbody_cuda


main: kmeans.cpp
	g++ kmeans.cpp -o main -O3
omp: *.cpp
	icpc -o omp -std=c++14 -qopenmp -ipo -Ofast -simd -march=core-avx2 -mtune=core-avx2 -xCORE-AVX2 -m64 -qopt-zmm-usage=high ompkmeans.cpp
cuda:
	nvcc -O3 cudakmeans.cu -o cuda -Xcompiler -fopenmp -std=c++14 -arch=sm_35 -rdc=true -lcudadevrt
scuda:
	nvcc -O3 shared_memory_cudakmeans.cu -o scuda -Xcompiler -fopenmp -std=c++14 -arch=sm_35 -rdc=true -lcudadevrt
clean:
	rm -f main
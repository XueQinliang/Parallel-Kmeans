main: kmeans.cpp
	g++ kmeans.cpp -o main -O3
omp: *.cpp
	icpc -o omp -std=c++14 -qopenmp -ipo -Ofast -simd -march=core-avx2 -mtune=core-avx2 -xCORE-AVX2 -m64 -qopt-zmm-usage=high ompkmeans.cpp
clean:
	rm -f main
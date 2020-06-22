# Parallel-Kmeans
This is the program for Parallel final paper.

## Multidimensional_array

There are two *.cpp in folder, which use multidimensional array to storage data.

### How to run

cd multidimensional_array

#### original program

make

./main argv1 argv2 argv3 (represent amount of points, clusters, dimension)

#### openmp program

make omp

./omp argv1 argv2 argv3 (represent amount of points, clusters, dimension)

## One-dimensional_array

There are four *.cpp and *.cu in folder, which use one-dimensional array to storage data.

### How to run

cd one-dimensional_array

#### original program

make

./main argv1 argv2 argv3 (represent amount of points, clusters, dimension)

#### openmp program

make omp

./omp argv1 argv2 argv3 (represent amount of points, clusters, dimension)

#### cuda program

make cuda

./cuda argv1 argv2 argv3 (represent amount of points, clusters, dimension)

#### shared memory cuda program

make scuda

./scuda argv1 argv2 argv3 (represent amount of points, clusters, dimension)
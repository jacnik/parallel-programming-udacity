# parallel-programming-udacity
Udacity Intro to Parallel Programming course


# compile .cu file
```sh
nvcc cubingNumbers.cu -o cube.out
```

# run compiled file
```sh
./cube.out
```

# Build

```sh
mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS="-std=c++11" ..
make
```

## Number of parallel computations:
KERNEL<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(...)

N_BLOCK -> any number
N_THREADS_PER_BLOCK -> max 1024

1280 parallel computations:
KERNEL<<<10, 128>>>(...) == KERNEL<<<5, 256>>>(...)

# More general:
```
KERNEL<<<grid_of_blocks, block_of_threads>>>(...)
                |	    	    |
                V		        V
            (1, 2 or 3D)	(1, 2 or 3D)
```

const dim3 grid_of_blocks(1, 1, 1);  // dim3(w,1,1) == dim3(w) == w
const dim3 block_of_threads( 1, 1, 1);


128 x 128 image (16384 pixels) can be divided into
   KERNEL<<<128, 128>>>
or KERNEL<<<dim3(8,8,1), dim3(16,16,1)>>> // (64x256)


## Communication Patterns:
```sh
1. Map:
[1][2][3][4][5]  In
 |  |  |  |  |
 V  V  V  V  V
[6][7][8][9][10] Out

2. Gather:
[1][2][3][4][5]  In
 |_/|_/|_/|_/|
 |  |  |  |  |
 V  V  V  V  V
[6][7][8][9][10] Out

3. Scatter:
[1][2][3][4][5]  In
 |_ |_ |_ |_ |
 | \| \| \| \|
 V  V  V  V  V
[6][7][8][9][10] Out


4. Traspose:
[ 1][ 2][ 3][ 4][ 5]
[ 6][ 7][ 8][ 9][10]
[11][12][13][14][15]

== In Row Major Order == [ 1][ 2][ 3][ 4][ 5][ 6][ 7][ 8][ 9][10][11][12][13][14][15]

After transpose:
[ 1][ 6][11]
[ 2][ 7][12]
[ 3][ 8][13]
[ 4][ 9][14]
[ 5][10][15]

== In Column Major Order == [ 1][ 6][11][ 2][ 7][12][ 3][ 8][13][ 4][ 9][14][ 5][10][15]

Transpose on structures:

struct foo {
  float f;
  int i;
};

array of structures (AoS) => [f][i][f][i][f][i][f][i]
	   ||
Transpose  ||
	   VV
structure of arrays (SoA) => [f][f][f][f][i][i][i][i]


5. Stencil -> update each element of an array using neighbouring elements (using pattern called 'stencil').
```

## Shared memory declaration: __shared__ int array[128];
## Barrier: __syncthreads();


# Building OpenCV
```sh
# in openCV Directory
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=ON -D WITH_VTK=ON ..
sudo make -j8
sudo make install
```
path to linking openCV:
```sh
/usr/local/include/opencv4/
```
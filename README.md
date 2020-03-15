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

Read from one address, compute result and write to one adress.
one-to-one

2. Gather:
[1][2][3][4][5]  In
 |_/|_/|_/|_/|
 |  |  |  |  |
 V  V  V  V  V
[6][7][8][9][10] Out

Read from multiple addresses, compute result and write to one adress.
many-to-one

3. Scatter:
[1][2][3][4][5]  In
 |_ |_ |_ |_ |
 | \| \| \| \|
 V  V  V  V  V
[6][7][8][9][10] Out

Read from one address, compute result and write to multiple adresses.
Or
Read from one address, compute result and one adress to write to, and write to one adress.
one-to-many

4. Traspose:
one-to-one

[ 1][ 2][ 3][ 4][ 5]
[ 6][ 7][ 8][ 9][10]
[11][12][13][14][15]

:: In Row Major Order => [ 1][ 2][ 3][ 4][ 5][ 6][ 7][ 8][ 9][10][11][12][13][14][15]

After transpose:
[ 1][ 6][11]
[ 2][ 7][12]
[ 3][ 8][13]
[ 4][ 9][14]
[ 5][10][15]

:: In Column Major Order => [ 1][ 6][11][ 2][ 7][12][ 3][ 8][13][ 4][ 9][14][ 5][10][15]

Transpose on structures:

struct foo {
  float f;
  int i;
};

:: array of structures (AoS) => [f][i][f][i][f][i][f][i]
              ||
              || Transpose
              VV
:: structure of arrays (SoA) => [f][f][f][f][i][i][i][i]


5. Stencil -> update each element of an array using neighbouring elements (using pattern called 'stencil').
it is like a specialized gather (several-to-one)
        [a]
     [b][c][d]
        [e]
         |
         V
        [ ]
[ ][a+b+c+d+e][ ]
        [ ]

6. Reduce:
all-to-one
[]----->[]--->[]
[]-/ -->[]-/
[]--//
[]-//
[]-/

7. Scan/sort
all-to-all
[ ][ ][ ][ ][ ]
 |  |  \  |  /
 \ /    \/  /
 / \    /\ /
[ ][ ][ ][ ][ ]
```

## Shared memory declaration: __shared__ int array[128];
## Barrier: __syncthreads();


# Building OpenCV
```sh
# in openCV Directory
mkdir build
cd build
sudo cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.2.0/modules -D BUILD_opencv_surface_matching=OFF -D BUILD_opencv_superres=OFF -D BUILD_opencv_tracking=OFF -D BUILD_opencv_rgbd=OFF -D BUILD_opencv_line_descriptor=OFF ..
sudo make -j8
sudo make install

# to uninstall go to the same directory and run:
sudo make uninstall
```
path to linking openCV:
```sh
/usr/local/include/opencv4/
```
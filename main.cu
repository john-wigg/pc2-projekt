#include <cuda.h>
#include <iostream>

#include "io.cpp"

#define double double

void __global__ step(double *u, int size, int g, double c, double* max_dev,
                     double* snapshots, bool take_snapshot, int cur_snapshot_index) {
    
    extern __shared__ double shared_mem[];
    double *u_local = shared_mem;
    double *u_local_new = shared_mem + blockDim.x*blockDim.y;

    int i = blockIdx.x * (blockDim.x - 2 * g) + threadIdx.x - g;
    int j = blockIdx.y * (blockDim.y - 2 * g) + threadIdx.y - g;
    
    // Geisterzonen austauschen
    int local_index = threadIdx.x * blockDim.x + threadIdx.y;
    
    
    if (i >= 0 && j >= 0 && i < size && j < size) {
        u_local_new[local_index] = u[i * size + j];
    } else {
        // TODO: Das geht doch sicher schöner
        u_local_new[local_index] = 0.0;
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        max_dev[blockIdx.x * 32 + blockIdx.y] = 0.0;
    }

    __syncthreads();
    
    for (int k = 0; k < g; ++k) {
        u_local[local_index] = u_local_new[local_index];
        __syncthreads();
        if (threadIdx.x > k && threadIdx.y > k && threadIdx.x < blockDim.x - 1 - k && threadIdx.y < blockDim.y - 1 - k) {
            u_local_new[local_index] = u_local[local_index] + c * (u_local[local_index + 1] 
                                                                   + u_local[local_index - 1]
                                                                   + u_local[local_index - blockDim.x]
                                                                   + u_local[local_index + blockDim.x]
                                                                - 4.0 * u_local[local_index]);
                                                            }
        __syncthreads();
    }

    // Zurückschreiben.
    if (threadIdx.x >= g && threadIdx.y >= g && threadIdx.x < (blockDim.x - g) && threadIdx.y < (blockDim.y - g)) {
        u[i * size + j] = u_local_new[local_index];
        if (fabs(u_local[local_index] - u_local_new[local_index]) > max_dev[blockIdx.x * 32 + blockIdx.y]) {
            max_dev[blockIdx.x * 32 + blockIdx.y] = fabs(u_local[local_index] - u_local_new[local_index]);
        }
    }

    __syncthreads();

    
    /* Snapshotting. */
    
    if (take_snapshot) {
        int offset = size*size*cur_snapshot_index;
        snapshots[offset + i * size + j] = u_local_new[local_index];
    }
}

void __global__ master(double *u, int size, int g, double c, dim3 gridSize, dim3 blockSize, double *max_dev,
                       int number_snapshots, int* snapshot_steps, double* snapshots) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idx * gridSize.x + idy;
    int cur_snapshot_index = 0;
    __shared__ int abort[1];
    for(int k = 0; k < 10000; ++k) {
        if (id == 0) {
            /* Snapshotting. */
            bool take_snapshot = false;
            if (k * g  >= snapshot_steps[cur_snapshot_index]) {
                take_snapshot = true;
                ++cur_snapshot_index;
            }
            abort[0] = 1;
            step<<<gridSize, blockSize, blockSize.x*blockSize.y*sizeof(double)*2>>>(u, size, g, c, max_dev, snapshots, take_snapshot, cur_snapshot_index-1);
        }

        __syncthreads();
        if (id == 0) cudaDeviceSynchronize();
        __syncthreads();

        if (max_dev[threadIdx.x * blockDim.y + threadIdx.y] > 0.01) { // epsilon
            abort[0] = 0; // continue computation
        }
        
        __syncthreads();
        if (abort[0] == 1) break;
    }
}

int main(int argc, char** argv) {
    cudaFuncSetCacheConfig(step, cudaFuncCachePreferShared);

    int size = 512;
    int g = 5;
    int number_snapshots = 5;
    int snapshot_steps[5] = {0, 100, 200, 300, 400};
    dim3 gridSize = dim3(32, 32);
    dim3 blockSize = dim3(16 + 2*g, 16 + 2*g);

    double h = 0.1;
    double alpha = 1.0;
    double dt = 0.1;
    double c = alpha * dt / h*h;

    double *u_host, *u_dev, *max_dev, *max_host, *snapshots_host, *snapshots_dev;
    int *snapshot_steps_dev;

    u_host = (double *)malloc(size*size*sizeof(double));
    max_host = (double *)malloc(gridSize.x*gridSize.y*sizeof(double));
    snapshots_host = (double *)malloc(size*size*number_snapshots*sizeof(double));
    cudaMalloc((void **)&snapshot_steps_dev, number_snapshots*sizeof(int));
    cudaMalloc((void **)&snapshots_dev, size*size*number_snapshots*sizeof(double));
    cudaMalloc((void **)&u_dev, size*size*sizeof(double));
    cudaMalloc((void **)&max_dev, gridSize.x*gridSize.y*sizeof(double));

    // Initialize the grid.
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            u_host[i * size + j] = 0.0;
        }
    }
    for (int i = 200; i < 300; ++i) {
        for (int j = 200; j < 300; ++j) {
            u_host[i * size + j]= 25.0;
        }
    }
    
    cudaMemcpy(u_dev, u_host, size*size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(snapshot_steps_dev, snapshot_steps, number_snapshots*sizeof(int), cudaMemcpyHostToDevice);
    master<<<1, gridSize>>>(u_dev, size, g, c, gridSize, blockSize, max_dev, number_snapshots, snapshot_steps_dev, snapshots_dev);
    cudaMemcpy(u_host, u_dev, size*size*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_host, max_dev, gridSize.x*gridSize.y*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(snapshots_host, snapshots_dev, size*size*number_snapshots*sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < number_snapshots; ++i) {
        int offset = size*size*i;
        char fname[32];
        snprintf(fname, 32, "snap%d.ppm", i);
        printPPM(snapshots_host + offset, size, fname);
    }
    printPPM(u_host, size, "out.ppm");

    std::cout << "Happy noises." << std::endl;
}
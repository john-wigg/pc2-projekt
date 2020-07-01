#include <cuda.h>
#include <iostream>
#include <vector>
#include <cooperative_groups.h>
#include "io.cpp"

#define double double

void __global__ step(double *u, int n, int g, double c, double* max_dev,
                     double* snapshots, bool take_snapshot, int cur_snapshot_index) {
    
    extern __shared__ double shared_mem[];
    double *u_local = shared_mem;
    double *u_local_new = shared_mem + blockDim.x*blockDim.y;

    int i = blockIdx.x * (blockDim.x - 2 * g) + threadIdx.x - g;
    int j = blockIdx.y * (blockDim.y - 2 * g) + threadIdx.y - g;
    
    // Geisterzonen austauschen
    int local_index = threadIdx.x * blockDim.x + threadIdx.y;
    
    
    if (i >= 0 && j >= 0 && i < n && j < n) {
        u_local_new[local_index] = u[i * n + j];
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
        u[i * n + j] = u_local_new[local_index];
        if (fabs(u_local[local_index] - u_local_new[local_index]) > max_dev[blockIdx.x * 32 + blockIdx.y]) {
            max_dev[blockIdx.x * 32 + blockIdx.y] = fabs(u_local[local_index] - u_local_new[local_index]);
        }
    }

    __syncthreads();

    
    /* Snapshotting. */
    
    if (take_snapshot) {
        int offset = n*n*cur_snapshot_index;
        snapshots[offset + i * n + j] = u_local_new[local_index];
    }
}

/* This is the "master" kernel.
 * For each "major" iteration, its upper left thread starts a new "step" kernel. The kernel then waits for
 * for the "step" kernel to finish updating the grid and checks (in parallel), whether any thread blocks
 * of the "step" kernel have not converged yet. If at least one has not, the next "major" iteration is
 * started. If all thread blocks have converged, it aborts execution.
 * The convergence criteria used here is that the maximum absolute change of all grid values since the
 * last iteration is below a certain epsilon.
 */
void __global__ master(double *u, int n, int g, double c, dim3 gridSize, dim3 blockSize, double *max_dev,
                       int number_snapshots, int* snapshot_steps, double* snapshots, bool *global_abort) {
    cooperative_groups::grid_group grp = cooperative_groups::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idx * gridSize.x + idy;
    int cur_snapshot_index = 0;

    /* The convergence detection is done in two steps:
     * First, each thread block checks whether any thread has not converged;
     * if so, set the local shared "abort" flag to 0. Then, for each block
     * in the grid the "abort" flags are checked. If one of them is 0, set
     * the flag "global_abort" in global memory to 0.
     * This should be a performance improvement since most threads write to
     * shared memory.
     */
    __shared__ int abort[1];
    for(int k = 0; k < 10000; ++k) {
        if (id == 0) {
            /* Snapshotting. */
            bool take_snapshot = false;
            if (cur_snapshot_index < number_snapshots) {
                if (k * g  >= snapshot_steps[cur_snapshot_index]) {
                    take_snapshot = true;
                    ++cur_snapshot_index;
                }
            }

            step<<<gridSize, blockSize, blockSize.x*blockSize.y*sizeof(double)*2>>>(u, n, g, c, max_dev, snapshots, take_snapshot, cur_snapshot_index-1);

            abort[0] = 1;
        }

        __syncthreads();
        if (id == 0) cudaDeviceSynchronize();
        __syncthreads();

        grp.sync();

        if (idx < gridSize.x && idy < gridSize.y)
        {
            if (max_dev[idx * gridSize.y + idy] > 0.01) { // epsilon
                abort[0] = 0; // continue computation
            }
        }
        
        grp.sync();
        /* Save write operations by only having the top left thread of the
         * block write to global memory. */
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            if (abort[0] == 1) *global_abort = 1;
        }
        grp.sync();

        if (*global_abort == 1) break;
    }
}

int main(int argc, char** argv) {
    /* Cache configuration; we mainly want to use shared memory. */
    cudaFuncSetCacheConfig(step, cudaFuncCachePreferShared);

    /* Grid size in both directions. */
    int n = 512;

    /* Width of ghost zones. */
    int g = 5;

    /* Simulation parameters. */
    double h = 0.1;
    double alpha = 1.0;
    double dt = 0.1;
    double c = alpha * dt / h*h;

    /* Snapshotting. */
    std::vector<int> snapshot_steps = {0, 100, 200, 300, 400}; // Steps at which to perform snapshots.

    /* Thread block grid size. */
    dim3 gridSize = dim3(32, 32);
    
    dim3 blockSize = dim3(n / gridSize.x + 2*g, n / gridSize.y + 2*g);

    double *u_host; // Holds the grid values on the host.
    double *u_dev; // Holds the grid values on the device.
    double *max_dev; // Holds the maximum differences between grid points of previous and current time step for each thread block on device.
    double *snapshots_dev; // Holds the data of grid snaphots on the device.
    double *snapshots_host; // Holds the data of grid snapshots on the host.
    int *snapshot_steps_dev; // Holds the iteration steps at which to perform screenshots on the device. */

    /* Host allocations */
    u_host = (double *)malloc(n*n*sizeof(double));
    snapshots_host = (double *)malloc(n*n*snapshot_steps.size()*sizeof(double));

    /* Device allocations. */
    cudaMalloc((void **)&snapshot_steps_dev, snapshot_steps.size()*sizeof(int));
    cudaMalloc((void **)&snapshots_dev, n*n*snapshot_steps.size()*sizeof(double));
    cudaMalloc((void **)&u_dev, n*n*sizeof(double));
    cudaMalloc((void **)&max_dev, gridSize.x*gridSize.y*sizeof(double));

    // Initialize the grid.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            u_host[i * n + j] = 0.0;
        }
    }
    for (int i = 200; i < 300; ++i) {
        for (int j = 200; j < 300; ++j) {
            u_host[i * n + j]= 25.0;
        }
    }

    bool *global_abort;
    cudaMalloc((void **)&global_abort, sizeof(bool));
    
    /* Copy from host to device. */
    cudaMemcpy(u_dev, u_host, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(snapshot_steps_dev, snapshot_steps.data(), snapshot_steps.size()*sizeof(int), cudaMemcpyHostToDevice);

    /* Start the "master" kernel that is responible for the dynamic parallelism. */
    master<<<dim3(1, 1), dim3(32, 32)>>>(u_dev, n, g, c, gridSize, blockSize, max_dev, snapshot_steps.size(), snapshot_steps_dev, snapshots_dev, global_abort);

    /* Copy from device to host. */
    cudaMemcpy(u_host, u_dev, n*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(snapshots_host, snapshots_dev, n*n*snapshot_steps.size()*sizeof(double), cudaMemcpyDeviceToHost);
    
    /* Write snapshoits. */
    for (int i = 0; i < snapshot_steps.size(); ++i) {
        int offset = n*n*i;
        char fname[32];
        snprintf(fname, 32, "snap%d.ppm", i);
        printPPM(snapshots_host + offset, n, fname);
    }

    /* Write final results. */
    printPPM(u_host, n, "out.ppm");

    std::cout << "Processing finished." << std::endl;
}
#include <cuda.h>
#include <iostream>
#include <vector>
#include <cooperative_groups.h>
#include "io.cpp"

#define float float

void __global__ computeBlocks(float *u, int n, int g, float c, bool* not_converged,
                     float* snapshots, bool take_snapshot, int cur_snapshot_index) {
    
    extern __shared__ float shared_mem[];
    float *u_local = shared_mem;
    float *u_local_new = shared_mem + blockDim.x*blockDim.y;

    int i = blockIdx.x * (blockDim.x - 2 * g) + threadIdx.x - g;
    int j = blockIdx.y * (blockDim.y - 2 * g) + threadIdx.y - g;
    
    // Geisterzonen austauschen
    int local_index = threadIdx.x * blockDim.x + threadIdx.y;
    
    
    if (i >= 0 && j >= 0 && i < n && j < n) {
        u_local_new[local_index] = u[i * n + j];
    } else {
        u_local_new[local_index] = 0.0;
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        not_converged[blockIdx.x * 32 + blockIdx.y] = false;
    }
    
    for (int k = 0; k < g; ++k) {
        u_local[local_index] = u_local_new[local_index];
        __syncthreads();
        if (threadIdx.x > k && threadIdx.y > k && threadIdx.x < blockDim.x - 1 - k && threadIdx.y < blockDim.y - 1 - k) {
            if (i > 0 && j > 0 && i < n - 1 && j < n - 1) {
                u_local_new[local_index] = u_local[local_index] + c * (u_local[local_index + 1] 
                                                                    + u_local[local_index - 1]
                                                                    + u_local[local_index - blockDim.x]
                                                                    + u_local[local_index + blockDim.x]
                                                                    - 4.0 * u_local[local_index]);
                                                                }
                                                        }
        __syncthreads();
    }

    // Zurückschreiben.
    if (threadIdx.x >= g && threadIdx.y >= g && threadIdx.x < (blockDim.x - g) && threadIdx.y < (blockDim.y - g)) {
        u[i * n + j] = u_local_new[local_index];
        if (fabs(u_local[local_index] - u_local_new[local_index]) > 0.0001) {
            not_converged[blockIdx.x * 32 + blockIdx.y] = true;
        }
    }

    __syncthreads();

    
    /* Snapshotting. */
    /*
    if (take_snapshot) {
        int offset = n*n*cur_snapshot_index;
        snapshots[offset + i * n + j] = u_local_new[local_index];
    }*/
}

/* This is the "master" kernel.
 * For each "major" iteration, its upper left thread starts a new "step" kernel. The kernel then waits for
 * for the "step" kernel to finish updating the grid and checks (in parallel), whether any thread blocks
 * of the "step" kernel have not converged yet. If at least one has not, the next "major" iteration is
 * started. If all thread blocks have converged, it aborts execution.
 * The convergence criteria used here is that the maximum absolute change of all grid values since the
 * last iteration is below a certain epsilon.
 */
void __global__ master(float *u, int n, int g, float c, dim3 gridSize, dim3 blockSize, bool *not_converged,
                       int number_snapshots, int* snapshot_steps, float* snapshots, bool *global_abort, int max_it) {
    /* Use a cooperative group to sync all threads. */
    cooperative_groups::grid_group grp = cooperative_groups::this_grid();
    /* Get x and y indices in the grid. */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    /* Assign a singular id as well (makes checking for first thread easier). */
    int id = idx * gridSize.x + idy;

    /* Keep track of snapshot index. */
    int cur_snapshot_index = 0;

    /* The convergence detection is done in two steps:
     * First, each thread block checks whether any thread has not converged;
     * if so, set the local shared "abort" flag to 0. Then, for each block
     * in the grid the "abort" flags are checked. If one of them is 0, set
     * the flag "global_abort" in global memory to 0.
     * This should be a performance improvement since most threads write to
     * shared memory.
     */
    __shared__ int abort; // block-local abort flag

    for(int k = 0; k < max_it / g; ++k) {
        if (id == 0) {
            /* Snapshotting. */
            bool take_snapshot = false;
            if (cur_snapshot_index < number_snapshots) { // check whether there are still snapshots to do
                if (k * g  >= snapshot_steps[cur_snapshot_index]) { // check whether we should to a snapshot
                    take_snapshot = true;
                    ++cur_snapshot_index;
                }
            }

            /* Execute a "step" kernel */
            computeBlocks<<<gridSize, blockSize, blockSize.x*blockSize.y*sizeof(float)*2>>>(u, n, g, c, not_converged, snapshots, take_snapshot, cur_snapshot_index-1);

            /* Initialize global abort flag. */
            if (id == 0) *global_abort = 1;
        }

        /* Initialize (block-local) abort flag */
        if (threadIdx.x == 0 && threadIdx.y == 0) abort = 1;

        /* Wait for "step" execution. */
        __syncthreads();
        if (id == 0) cudaDeviceSynchronize();
        grp.sync();

        if (idx < gridSize.x && idy < gridSize.y)
        {
            if (not_converged[idx * gridSize.y + idy] == true) { // epsilon
                abort = 0; // continue computation
            }
        }

        /* Wait till all threads in the block have checked. */
        __syncthreads();
        
        /* Save write operations by only having the top left thread of the
         * block write to global memory. */
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            if (abort == 0) *global_abort = 0;
        }

        /* Wait till all blocks in the grid have checked. */
        grp.sync();
        //if (*global_abort == 1) break;
    }
}

int main(int argc, char** argv) {
    /* Cache configuration; we mainly want to use shared memory. */
    cudaFuncSetCacheConfig(computeBlocks, cudaFuncCachePreferShared);

    /* Grid size in both directions. */
    int n = 512;

    /* Width of ghost zones. */
    int g = 5;

    /* Maximum iterations (will be rounded down to multiple of g). */
    int max_it = 100000;

    /* Simulation parameters. */
    float h = 0.1;
    float alpha = 1.0;
    float dt = 0.1;
    float c = alpha * dt / h*h;

    /* Snapshotting. */
    std::vector<int> snapshot_steps = {0, 100, 200, 300, 400}; // Steps at which to perform snapshots.

    /* Thread block grid size. */
    dim3 gridSize = dim3(32, 32);
    
    dim3 blockSize = dim3(n / gridSize.x + 2*g, n / gridSize.y + 2*g);

    float *u_host; // Holds the grid values on the host.
    float *u_dev; // Holds the grid values on the device.
    bool *max_dev; // Holds the maximum differences between grid points of previous and current time step for each thread block on device.
    float *snapshots_dev; // Holds the data of grid snaphots on the device.
    float *snapshots_host; // Holds the data of grid snapshots on the host.
    int *snapshot_steps_dev; // Holds the iteration steps at which to perform screenshots on the device. */

    /* Host allocations */
    u_host = (float *)malloc(n*n*sizeof(float));
    snapshots_host = (float *)malloc(n*n*snapshot_steps.size()*sizeof(float));

    /* Device allocations. */
    cudaMalloc((void **)&snapshot_steps_dev, snapshot_steps.size()*sizeof(int));
    cudaMalloc((void **)&snapshots_dev, n*n*snapshot_steps.size()*sizeof(float));
    cudaMalloc((void **)&u_dev, n*n*sizeof(float));
    cudaMalloc((void **)&max_dev, gridSize.x*gridSize.y*sizeof(bool));

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

    printPPM(u_host, n, "init.ppm");

    bool *global_abort;
    cudaMalloc((void **)&global_abort, sizeof(bool));
    
    /* Copy from host to device. */
    cudaMemcpy(u_dev, u_host, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(snapshot_steps_dev, snapshot_steps.data(), snapshot_steps.size()*sizeof(int), cudaMemcpyHostToDevice);

    /* Start the "master" kernel that is responible for the dynamic parallelism. */
    master<<<dim3(1, 1), dim3(32, 32)>>>(u_dev, n, g, c, gridSize, blockSize, max_dev, snapshot_steps.size(), snapshot_steps_dev, snapshots_dev, global_abort, max_it);

    /* Copy from device to host. */
    cudaMemcpy(u_host, u_dev, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(snapshots_host, snapshots_dev, n*n*snapshot_steps.size()*sizeof(float), cudaMemcpyDeviceToHost);
    
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
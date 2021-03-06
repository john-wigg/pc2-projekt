#include <cuda.h>
#include <iostream>
#include <vector>
#include "io.cpp"
#include <fstream>

//////////////////////////////////////////////////////////
/*                    USER SETTINGS                     */
/* Can be edited by the user to modify the behaviour of */
/* the program.                                         */
/* Use nvcc main.cu -o main -rdc=true -arch=sm60 or the */
/* supplemented CMakeLists.txt to compile the program   */
/* after changing any of the settings.                  */
//////////////////////////////////////////////////////////

/* Prefer shared cache in cache configuration. */
bool prefer_shared = true;

/* Grid size in both directions. */
int n = 1024;

/* Width of ghost zones. */
int g = 3;

/* Maximum iterations (will be rounded down to multiple of g). */
int max_it = 1000000;

/* Epsilon for convergence. */
float conv_epsilon = 0.0000001;

/* Simulation parameters. */
float h = 0.1;
float alpha = 1.0;
float dt = 0.1;

/* Snapshotting. */
std::vector<int> snapshot_steps = {0, 25000, 50000, 75000}; // Steps at which to perform snapshots.

/* Thread block size. */
dim3 gridSize = dim3(64, 64);

//////////////////////////////////////////////////////////
/*                 START OF PROGRAM                     */
/*                Do not touch below!                   */
//////////////////////////////////////////////////////////

void __global__ computeBlocks(float *u, int n, int g, float c, bool* grid_converged,
                     float* snapshots, bool take_snapshot, int cur_snapshot_index, float conv_epsilon) {
    
    /* Store new and old grid in shared memory. */
    extern __shared__ float shared_mem[];
    float *u_local = shared_mem;
    float *u_local_new = shared_mem + blockDim.x*blockDim.y;

    /* Get the indices in the grid */
    int i = blockIdx.x * (blockDim.x - 2 * g) + threadIdx.x - g;
    int j = blockIdx.y * (blockDim.y - 2 * g) + threadIdx.y - g;
    
    /* Get the index/offset inside the current thread block */
    int local_index = threadIdx.x * blockDim.x + threadIdx.y;
    
    /* Write values from grid to shared memory. */
    if (i >= 0 && j >= 0 && i < n && j < n) {
        u_local_new[local_index] = u[i * n + j];
    }
    
    /* Iterate until ghost zones exceeded. */
    for (int k = 0; k < g; ++k) {
        /* Swap old with new grid. */
        u_local[local_index] = u_local_new[local_index];
        __syncthreads();
        /* Update grid. */
        if (threadIdx.x > k && threadIdx.y > k && threadIdx.x < blockDim.x - 1 - k && threadIdx.y < blockDim.y - 1 - k) {
            if (i > 0 && j > 0 && i < n - 1 && j < n - 1) { // Exclude boundary
                u_local_new[local_index] = u_local[local_index] + c * (u_local[local_index + 1] 
                                                                    + u_local[local_index - 1]
                                                                    + u_local[local_index - blockDim.x]
                                                                    + u_local[local_index + blockDim.x]
                                                                    - 4.0 * u_local[local_index]);
                                                                }
                                                        }
        __syncthreads();
    }

    /* Write back to global memory. */
    if (threadIdx.x >= g && threadIdx.y >= g && threadIdx.x < (blockDim.x - g) && threadIdx.y < (blockDim.y - g)) {
        if (i > 0 && j > 0 && n - 1 && j < n - 1) {
            u[i * n + j] = u_local_new[local_index];
            /* Set the block's convergence flag to false if one point has not converged. */
            if (fabs(u_local[local_index] - u_local_new[local_index]) > 0.0001) {
                *grid_converged = false;
            }
        }
    }

    /* Snapshotting. */
    if (take_snapshot) {
        if (i >= 0 && j >= 0 && i < n && j < n) {
            int offset = n*n*cur_snapshot_index;
            snapshots[offset + i * n + j] = u_local_new[local_index];
        }
    }
}

/* This is the "director" kernel.
 * For each "major" iteration, its upper left thread starts a new "step" kernel. The kernel then waits for
 * for the "step" kernel to finish updating the grid and checks (in parallel), whether any thread blocks
 * of the "step" kernel have not converged yet. If at least one has not, the next "major" iteration is
 * started. If all thread blocks have converged, it aborts execution.
 * The convergence criteria used here is that the maximum absolute change of all grid values since the
 * last iteration is below a certain epsilon.
 */
void __global__ director(float *u, int n, int g, float c, dim3 gridSize, dim3 blockSize,
                       int number_snapshots, int* snapshot_steps, float* snapshots, bool *grid_converged, int max_it, int *iterations, float conv_epsilon) {

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

    int k;
    for(k = 0; k < max_it / g; ++k) {
        /* Snapshotting. */
        bool take_snapshot = false;
        if (cur_snapshot_index < number_snapshots) { // check whether there are still snapshots to do
            if (k * g  >= snapshot_steps[cur_snapshot_index]) { // check whether we should to a snapshot
                take_snapshot = true;
                ++cur_snapshot_index;
            }
        }

        /* Initialize global abort flag. */
        *grid_converged = true;

        /* Execute a "step" kernel */
        computeBlocks<<<gridSize, blockSize, blockSize.x*blockSize.y*sizeof(float)*2>>>(u, n, g, c, grid_converged, snapshots, take_snapshot, cur_snapshot_index-1, conv_epsilon);
    
        /* Wait for kernel execution. */
        cudaDeviceSynchronize();

        if (*grid_converged == true) break;
    }
    *iterations = k * g;
}

int main(int argc, char** argv) {
    float c = alpha * dt / h*h;

    /* Apply cache config. */
    if (prefer_shared) {
        cudaFuncSetCacheConfig(computeBlocks, cudaFuncCachePreferShared);
    } else {
        cudaFuncSetCacheConfig(computeBlocks, cudaFuncCachePreferL1);
    }

    /* Calculate block size. */
    dim3 blockSize = dim3(n / gridSize.x + 2*g, n / gridSize.y + 2*g);

    /* Catch configuration errors. */
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    if (blockSize.x * blockSize.y > device_properties.maxThreadsPerBlock) {
        std::cerr << "ERROR: maxThreadsPerBlock is exceeded. Choose a larger grid or reduce the ghost zones." << std::endl;
        return -1;
    }

    if (gridSize.x > device_properties.maxGridSize[0] || gridSize.y > device_properties.maxGridSize[1]) {
        std::cerr << "ERROR: Grid dimensions exceed maxGridSize. Choose a smaller grid." << std::endl;
        return -1;
    }

    std::cout << "Running on device: " << device_properties.name << "." << std::endl;

    /* Device variables. */
    float *h_u; // Holds the grid values on the host.
    float *d_u; // Holds the grid values on the device.
    bool *d_grid_converged; // Stores whether the complete grid has converged.
    float *d_snapshots; // Holds the data of grid snaphots on the device.
    int *d_snapshot_steps; // Holds the iteration steps at which to perform screenshots on the device. */
    int *d_iterations; // Holds how many grid iterations were done inside the kernel on the device.

    /* Host variables. */
    float *h_snapshots; // Holds the data of grid snapshots on the host.
    int h_iterations; // Holds how many grid iterations were done inside the kernel on the host.

    std::cout << "Host allocations..." << std::endl;

    /* Host allocations */
    h_u = (float *)malloc(n*n*sizeof(float));
    h_snapshots = (float *)malloc(n*n*snapshot_steps.size()*sizeof(float));

    std::cout << "Initializing grid on host..." << std::endl;

    // Initialize the grid.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_u[i * n + j] = 0.0;
        }
    }
    for (int i = 0; i < n; ++i) {
        h_u[i] = -25.0;
        h_u[(n - 1) * n + i] = 25.0;
        h_u[i * n + 0] = -25.0;
        h_u[i * n + n - 1] = 25.0;
    }

    std::cout << "Device allocations..." << std::endl;

    /* Device allocations. */
    cudaMalloc((void **)&d_snapshot_steps, snapshot_steps.size()*sizeof(int));
    cudaMalloc((void **)&d_snapshots, n*n*snapshot_steps.size()*sizeof(float));
    cudaMalloc((void **)&d_u, n*n*sizeof(float));
    cudaMalloc((void **)&d_grid_converged, sizeof(bool));
    cudaMalloc((void **)&d_iterations, sizeof(int));

    std::cout << "Copying from host to device..." << std::endl;
    
    /* Copy from host to device. */
    cudaMemcpy(d_u, h_u, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_snapshot_steps, snapshot_steps.data(), snapshot_steps.size()*sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Timing and starting director kernel... See you on the other side." << std::endl;

    float time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    /* Start the "director" kernel that is responible for the dynamic parallelism. */
    director<<<1, 1>>>(d_u, n, g, c, gridSize, blockSize, snapshot_steps.size(), d_snapshot_steps, d_snapshots, d_grid_converged, max_it, d_iterations, conv_epsilon);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    std::cout << "Copying data back to host..." << std::endl;
    
    /* Copy from device to host. */
    cudaMemcpy(h_u, d_u, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_snapshots, d_snapshots, n*n*snapshot_steps.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_iterations, d_iterations, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Writing snapshots and results to output files on host..." << std::endl;
    
    /* Write snapshots. */
    for (int i = 0; i < snapshot_steps.size(); ++i) {
        int offset = n*n*i;
        char fname[32];
        snprintf(fname, 32, "snap%d.ppm", i);
        printPPM(h_snapshots + offset, n, fname);
    }

    /* Write final results. */
    printPPM(h_u, n, "out.ppm");

    std::cout << "Processing finished." << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "Total elapsed KERNEL time: " << time << " ms" << std::endl;
    std::cout << "Iterations until convergence: " << h_iterations << std::endl;
}
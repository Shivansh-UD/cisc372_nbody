#include <math.h>
#include <stdio.h>
#include "vector.h"
#include "config.h"
#include "compute.h"

#define TILE 256   // tile size for shared memory

//global vriables defined in nbody.c
extern vector3 *hPos;
extern vector3 *hVel;
extern double  *mass;

__device__ inline void pair_accel_accumulate(
    double &ax, double &ay, double &az,
    const double px, const double py, const double pz,
    const double qx, const double qy, const double qz,
    const double mj)
{
    double dx = px - qx;
    double dy = py - qy;
    double dz = pz - qz;
    double r2 = dx*dx + dy*dy + dz*dz + 1e-12;
    double invr = rsqrt(r2);
    double invr3 = invr * invr * invr;
    double s = -GRAV_CONSTANT * mj * invr3;
    ax += s * dx;
    ay += s * dy;
    az += s * dz;
}

// One thread computes acceleration for one body i, but we load j's in tiles to shared memory
__global__ void kernel_compute_tiled(vector3 *dPos, vector3 *dVel, double *dMass, int N, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // load position of body i
    double px = dPos[i][0];
    double py = dPos[i][1];
    double pz = dPos[i][2];

    double ax = 0.0, ay = 0.0, az = 0.0;

    // shared memory tile for positions and masses
    __shared__ double sPosx[TILE];
    __shared__ double sPosy[TILE];
    __shared__ double sPosz[TILE];
    __shared__ double sMass[TILE];

    // loop over tiles of other bodies
    for (int tileStart = 0; tileStart < N; tileStart += TILE)
    {
        int idx = tileStart + threadIdx.x;
        // load into shared memory
        if (idx < N) {
            sPosx[threadIdx.x] = dPos[idx][0];
            sPosy[threadIdx.x] = dPos[idx][1];
            sPosz[threadIdx.x] = dPos[idx][2];
            sMass[threadIdx.x] = dMass[idx];
        } else {
            sPosx[threadIdx.x] = 0.0;
            sPosy[threadIdx.x] = 0.0;
            sPosz[threadIdx.x] = 0.0;
            sMass[threadIdx.x] = 0.0;
        }
        __syncthreads();

        // iterate local tile
        int tileLimit = min(TILE, N - tileStart);
        for (int t = 0; t < tileLimit; ++t) {
            int j = tileStart + t;
            if (j == i) continue; // skip self-interaction
            pair_accel_accumulate(ax, ay, az,
                                  px, py, pz,
                                  sPosx[t], sPosy[t], sPosz[t],
                                  sMass[t]);
        }
        __syncthreads();
    }

    // update velocity
    dVel[i][0] += ax * dt;
    dVel[i][1] += ay * dt;
    dVel[i][2] += az * dt;

    // update position
    dPos[i][0] += dVel[i][0] * dt;
    dPos[i][1] += dVel[i][1] * dt;
    dPos[i][2] += dVel[i][2] * dt;
}

extern "C" void compute()
{
    int N = NUMENTITIES;
    double dt = (double)INTERVAL;

    // persistent static device pointers
    static vector3 *dPos = NULL;
    static vector3 *dVel = NULL;
    static double  *dMass = NULL;
    static bool initialized = false;

    if (!initialized) {
        // allocate device memory once
        cudaMalloc((void**)&dPos, sizeof(vector3) * N);
        cudaMalloc((void**)&dVel, sizeof(vector3) * N);
        cudaMalloc((void**)&dMass, sizeof(double)  * N);

        // copy initial host arrays to device
        cudaMemcpy(dPos, hPos, sizeof(vector3) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(dVel, hVel, sizeof(vector3) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(dMass, mass, sizeof(double)  * N, cudaMemcpyHostToDevice);

        initialized = true;
    } else {
        // Only update masses if they might change
        cudaMemcpy(dPos, hPos, sizeof(vector3) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(dVel, hVel, sizeof(vector3) * N, cudaMemcpyHostToDevice);
    }

    // Kernel launch config
    int block = 256; // block threads
    int grid = (N + block - 1) / block;

    kernel_compute_tiled<<<grid, block>>>(dPos, dVel, dMass, N, dt);
    cudaDeviceSynchronize();

    // copy results back to host
    cudaMemcpy(hPos, dPos, sizeof(vector3) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dVel, sizeof(vector3) * N, cudaMemcpyDeviceToHost);

}

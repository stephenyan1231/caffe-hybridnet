#define BLOCK_SIZE 64

#include <stdio.h>
#include "caffe/util/modified_permutohedral.hpp"
#include "caffe/util/hash_helper.cu"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_math.cuh"

namespace caffe {

template<typename Dtype>
static void swapHashTableValues(Dtype* oldValues, Dtype *newValues, Dtype* table_values, size_t size) {
  CUDA_CHECK(cudaMemcpy(oldValues, table_values, size, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(table_values, newValues, size, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(newValues, oldValues, size, cudaMemcpyDeviceToDevice));
  // Works but give poorer results
  //oldValues = table_values;
  //table_values = newValues;
  //newValues = oldValues;
}

template<int pd, typename Dtype>
__global__ static void createMatrix(const int w, const int h, const Dtype *positions, int *table_entries,
    int table_capacity, signed short* table_keys, const Dtype *scaleFactor, MatrixEntry<Dtype> *matrix) {
  // scanline order
  //const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  //const bool outOfBounds = (idx>=num_points) ;
  //const int threadId = idx;

  // 8x8 blocks
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int threadId = threadIdx.y * blockDim.x + threadIdx.x;
  const int idx = y * w + x;
  const bool outOfBounds = (x >= w) || (y >= h);

  Dtype myElevated[pd + 1];
  const Dtype *myPosition = positions + idx * pd;

  int myGreedy[pd + 1];
  int myRank[pd + 1];

  Dtype myBarycentric[pd + 2];
  __shared__ short keys[pd * BLOCK_SIZE];
  short *myKey = keys + threadId * pd;

  if (!outOfBounds) {

    myElevated[pd] = -pd * (myPosition[pd - 1]) * scaleFactor[pd - 1];
    for (int i = pd - 1; i > 0; i--) {
      myElevated[i] = myElevated[i + 1] - i * myPosition[i - 1] * scaleFactor[i - 1]
          + (i + 2) * (myPosition[i]) * scaleFactor[i];
    }
    myElevated[0] = myElevated[1] + 2 * (myPosition[0]) * scaleFactor[0];

    // find the closest zero-colored lattice point

    // greedily search for the closest zero-colored lattice point
    signed short sum = 0;
    Dtype down_factor = (Dtype)1.0 / (Dtype)(pd + 1);
    for (int i = 0; i <= pd; i++) {
      Dtype v = myElevated[i] * down_factor;
      Dtype up = ceilf(v) * (pd + 1);
      Dtype down = floorf(v) * (pd + 1);
      if (up - myElevated[i] < myElevated[i] - down) {
        myGreedy[i] = (signed short) up;
      } else {
        myGreedy[i] = (signed short) down;
      }
      sum += myGreedy[i];
    }
    sum /= (pd + 1);

    // sort differential to find the permutation between this simplex and the canonical one
    for (int i = 0; i <= pd; i++) {
      myRank[i] = 0;
      for (int j = 0; j <= pd; j++) {
        if (myElevated[i] - myGreedy[i] < myElevated[j] - myGreedy[j]
            || (myElevated[i] - myGreedy[i] == myElevated[j] - myGreedy[j] && i > j)) {
          myRank[i]++;
        }
      }
    }

    if (sum > 0) { // sum too large, need to bring down the ones with the smallest differential
      for (int i = 0; i <= pd; i++) {
        if (myRank[i] >= pd + 1 - sum) {
          myGreedy[i] -= pd + 1;
          myRank[i] += sum - (pd + 1);
        } else {
          myRank[i] += sum;
        }
      }
    } else if (sum < 0) { // sum too small, need to bring up the ones with largest differential
      for (int i = 0; i <= pd; i++) {
        if (myRank[i] < -sum) {
          myGreedy[i] += pd + 1;
          myRank[i] += (pd + 1) + sum;
        } else {
          myRank[i] += sum;
        }
      }
    }

    // turn delta into barycentric coords
    for (int i = 0; i <= pd + 1; i++) {
      myBarycentric[i] = 0;
    }

    for (int i = 0; i <= pd; i++) {
      Dtype delta = (myElevated[i] - myGreedy[i]) * down_factor;
      myBarycentric[pd - myRank[i]] += delta;
      myBarycentric[pd + 1 - myRank[i]] -= delta;
    }
    myBarycentric[0] += 1.0f + myBarycentric[pd + 1];
  }

  for (int color = 0; color <= pd; color++) {
    // Compute the location of the lattice point explicitly (all but
    // the last coordinate - it's redundant because they sum to zero)
    if (!outOfBounds) {
      for (int i = 0; i < pd; i++) {
        myKey[i] = myGreedy[i] + color;
        if (myRank[i] > pd - color)
          myKey[i] -= (pd + 1);
      }
    }

    if (!outOfBounds) {
      MatrixEntry<Dtype> r;
      r.index = hashTableInsert<pd>(myKey, table_keys, table_entries, table_capacity, idx * (pd + 1) + color);
      r.weight = myBarycentric[color];
      matrix[idx * (pd + 1) + color] = r;
    }
  }
}

template<int kd>
__global__ static void cleanHashTable(const int n, int *table_entries, int table_capacity, signed short* table_keys) {
  const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

  if (idx >= n)
    return;

  // find my hash table entry
  int *e = table_entries + idx;

  // Check if I created my own key in the previous phase
  if (*e >= 0) {
    // Rehash my key and reset the pointer in order to merge with
    // any other pixel that created a different entry under the
    // same key. If the computation was serial this would never
    // happen, but sometimes race conditions can make the same key
    // be inserted twice. hashTableRetrieve always returns the
    // earlier, so it's no problem as long as we rehash now.
    *e = hashTableRetrieve<kd>(table_keys + *e * kd, table_entries, table_keys, table_capacity);
  }
}

template<int pd, typename Dtype>
__global__ static void resetIndex(const int w, const int h, MatrixEntry<Dtype> *matrix, int *table_entries) {
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + (blockIdx.y / (pd + 1)) * blockDim.y;
  const int color = blockIdx.y % (pd + 1);
  const int idx = y * w + x;
  const bool outOfBounds = (x >= w) || (y >= h);
  if (!outOfBounds) {
    MatrixEntry<Dtype> r = matrix[idx * (pd + 1) + color];
    matrix[idx * (pd + 1) + color].index = table_entries[r.index];
  }
}

template<int pd, typename Dtype>
__global__ static void splatCache(const int w, const int h, const int vd, const Dtype *values,
    const MatrixEntry<Dtype> *matrix, Dtype *table_values) {
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + (blockIdx.y / (pd + 1)) * blockDim.y;
  const int threadId = threadIdx.y * blockDim.x + threadIdx.x;
  const int color = blockIdx.y % (pd + 1);
  const int idx = y * w + x;
  const bool outOfBounds = (x >= w) || (y >= h);

  __shared__ int sharedOffsets[BLOCK_SIZE];
  extern __shared__ float sharedValues[];
  int myOffset = -1;
  float *myValue = sharedValues + threadId * (vd + 1);

  if (!outOfBounds) {

    const Dtype *value = values + idx;

    MatrixEntry<Dtype> r = matrix[idx * (pd + 1) + color];

    // convert the matrix entry from a pointer into the entries array to a pointer into the keys/values array
    //matrix[idx*(pd+1)+color].index = r.index = table_entries[r.index];
    // record the offset into the keys/values array in shared space
    myOffset = sharedOffsets[threadId] = r.index * (vd + 1);

    for (int j = 0; j < vd; j++) {
      myValue[j] = value[j * w * h] * r.weight;
    }
    myValue[vd] = r.weight;

  } else {
    sharedOffsets[threadId] = -1;
  }

  __syncthreads();

  // am I the first thread in this block to care about this key?

  if (outOfBounds)
    return;

  for (int i = 0; i < BLOCK_SIZE; i++) {
    if (i < threadId) {
      if (myOffset == sharedOffsets[i]) {
        // somebody else with higher priority cares about this key
        return;
      }
    } else if (i > threadId) {
      if (myOffset == sharedOffsets[i]) {
        // someone else with lower priority cares about this key, accumulate it into mine
        for (int j = 0; j <= vd; j++) {
          sharedValues[threadId * (vd + 1) + j] += sharedValues[i * (vd + 1) + j];
        }
      }
    }
  }

  // only the threads with something to write to main memory are still going
  Dtype *val = table_values + myOffset;
  for (int j = 0; j <= vd; j++) {
    atomicAdd_dev<Dtype>(val + j, myValue[j]);
  }
}

template<int pd, typename Dtype>
__global__ static void blur(int n, Dtype *newValues, const MatrixEntry<Dtype> *matrix, const int *table_entries,
    const signed short *table_keys, const int table_capacity, Dtype *table_values, int color, const int vd) {
  const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

  if (idx >= n)
    return;

  // Check if I'm valid
  if (matrix[idx].index != idx)
    return;

  // find my key and the keys of my neighbours
  short myKey[pd + 1];
  short np[pd + 1];
  short nm[pd + 1];

  for (int i = 0; i < pd; i++) {
    myKey[i] = table_keys[idx * pd + i];
    np[i] = myKey[i] + 1;
    nm[i] = myKey[i] - 1;
  }

  np[color] -= pd + 1;
  nm[color] += pd + 1;

  int offNp = hashTableRetrieve<pd>(np, table_entries, table_keys, table_capacity);
  int offNm = hashTableRetrieve<pd>(nm, table_entries, table_keys, table_capacity);

  Dtype *valMe = table_values + (vd + 1) * idx;
  Dtype *valNp = table_values + (vd + 1) * offNp;
  Dtype *valNm = table_values + (vd + 1) * offNm;
  Dtype *valOut = newValues + (vd + 1) * idx;

  if (offNp >= 0 && offNm >= 0) {
    for (int i = 0; i <= vd; i++) {
      valOut[i] = (valNp[i] + (valMe[i] * 2) + valNm[i]) / 2;
    }
  } else if (offNp >= 0) {
    for (int i = 0; i <= vd; i++) {
      valOut[i] = (valNp[i] + (valMe[i] * 2)) / 2;
    }
  } else if (offNm >= 0) {
    for (int i = 0; i <= vd; i++) {
      valOut[i] = (valNm[i] + (valMe[i] * 2)) / 2;
    }
  } else {
    for (int i = 0; i <= vd; i++) {
      valOut[i] = valMe[i];
    }
  }
}

template<int pd, typename Dtype>
__global__ static void slice(const int w, const int h, const int vd, Dtype *values, const MatrixEntry<Dtype> *matrix,
    Dtype *table_values, bool add) {
  //const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int threadId = threadIdx.y * blockDim.x + threadIdx.x;
  const int idx = y * w + x;
  const bool outOfBounds = (x >= w) || (y >= h);

  if (outOfBounds)
    return;

  extern __shared__ float localValue[];

  float *myValue = localValue + threadId * vd;
  Dtype myWeight = 0;

  for (int i = 0; i < vd; i++) {
    myValue[i] = 0;
  }

  for (int i = 0; i <= pd; i++) {
    MatrixEntry<Dtype> r = matrix[idx * (pd + 1) + i];
    const Dtype *val = table_values + r.index * (vd + 1);
    for (int j = 0; j < vd; j++) {
      myValue[j] += r.weight * val[j];
    }
    myWeight += r.weight * val[vd];
  }

  //myWeight = 1.0f/myWeight;
  Dtype alpha = 1.0f / (1 + powf(2, -pd));
  for (int j = 0; j < vd; j++) {
    if (!add) {
      values[j * w * h + idx] = 0;
    }
    values[j * w * h + idx] += myValue[j] * alpha;
  }
}

template<int pd, typename Dtype>
void gpu_init(const Dtype* features, HashTable* table, MatrixEntry<Dtype>* matrix, const int w, const int h) {
  int num_points = w * h;
  // Scan line order
  //unsigned int blocks = (num_points-1)/64 + 1;
  //unsigned int blockSize = 64;
  dim3 blocks((w - 1) / 8 + 1, (h - 1) / 8 + 1, 1);
  dim3 blockSize(8, 8, 1);

  Dtype blurVariance = 0.5;
  Dtype *scaleFactor;
  Dtype *scaleFactorHost = new Dtype[pd];

  // Create Scale factor vector and give it to GPU
  // num_dimensions is likely to be low so do that
  // on the CPU
  for (int i = 0; i < pd; i++) {
    scaleFactorHost[i] = (pd + 1) * sqrtf((1.0 / 6 + blurVariance) / ((i + 1) * (i + 2)));
  }
  CUDA_CHECK(cudaMalloc((void** )&scaleFactor, sizeof(Dtype) * pd));
  CUDA_CHECK(cudaMemcpy(scaleFactor, scaleFactorHost, sizeof(Dtype) * pd, cudaMemcpyHostToDevice));

  createMatrix<pd, Dtype> <<<blocks, blockSize>>>(w, h,
      features,
      table->table_entries,
      table->table_capacity,
      table->table_keys,
      scaleFactor,
      matrix);
  CUDA_POST_KERNEL_CHECK
  ;

  // fix duplicate hash table entries
  int cleanBlockSize = 32;
  dim3 cleanBlocks((num_points - 1) / cleanBlockSize + 1, 2 * (pd + 1), 1);
  cleanHashTable<pd> <<<cleanBlocks, cleanBlockSize>>>(2*num_points*(pd+1),
      table->table_entries, table->table_capacity, table->table_keys);
  CUDA_POST_KERNEL_CHECK
  ;

  blocks.y *= pd + 1;
  resetIndex<pd, Dtype> <<<blocks, blockSize>>>(w, h, matrix, table->table_entries);
  CUDA_POST_KERNEL_CHECK
  ;

  // Clean intermediate variables
  delete[] scaleFactorHost;
  CUDA_CHECK(cudaFree(scaleFactor));
}

template<int pd, typename Dtype>
void gpu_compute(Dtype* out, const Dtype* in, const HashTable &table, const MatrixEntry<Dtype>* matrix, int w, int h, int vd,
    bool reverse, bool add) {

  // Create table_values
  int num_points = w * h;
  Dtype *table_values;
  CUDA_CHECK(cudaMalloc((void**)&table_values, sizeof(Dtype)*(vd+1)*num_points*(pd+1)))
  ;
  caffe_gpu_set<Dtype>(num_points * (vd + 1) * (pd + 1), 0, table_values);

  dim3 blocks((w - 1) / 8 + 1, (h - 1) / 8 + 1, 1);
  dim3 blockSize(8, 8, 1);

  // splat splits by color, so extend the y coordinate to our blocks to represent that
  blocks.y *= pd + 1;
  splatCache<pd, Dtype> <<<blocks, blockSize, BLOCK_SIZE*(vd+1)*sizeof(Dtype)>>>(w, h, vd,
      in,
      matrix,
      table_values);
  CUDA_POST_KERNEL_CHECK;

  // blur
  int cleanBlockSize = 32;
  dim3 cleanBlocks((num_points - 1) / cleanBlockSize + 1, 2 * (pd + 1), 1);
  Dtype *newValues;
  Dtype *oldValues;
  size_t size = num_points * (pd + 1) * (vd + 1) * sizeof(Dtype);
  CUDA_CHECK(cudaMalloc((void** )&(newValues), size));
  CUDA_CHECK(cudaMalloc((void** )&(oldValues), size));
  caffe_gpu_set<Dtype>(num_points * (vd + 1) * (pd + 1), 0, newValues);
  for (int color = reverse ? pd : 0; color <= pd && color >= 0; reverse ? color-- : color++) {
    blur<pd, Dtype> <<<cleanBlocks, cleanBlockSize>>>(num_points*(pd+1), newValues,
        matrix,
        table.table_entries,
        table.table_keys,
        table.table_capacity,
        table_values,
        color,
        vd);
    CUDA_POST_KERNEL_CHECK;
    // swap pointers does not seem to work...
    swapHashTableValues<Dtype>(oldValues, newValues, table_values, size);
  }

  // slice
  blocks.y /= (pd + 1);
  slice<pd, Dtype> <<<blocks, blockSize, sizeof(Dtype)*BLOCK_SIZE*vd>>>(w, h, vd, out, matrix, table_values, add);
  CUDA_POST_KERNEL_CHECK;

  // Free memory
  CUDA_CHECK(cudaFree(table_values));
  CUDA_CHECK(cudaFree(newValues));
  CUDA_CHECK(cudaFree(oldValues));
}

template<typename Dtype>
void ModifiedPermutohedral<Dtype>::init_gpu(const Dtype* features, int d, int w, int h) {
  //Initialize Hash table
  if (!is_init) {
    table.createHashTable(w * h * (d + 1), d);
    CUDA_CHECK(cudaMalloc((void ** )&matrix, sizeof(MatrixEntry<Dtype> ) * (w * h * (d + 1))));
  } else {
    table.resetHashTable(w_ * h_ * (d_ + 1), d_);
  }
  w_ = w;
  h_ = h;
  d_ = d;
  N_ = w * h;
  switch (d) {
  case 2:
    gpu_init<2, Dtype>(features, &table, matrix, w_, h_);
    break;
  case 5:
    gpu_init<5, Dtype>(features, &table, matrix, w_, h_);
    break;
  default:
    LOG(FATAL)<< "num_dimensions should be 2 or 5";
  }
}

template<typename Dtype>
void ModifiedPermutohedral<Dtype>::compute_gpu(Dtype* out, const Dtype* in, int value_size, bool reverse,
    bool add) const {
  // Losing time by dynamically allocating memory but more general function
  if (!is_init) {
    LOG(FATAL)<<"Initialize lattice before doing any computing";
  }
  switch(d_) {
    case 2:
    gpu_compute<2, Dtype>(out, in, table, matrix, w_, h_, value_size, reverse, add);
    break;
    case 5:
    gpu_compute<5, Dtype>(out, in, table, matrix, w_, h_, value_size, reverse, add);
    break;
    default:
    LOG(FATAL) << "num_dimensions should be 2 or 5";
  }
}

//void ModifiedPermutohedral::compute_gpu(double* out, const double* in, int value_size, bool reverse, bool add) const {
//  // Losing time by dynamically allocating memory but more general function
//  if (!is_init) {
//    LOG(FATAL)<<"Initialize lattice before doing any computing";
//  }
//  switch(d_) {
//    case 2:
//    gpu_compute<2, double>(out, in, table, matrix, w_, h_, value_size, reverse, add);
//    break;
//    case 5:
//    gpu_compute<5, double>(out, in, table, matrix, w_, h_, value_size, reverse, add);
//    break;
//    default:
//    LOG(FATAL) << "num_dimensions should be 2 or 5";
//  }
//}

// Instantiate class
template void ModifiedPermutohedral<float>::init_gpu(const float* features, int d, int w, int h);
template void ModifiedPermutohedral<double>::init_gpu(const double* features, int d, int w, int h);
template void ModifiedPermutohedral<float>::compute_gpu(float* out, const float* in, int value_size,
    bool reverse,
    bool add) const;
template void ModifiedPermutohedral<double>::compute_gpu(double* out, const double* in, int value_size, bool reverse,
    bool add) const;
}
  //namespace caffe

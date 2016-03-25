#ifndef CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
#define CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_

#include <cstdlib>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/hash_table.hpp"

/************************************************/
/***          ModifiedPermutohedral Lattice   ***/
/************************************************/

namespace caffe {

template<typename Dtype>
struct MatrixEntry {
  int index;
  Dtype weight;
};

template<typename Dtype>
class ModifiedPermutohedral {
protected:
  struct Neighbors {
    int n1, n2;
    Neighbors(int n1 = 0, int n2 = 0) :
        n1(n1), n2(n2) {
    }
  };

  // Check if GPU hash table if initialize
  bool is_init;

  std::vector<int> offset_, rank_;
  std::vector<Dtype> barycentric_;
  std::vector<Neighbors> blur_neighbors_;

  // GPU specific
  MatrixEntry<Dtype> *matrix;
  HashTable table;

  int N_; // number of elements
  int M_; // number of lattice points
  int d_; // feature dim
  int w_, h_; // width and height

  void init_cpu(const Dtype* features, int num_dimensions, int num_points);
  void init_gpu(const Dtype* features, int num_dimensions, int w, int h);

  void compute_cpu(Dtype* out, const Dtype* in, int value_size, bool reverse = false, bool add = false) const;
//  void compute_cpu(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

  void compute_gpu(Dtype* out, const Dtype* in, int value_size, bool reverse = false, bool add = false) const;
//  void compute_gpu(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

  void sseCompute(Dtype* out, const Dtype* in, int value_size, bool reverse = false, bool add = false) const;
//  void sseCompute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

  void seqCompute(Dtype* out, const Dtype* in, int value_size, bool reverse = false, bool add = false) const;
//  void seqCompute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

public:
  ModifiedPermutohedral();
  ~ModifiedPermutohedral() {
#ifndef CPU_ONLY
    if (is_init)
      CUDA_CHECK(cudaFree(matrix));
#endif
  }

  void init(const Dtype* features, int num_dimensions, int w, int h) {
    switch (Caffe::mode()) {
    case Caffe::CPU:
      init_cpu(features, num_dimensions, w * h);
      break;
#ifndef CPU_ONLY
    case Caffe::GPU:
      init_gpu(features, num_dimensions, w, h);
      is_init = true;
      break;
#endif
    default:
      LOG(FATAL)<<"Unknown caffe mode.";
    }
  }
  void compute(Dtype* out, const Dtype* in, int value_size, bool reverse = false, bool add = false) const {
    switch (Caffe::mode()) {
      case Caffe::CPU:
      compute_cpu(out, in, value_size, reverse, add);
      break;
#ifndef CPU_ONLY
      case Caffe::GPU:
      compute_gpu(out, in, value_size, reverse, add);
      break;
#endif
      default:
      LOG(FATAL) << "Unknown caffe mode.";
    }
  }
//  void compute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const {
//    switch (Caffe::mode()) {
//      case Caffe::CPU:
//      compute_cpu(out, in, value_size, reverse, add);
//      break;
//#ifndef CPU_ONLY
//      case Caffe::GPU:
//      compute_gpu(out, in, value_size, reverse, add);
//      break;
//#endif
//      default:
//      LOG(FATAL) << "Unknown caffe mode.";
//    }
//  }

};

}  //namespace caffe
#endif //CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_

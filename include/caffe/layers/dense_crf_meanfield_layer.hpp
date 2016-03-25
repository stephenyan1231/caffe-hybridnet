#ifndef CAFFE_DENSE_CRF_MEANFIELD_LAYER_HPP_
#define CAFFE_DENSE_CRF_MEANFIELD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/modified_permutohedral.hpp"

namespace caffe {

/* This is an optimized implementation of CRF-RNN layer
 * CRF-RNN paper: Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 * */
template<typename Dtype>
class DenseCRFMeanfieldLayer: public Layer<Dtype> {

public:
  explicit DenseCRFMeanfieldLayer(const LayerParameter& param) :
      Layer<Dtype>(param) {
  }
  virtual ~DenseCRFMeanfieldLayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "DenseCRFMeanfield";
  }

  virtual inline int ExactNumBottomBlobs() const {
    return 3;
  }
  virtual inline int ExactNumTopBlobs() const {
    return 1;
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  virtual void compute_spatial_kernel(Dtype* output_kernel);
  virtual void compute_bilateral_kernel(const Blob<Dtype>* const rgb_blob, const int n, Dtype* const output_kernel);

  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int num_pixels_;

  bool init_cpu;
  bool init_gpu;

  Dtype theta_alpha_;
  Dtype theta_beta_;
  Dtype theta_gamma_;
  Dtype bilateral_filter_weight_;
  Dtype spatial_filter_weight_;
  int num_iterations_;

  vector<shared_ptr<Blob<Dtype> > > iteration_output_blobs_;

  vector<shared_ptr<SoftmaxLayer<Dtype> > > softmax_layers_;
  vector<vector<Blob<Dtype>*> > softmax_bottom_vec_vec_;
  vector<vector<Blob<Dtype>*> > softmax_top_vec_vec_;

  shared_ptr<ModifiedPermutohedral<Dtype> > spatial_lattice_;
  vector<shared_ptr<ModifiedPermutohedral<Dtype> > > bilateral_lattices_;

  Blob<Dtype> spatial_norm_;
  Blob<Dtype> bilateral_norms_;

  Blob<Dtype> spatial_kernel_;
  Blob<Dtype> bilateral_kernel_;
  Blob<Dtype> norm_feed_;


  vector<shared_ptr<Blob<Dtype> > > spatial_out_blobs_;
  vector<shared_ptr<Blob<Dtype> > > bilateral_out_blobs_;
  vector<shared_ptr<Blob<Dtype> > > probs_;
  vector<shared_ptr<Blob<Dtype> > > message_passings_;

  Blob<Dtype> pairwise_;

};

}  // namespace caffe

#endif  // CAFFE_DENSE_CRF_MEANFIELD_LAYER_HPP_

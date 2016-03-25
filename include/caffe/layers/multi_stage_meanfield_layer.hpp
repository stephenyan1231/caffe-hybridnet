#ifndef CAFFE_MULTI_STAGE_MEANFIELD_LAYER_HPP_
#define CAFFE_MULTI_STAGE_MEANFIELD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/modified_permutohedral.hpp"

namespace caffe {

/*!
 *  \brief     A helper class for {@link MultiStageMeanfieldLayer} class, which is the Caffe layer that implements the
 *             CRF-RNN described in the paper: Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             This class itself is not a proper Caffe layer although it behaves like one to some degree.
 *
 *  \authors   Sadeep Jayasumana, Bernardino Romera-Paredes, Shuai Zheng, Zhizhong Su.
 *  \version   1.0
 *  \date      2015
 *  \copyright Torr Vision Group, University of Oxford.
 *  \details   If you use this code, please consider citing the paper:
 *             Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du,
 *             Chang Huang, Philip H. S. Torr. Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             For more information about CRF-RNN, please visit the project website http://crfasrnn.torr.vision.
 */
template<typename Dtype>
class MeanfieldIteration {
public:
  /**
   * Must be invoked only once after the construction of the layer.
   */
  void SetUp(Blob<Dtype>* const unary_terms, Blob<Dtype>* const softmax_input, Blob<Dtype>* const output_blob,
      const shared_ptr<ModifiedPermutohedral<Dtype> > spatial_lattice, const Blob<Dtype>* const spatial_norm);

  void Reshape(Blob<Dtype>* const unary_terms, Blob<Dtype>* const softmax_input, Blob<Dtype>* const output_blob,
      const shared_ptr<ModifiedPermutohedral<Dtype> > spatial_lattice, const Blob<Dtype>* const spatial_norm);

  /**
   * Must be invoked before invoking {@link Forward_cpu()}
   */
  void PrePass(const vector<shared_ptr<Blob<Dtype> > >& parameters_to_copy_from,
      const vector<shared_ptr<ModifiedPermutohedral<Dtype> > >* const bilateral_lattices,
      const Blob<Dtype>* const bilateral_norms);

  /**
   * Forward pass - to be called during inference.
   */
  void Forward_cpu();
  void Forward_gpu();

  /**
   * Backward pass - to be called during training.
   */
  void Backward_cpu();
  void Backward_gpu();

  // A quick hack. This should be properly encapsulated.
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

protected:
  vector<shared_ptr<Blob<Dtype> > > blobs_;

  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int num_pixels_;

  Blob<Dtype> spatial_out_blob_;
  Blob<Dtype> bilateral_out_blob_;
  Blob<Dtype> pairwise_;
  Blob<Dtype> prob_;
  Blob<Dtype> message_passing_;

  vector<Blob<Dtype>*> softmax_top_vec_;
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> sum_top_vec_;
  vector<Blob<Dtype>*> sum_bottom_vec_;

  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  shared_ptr<EltwiseLayer<Dtype> > sum_layer_;

  shared_ptr<ModifiedPermutohedral<Dtype> > spatial_lattice_;
  const vector<shared_ptr<ModifiedPermutohedral<Dtype> > >* bilateral_lattices_;

  const Blob<Dtype>* spatial_norm_;
  const Blob<Dtype>* bilateral_norms_;

};

/*!
 *  \brief     The Caffe layer that implements the CRF-RNN described in the paper:
 *             Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *  \authors   Sadeep Jayasumana, Bernardino Romera-Paredes, Shuai Zheng, Zhizhong Su.
 *  \version   1.0
 *  \date      2015
 *  \copyright Torr Vision Group, University of Oxford.
 *  \details   If you use this code, please consider citing the paper:
 *             Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du,
 *             Chang Huang, Philip H. S. Torr. Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             For more information about CRF-RNN, please visit the project website http://crfasrnn.torr.vision.
 */
template<typename Dtype>
class MultiStageMeanfieldLayer: public Layer<Dtype> {

public:
  explicit MultiStageMeanfieldLayer(const LayerParameter& param) :
      Layer<Dtype>(param) {
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "MultiStageMeanfield";
  }

  virtual ~MultiStageMeanfieldLayer();

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

//  Dtype* norm_feed_;
  Blob<Dtype> spatial_norm_;
  Blob<Dtype> bilateral_norms_;

  vector<Blob<Dtype>*> split_layer_bottom_vec_;
  vector<Blob<Dtype>*> split_layer_top_vec_;
  vector<shared_ptr<Blob<Dtype> > > split_layer_out_blobs_;
  vector<shared_ptr<Blob<Dtype> > > iteration_output_blobs_;
  vector<shared_ptr<MeanfieldIteration<Dtype> > > meanfield_iterations_;

  shared_ptr<SplitLayer<Dtype> > split_layer_;

  shared_ptr<ModifiedPermutohedral<Dtype> > spatial_lattice_;
//  float* bilateral_kernel_buffer_;
  vector<shared_ptr<ModifiedPermutohedral<Dtype> > > bilateral_lattices_;

  Blob<Dtype> spatial_kernel_;
  Blob<Dtype> bilateral_kernel_;
  Blob<Dtype> norm_feed_;

};

}  // namespace caffe

#endif  // CAFFE_MULTI_STAGE_MEANFIELD_LAYER_HPP_

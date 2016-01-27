#ifndef CAFFE_INTERPOLATION_LAYERS_HPP_
#define CAFFE_INTERPOLATION_LAYERS_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Bilinear upsample the input blob of shape
 * (number, channels, height, width)
 * The target resolution can be determined by
 * 1) specifying an integer 'interpolation_factor_'
 * 2) providing a 2nd input blob, whose height/width are used as target resolution
 */
template <typename Dtype>
class BilinearInterpolationLayer : public Layer<Dtype> {
 public:
  explicit BilinearInterpolationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BilinearInterpolation"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // @brief currently, only support positive integer interpolation factor
  // (a.k.a. upsampling)
  int interpolation_factor_;
  float target_size_factor_;
};

} // namespace caffe

#endif // CAFFE_INTERPOLATION_LAYERS_HPP_

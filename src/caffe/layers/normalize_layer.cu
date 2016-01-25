#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int n = bottom[0]->num();
  const int d = bottom[0]->count() / n;
  Dtype a;
  for(int i=0;i<n;++i) {
    caffe_gpu_dot<Dtype>(d, bottom_data + d * i, bottom_data + d * i, &a);
    caffe_gpu_scale<Dtype>(d, uniform_scaling_ * pow(a, -0.5), bottom_data + d * i, top_data + d * i);
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int n = bottom[0]->num();
  const int d = bottom[0]->count() / n;
  Dtype a;
  for (int i = 0; i < n; ++i) {
    caffe_gpu_dot<Dtype>(d, top_diff + d * i, top_data + d * i, &a);
    a /= uniform_scaling_;
    caffe_gpu_scale<Dtype>(d, a / uniform_scaling_, top_data + d * i, bottom_diff + d * i);
    caffe_gpu_sub<Dtype>(d, top_diff + d * i, bottom_diff + d * i, bottom_diff + d * i);
    caffe_gpu_dot<Dtype>(d, bottom_data + d * i, bottom_data + d * i, &a);
    caffe_gpu_scale<Dtype>(d, uniform_scaling_ * pow(a, -0.5), bottom_diff + d * i, bottom_diff + d * i);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);

}  // namespace caffe

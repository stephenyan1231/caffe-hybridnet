#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NormalizeParameter param = this->layer_param_.normalize_param();
  uniform_scaling_ = param.uniform_scaling();
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int n = bottom[0]->num();
  const int d = bottom[0]->count() / n;
  for (int i = 0; i < n; ++i) {
    Dtype a = caffe_cpu_dot<Dtype>(d, bottom_data + d * i, bottom_data + d * i);
    caffe_cpu_scale<Dtype>(d, uniform_scaling_ * pow(a, -0.5), bottom_data + d * i, top_data + d * i);
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int n = bottom[0]->num();
  const int d = bottom[0]->count() / n;
  for (int i = 0; i < n; ++i) {
    Dtype a = caffe_cpu_dot<Dtype>(d, top_diff + d * i, top_data + d * i) / uniform_scaling_;
    caffe_cpu_scale<Dtype>(d, a / uniform_scaling_, top_data + d * i, bottom_diff + d * i);
    caffe_sub<Dtype>(d, top_diff + d * i, bottom_diff + d * i, bottom_diff + d * i);
    a = caffe_cpu_dot<Dtype>(d, bottom_data + d * i, bottom_data + d * i);
    caffe_cpu_scale<Dtype>(d, uniform_scaling_ * pow(a, -0.5), bottom_diff + d * i, bottom_diff + d * i);
  }
}

#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe

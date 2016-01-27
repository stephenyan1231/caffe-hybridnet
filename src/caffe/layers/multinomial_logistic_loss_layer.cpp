#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/multinomial_logistic_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), 1);
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  inner_num_ = bottom[0]->count(2);
}

template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();

  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  Dtype loss = 0;
  int count = 0;
  for (int i = 0; i < num; ++i) {
    for (int j=0;j<inner_num_;++j) {
      const int label = static_cast<int>(bottom_label[i * inner_num_+j]);
      if(has_ignore_label_ && label == ignore_label_) {
        continue;
      }
      DCHECK_GE(label, 0);
      DCHECK_LT(label, bottom[0]->channels());
      Dtype prob = std::max(
          bottom_data[i * dim + label * inner_num_ + j], Dtype(kLOG_THRESHOLD));
      loss -= log(prob);
      count++;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / count;
}

template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

    int count = 0;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label = static_cast<int>(bottom_label[i * inner_num_+j]);
        if (has_ignore_label_ && label == ignore_label_) {
          continue;
        } else {
          Dtype prob = std::max(
              bottom_data[i * dim + label * inner_num_ + j], Dtype(kLOG_THRESHOLD));
          bottom_diff[i * dim + label * inner_num_ + j] = 1.0 / prob;
          count++;
        }
      }
    }
    caffe_scal<Dtype>(bottom[0]->count(), -top[0]->cpu_diff()[0]/(Dtype)count, bottom_diff);
  }
}

INSTANTIATE_CLASS(MultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(MultinomialLogisticLoss);

}  // namespace caffe

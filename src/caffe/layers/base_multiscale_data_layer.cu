#include <vector>

#include "caffe/layers/base_multiscale_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingMultiscaleDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  MultiscaleBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  int num_scale = this->scales_.size();
  for (int s = 0; s < num_scale; ++s) {
    // Reshape to loaded data.
    top[s]->ReshapeLike(*(batch->data_[s].get()));
    // Copy the data
    caffe_copy(batch->data_[s]->count(), batch->data_[s]->gpu_data(),
        top[s]->mutable_gpu_data());
  }

  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[num_scale]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[num_scale]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingMultiscaleDataLayer);

}  // namespace caffe

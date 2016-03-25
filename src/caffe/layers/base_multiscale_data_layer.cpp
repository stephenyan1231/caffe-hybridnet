#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_multiscale_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
BaseMultiscaleDataLayer<Dtype>::BaseMultiscaleDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
  const int num_scales = this->layer_param_.data_param().multi_scale_size();
  if(num_scales == 0) {
    scales_.push_back(1.0);
  } else {
    CHECK_EQ(this->layer_param_.data_param().multi_scale(0), 1.0);
    for(int i = 0; i < num_scales; ++i) {
      scales_.push_back(this->layer_param_.data_param().multi_scale(i));
    }
  }
}

template <typename Dtype>
void BaseMultiscaleDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(top.size(), scales_.size());
  if (top.size() == scales_.size()) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template<typename Dtype>
BasePrefetchingMultiscaleDataLayer<Dtype>::BasePrefetchingMultiscaleDataLayer(
    const LayerParameter& param) :
    BaseMultiscaleDataLayer<Dtype>(param), prefetch_free_(), prefetch_full_() {
  int num_scale = this->scales_.size();
  LOG(ERROR)<<"num_scale "<<num_scale;
  this->transformed_data_.resize(num_scale);
  for (int s = 0; s < num_scale; ++s) {
    this->transformed_data_[s].reset(new Blob<Dtype>());
  }
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.resize(num_scale);
    for (int s = 0; s < num_scale; ++s) {
      prefetch_[i].data_[s].reset(new Blob<Dtype>());
    }
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
void BasePrefetchingMultiscaleDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseMultiscaleDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
//    prefetch_[i].data_.mutable_cpu_data();
    for(int j=0;j<this->scales_.size();++j) {
      prefetch_[i].data_[j]->mutable_cpu_data();
    }

    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
//      prefetch_[i].data_.mutable_gpu_data();
      for(int j=0;j<this->scales_.size();++j) {
        prefetch_[i].data_[j]->mutable_gpu_data();
      }
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingMultiscaleDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      MultiscaleBatch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        for(int s=0;s<this->scales_.size();++s) {
          batch->data_[s]->data().get()->async_gpu_push(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void BasePrefetchingMultiscaleDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  MultiscaleBatch<Dtype>* batch = prefetch_full_.pop(
      "Data layer prefetch queue empty");
  int num_scale = this->scales_.size();
  for (int s = 0; s < num_scale; ++s) {
    // Reshape to loaded data.
    top[s]->ReshapeLike(*(batch->data_[s].get()));
    // Copy the data
    caffe_copy(batch->data_[s]->count(), batch->data_[s]->cpu_data(),
        top[s]->mutable_cpu_data());
  }

  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[num_scale]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[num_scale]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingMultiscaleDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseMultiscaleDataLayer);
INSTANTIATE_CLASS(BasePrefetchingMultiscaleDataLayer);

}  // namespace caffe

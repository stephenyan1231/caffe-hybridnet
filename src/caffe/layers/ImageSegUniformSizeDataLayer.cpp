#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/ImageSegUniformSizeDataLayer.hpp"
#include "caffe/util/benchmark.hpp"
//#include "caffe/util/io.hpp"
//#include "caffe/util/math_functions.hpp"
//#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
ImageSegUniformSizeDataLayer<Dtype>::ImageSegUniformSizeDataLayer(
    const LayerParameter& param) :
    BasePrefetchingDataLayer<Dtype>(param),
    reader_(param){
}

template<typename Dtype>
ImageSegUniformSizeDataLayer<Dtype>::~ImageSegUniformSizeDataLayer() {
  this->StopInternalThread();
}

template<typename Dtype>
void ImageSegUniformSizeDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_image_shape = this->data_transformer_->InferBlobShape(datum);
  vector<int> top_label_shape(top_image_shape);
  top_label_shape[1] = 1;

  top_image_shape[0] = batch_size;
  top_label_shape[0] = batch_size;

  this->transformed_data_.Reshape(top_image_shape);
  if (this->output_labels_) {
    transformed_label_.Reshape(top_label_shape);
  }

  // Reshape top[0] and prefetch_data according to the batch_size.
  top[0]->Reshape(top_image_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_image_shape);
  }
  LOG(INFO)<< "output data size: " << top[0]->num() << ","
  << top[0]->channels() << "," << top[0]->height() << ","
  << top[0]->width();
  // label
  if (this->output_labels_) {
    LOG(INFO)<< "output label size: " << top_label_shape[0] << ","
    << top_label_shape[1] << "," << top_label_shape[2] << ","
    << top_label_shape[3];
    top[1]->Reshape(top_label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(top_label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void ImageSegUniformSizeDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_image_shape = this->data_transformer_->InferBlobShape(datum);
  vector<int> top_label_shape(top_image_shape);
  top_label_shape[1] = 1;

  this->transformed_data_.Reshape(top_image_shape);
  if (this->output_labels_) {
    transformed_label_.Reshape(top_label_shape);
  }
  // Reshape batch according to the batch_size.
  top_image_shape[0] = batch_size;
  top_label_shape[0] = batch_size;
  batch->data_.Reshape(top_image_shape);
  if (this->output_labels_) {
    batch->label_.Reshape(top_label_shape);
  }

  Dtype *top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int data_offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + data_offset);

    if (this->output_labels_) {
      int label_offset = batch->label_.offset(item_id);
      transformed_label_.set_cpu_data(top_label + label_offset);
      this->data_transformer_->TransformImageAndDenseLabel(datum, &(this->transformed_data_),
          &transformed_label_);
    } else {
      this->data_transformer_->TransformImageAndDenseLabel(datum, &(this->transformed_data_), NULL);
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageSegUniformSizeDataLayer);
REGISTER_LAYER_CLASS(ImageSegUniformSizeData);

template<typename Dtype>
ImageSegUniformSizeMultiscaleDataLayer<Dtype>::ImageSegUniformSizeMultiscaleDataLayer(
    const LayerParameter& param) :
    BasePrefetchingMultiscaleDataLayer<Dtype>(param),
    reader_(param){
}

template<typename Dtype>
ImageSegUniformSizeMultiscaleDataLayer<Dtype>::~ImageSegUniformSizeMultiscaleDataLayer() {
  this->StopInternalThread();
}

template<typename Dtype>
void ImageSegUniformSizeMultiscaleDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_image_shape = this->data_transformer_->InferBlobShape(datum);
  vector<int> top_label_shape(top_image_shape);
  top_label_shape[1] = 1;

  top_image_shape[0] = batch_size;
  top_label_shape[0] = batch_size;

  vector<int> scale_top_image_shape(4);
  scale_top_image_shape[0] = batch_size;
  scale_top_image_shape[1] = top_image_shape[1];

  int num_scale = this->scales_.size();
//  this->transformed_data_.Reshape(top_image_shape);
  if (this->output_labels_) {
    transformed_label_.Reshape(top_label_shape);
  }

  // Reshape top[0] and prefetch_data according to the batch_size.
  for (int s = 0; s < num_scale; ++s) {
    scale_top_image_shape[2] = ceil(top_image_shape[2] * this->scales_[s]);
    scale_top_image_shape[3] = ceil(top_image_shape[3] * this->scales_[s]);
    this->transformed_data_[s]->Reshape(scale_top_image_shape);
    top[s]->Reshape(scale_top_image_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_[s]->Reshape(scale_top_image_shape);
    }
    LOG(INFO)<< "output data size: " << top[s]->num() << ","
    << top[s]->channels() << "," << top[s]->height() << ","
    << top[s]->width();
  }
//  top[0]->Reshape(top_image_shape);
//  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
//    this->prefetch_[i].data_.Reshape(top_image_shape);
//  }

  // label
  if (this->output_labels_) {
    LOG(INFO) << "output label size: " << top_label_shape[0] << ","
        << top_label_shape[1] << "," << top_label_shape[2] << ","
        << top_label_shape[3];
    top[this->scales_.size()]->Reshape(top_label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(top_label_shape);
    }
  }
//  if (top.size() == 3) {
//    vector<int> top_image_size_shape(1);
//    top_image_size_shape[0] = 2;
//    top[2]->Reshape(top_image_size_shape);
//    Dtype *top_image_size_data = top[2]->mutable_cpu_data();
//    top_image_size_data[0] = top_image_shape[2];
//    top_image_size_data[1] = top_image_shape[3];
//  }
}

// This function is called on prefetch thread
template<typename Dtype>
void ImageSegUniformSizeMultiscaleDataLayer<Dtype>::load_batch(MultiscaleBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

  int num_scale = this->scales_.size();
  for (int s = 0; s < num_scale; ++s) {
    CHECK(batch->data_[s]->count());
    CHECK(this->transformed_data_[s]->count());
  }
//  CHECK(batch->data_.count());
//  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_image_shape = this->data_transformer_->InferBlobShape(datum);
  vector<int> top_label_shape(top_image_shape);
  top_label_shape[1] = 1;

  vector<int> scale_top_image_shape(4);
  scale_top_image_shape[0] = batch_size;
  scale_top_image_shape[1] = top_image_shape[1];

  for (int s = 0; s < num_scale; ++s) {
    scale_top_image_shape[2] = ceil(top_image_shape[2] * this->scales_[s]);
    scale_top_image_shape[3] = ceil(top_image_shape[3] * this->scales_[s]);
    this->transformed_data_[s]->Reshape(scale_top_image_shape);
    batch->data_[s]->Reshape(scale_top_image_shape);
  }

  if (this->output_labels_) {
    transformed_label_.Reshape(top_label_shape);
  }
  // Reshape batch according to the batch_size.
  top_image_shape[0] = batch_size;
  top_label_shape[0] = batch_size;
//  batch->data_.Reshape(top_image_shape);
  if (this->output_labels_) {
    batch->label_.Reshape(top_label_shape);
  }

  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)

    for (int s = 0; s < num_scale; ++s) {
      Dtype* top_data = batch->data_[s]->mutable_cpu_data();
      int data_offset = batch->data_[s]->offset(item_id);
      this->transformed_data_[s]->set_cpu_data(top_data + data_offset);
    }
//    this->transformed_data_.set_cpu_data(top_data + data_offset);
    if (this->output_labels_) {
      int label_offset = batch->label_.offset(item_id);
      transformed_label_.set_cpu_data(top_label + label_offset);
      this->data_transformer_->TransformImageAndDenseLabel(datum, (this->transformed_data_),
          &transformed_label_);
    } else {
      this->data_transformer_->TransformImageAndDenseLabel(datum, (this->transformed_data_), NULL);
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageSegUniformSizeMultiscaleDataLayer);
REGISTER_LAYER_CLASS(ImageSegUniformSizeMultiscaleData);

}  // namespace caffe

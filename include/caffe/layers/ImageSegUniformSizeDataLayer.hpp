#ifndef CAFFE_IMAGE_SEG_UNIFORM_SIZE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SEG_UNIFORM_SIZE_DATA_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/base_multiscale_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"



namespace caffe {

/**
 * @brief Image and segmentation pair data provider.
 * Image sizes are uniform within the mini-batch
 * OUTPUT:
 * 0 : (num, channels, height, width): image values
 * 1: (num, 1, height, width): labels
 */
template <typename Dtype>
class ImageSegUniformSizeDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageSegUniformSizeDataLayer(const LayerParameter& param);
  virtual ~ImageSegUniformSizeDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // ImageSegUniformSizeDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "ImageSegUniformSizeData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  Blob<Dtype> transformed_label_;
  DataReader reader_;
};

/**
 * @brief Image and segmentation pair data provider.
 * Image sizes are uniform within the mini-batch
 * OUTPUT:
 * 0 ~ (n-1): (num, channels, height, width): image values at multi-scales
 * the first scale must be 100% (i.e. the original resolution of image cropping)
 * n: (num, 1, height, width): labels
 */
template <typename Dtype>
class ImageSegUniformSizeMultiscaleDataLayer : public BasePrefetchingMultiscaleDataLayer<Dtype> {
 public:
  explicit ImageSegUniformSizeMultiscaleDataLayer(const LayerParameter& param);
  virtual ~ImageSegUniformSizeMultiscaleDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // ImageSegUniformSizeMultiscaleDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "ImageSegUniformSizeMultiscaleData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
 protected:
  virtual void load_batch(MultiscaleBatch<Dtype>* batch);

  Blob<Dtype> transformed_label_;
  DataReader reader_;
};



}  // namespace caffe

#endif  // CAFFE_IMAGE_SEG_UNIFORM_SIZE_DATA_LAYER_HPP_

#ifndef CAFFE_IMAGE_SEG_UNIFORM_SIZE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SEG_UNIFORM_SIZE_DATA_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"



namespace caffe {
/**
 * @brief Image and segmentation pair data provider.
 * Image sizes are uniform within the mini-batch
 * OUTPUT:
 * 0: (num, channels, height, width): image values
 * 1: (num, 1, height, width): labels
 * 2: (2): image size (height, width)
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
  virtual inline int MaxTopBlobs() const { return 3; }
 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  Blob<Dtype> transformed_label_;
  DataReader reader_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_SEG_UNIFORM_SIZE_DATA_LAYER_HPP_

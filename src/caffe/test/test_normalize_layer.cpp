#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class NormalizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  NormalizeLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 2, 1, 2)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Dtype *bottom_data = blob_bottom_->mutable_cpu_data();
    bottom_data[4 + 0] = bottom_data[0] = 1;
    bottom_data[4 + 1] = bottom_data[1] = 1;
    bottom_data[4 + 2] = bottom_data[2] = 1;
    bottom_data[4 + 3] = bottom_data[3] = 1;
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~NormalizeLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(NormalizeLayerTest, TestDtypesAndDevices);

TYPED_TEST(NormalizeLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  shared_ptr<NormalizeLayer<Dtype> > layer(
      new NormalizeLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 2);
}

TYPED_TEST(NormalizeLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    NormalizeParameter* normalize_param =
        layer_param.mutable_normalize_param();
    normalize_param->set_uniform_scaling(1.0);
    shared_ptr<NormalizeLayer<Dtype> > layer(
        new NormalizeLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(data[i], 0.5);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(NormalizeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    NormalizeParameter* normalize_param =
        layer_param.mutable_normalize_param();
    normalize_param->set_uniform_scaling(100.0);
    NormalizeLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe

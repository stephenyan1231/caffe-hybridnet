#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/dense_crf_meanfield_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

const int num_image = 4;
const int channels = 3;
const int image_size = 6;

template <typename TypeParam>
class DenseCRFMeanfieldLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  DenseCRFMeanfieldLayerTest()
      : blob_bottom_unary_(new Blob<Dtype>(num_image, channels, image_size, image_size)),
        blob_bottom_softmax_input_(new Blob<Dtype>(num_image, channels, image_size, image_size)),
        blob_bottom_data_(new Blob<Dtype>(num_image, 3, image_size, image_size)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter unary_filler_param;
    unary_filler_param.set_mean(0);
    unary_filler_param.set_std(1.0);
    GaussianFiller<Dtype> unary_filler(unary_filler_param);
    unary_filler.Fill(this->blob_bottom_unary_);
    caffe_copy<Dtype>(this->blob_bottom_unary_->count(), this->blob_bottom_unary_->cpu_data(),
        this->blob_bottom_softmax_input_->mutable_cpu_data());

    FillerParameter data_filler_param;
    data_filler_param.set_min(-128);
    data_filler_param.set_max(128);
    UniformFiller<Dtype> data_filler(data_filler_param);
    data_filler.Fill(this->blob_bottom_data_);

    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DenseCRFMeanfieldLayerTest() {
    delete blob_bottom_unary_;
    delete blob_bottom_softmax_input_;
    delete blob_bottom_data_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_unary_;
  Blob<Dtype>* const blob_bottom_softmax_input_;
  Blob<Dtype>* const blob_bottom_data_;

  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DenseCRFMeanfieldLayerTest, TestDtypesAndDevices);

TYPED_TEST(DenseCRFMeanfieldLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_unary_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_softmax_input_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_data_);
  LayerParameter layer_param;
  DenseCRFMeanfieldParameter* dense_crf_meanfield_param =
      layer_param.mutable_dense_crf_meanfield_param();
  dense_crf_meanfield_param->set_theta_alpha(3);
  dense_crf_meanfield_param->set_theta_beta(3);
  dense_crf_meanfield_param->set_theta_gamma(3);
  dense_crf_meanfield_param->set_num_iterations(3);
  dense_crf_meanfield_param->set_spatial_filter_weight(3);
  dense_crf_meanfield_param->set_bilateral_filter_weight(5);

  shared_ptr<DenseCRFMeanfieldLayer<Dtype> > layer(
      new DenseCRFMeanfieldLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num_image);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), image_size);
  EXPECT_EQ(this->blob_top_->width(), image_size);
}
//
//TYPED_TEST(InnerProductLayerTest, TestForward) {
//  typedef typename TypeParam::Dtype Dtype;
//  this->blob_bottom_vec_.push_back(this->blob_bottom_);
//  bool IS_VALID_CUDA = false;
//#ifndef CPU_ONLY
//  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
//#endif
//  if (Caffe::mode() == Caffe::CPU ||
//      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//    LayerParameter layer_param;
//    InnerProductParameter* inner_product_param =
//        layer_param.mutable_inner_product_param();
//    inner_product_param->set_num_output(10);
//    inner_product_param->mutable_weight_filler()->set_type("uniform");
//    inner_product_param->mutable_bias_filler()->set_type("uniform");
//    inner_product_param->mutable_bias_filler()->set_min(1);
//    inner_product_param->mutable_bias_filler()->set_max(2);
//    shared_ptr<InnerProductLayer<Dtype> > layer(
//        new InnerProductLayer<Dtype>(layer_param));
//    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//    const Dtype* data = this->blob_top_->cpu_data();
//    const int count = this->blob_top_->count();
//    for (int i = 0; i < count; ++i) {
//      EXPECT_GE(data[i], 1.);
//    }
//  } else {
//    LOG(ERROR) << "Skipping test due to old architecture.";
//  }
//}

//TYPED_TEST(InnerProductLayerTest, TestForwardNoBatch) {
//  typedef typename TypeParam::Dtype Dtype;
//  this->blob_bottom_vec_.push_back(this->blob_bottom_nobatch_);
//  bool IS_VALID_CUDA = false;
//#ifndef CPU_ONLY
//  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
//#endif
//  if (Caffe::mode() == Caffe::CPU ||
//      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//    LayerParameter layer_param;
//    InnerProductParameter* inner_product_param =
//        layer_param.mutable_inner_product_param();
//    inner_product_param->set_num_output(10);
//    inner_product_param->mutable_weight_filler()->set_type("uniform");
//    inner_product_param->mutable_bias_filler()->set_type("uniform");
//    inner_product_param->mutable_bias_filler()->set_min(1);
//    inner_product_param->mutable_bias_filler()->set_max(2);
//    shared_ptr<InnerProductLayer<Dtype> > layer(
//        new InnerProductLayer<Dtype>(layer_param));
//    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//    const Dtype* data = this->blob_top_->cpu_data();
//    const int count = this->blob_top_->count();
//    for (int i = 0; i < count; ++i) {
//      EXPECT_GE(data[i], 1.);
//    }
//  } else {
//    LOG(ERROR) << "Skipping test due to old architecture.";
//  }
//}

TYPED_TEST(DenseCRFMeanfieldLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_unary_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_softmax_input_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_data_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    DenseCRFMeanfieldParameter* dense_crf_meanfield_param =
        layer_param.mutable_dense_crf_meanfield_param();
    dense_crf_meanfield_param->set_theta_alpha(2);
    dense_crf_meanfield_param->set_theta_beta(3);
    dense_crf_meanfield_param->set_theta_gamma(3);
    dense_crf_meanfield_param->set_num_iterations(2);
    dense_crf_meanfield_param->set_spatial_filter_weight(3.0);
    dense_crf_meanfield_param->set_bilateral_filter_weight(5.0);


    DenseCRFMeanfieldLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe

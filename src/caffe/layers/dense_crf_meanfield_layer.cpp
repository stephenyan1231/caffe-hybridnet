#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/dense_crf_meanfield_layer.hpp"

namespace caffe {

template<typename Dtype>
void DenseCRFMeanfieldLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  init_cpu = false;
  init_gpu = false;

  const caffe::DenseCRFMeanfieldParameter meanfield_param =
      this->layer_param_.dense_crf_meanfield_param();
  num_iterations_ = meanfield_param.num_iterations();
  CHECK_GT(num_iterations_, 1)<< "Number of iterations must be greater than 1.";
  theta_alpha_ = meanfield_param.theta_alpha();
  theta_beta_ = meanfield_param.theta_beta();
  theta_gamma_ = meanfield_param.theta_gamma();
  bilateral_filter_weight_ = meanfield_param.bilateral_filter_weight();
  spatial_filter_weight_ = meanfield_param.spatial_filter_weight();

  channels_ = bottom[0]->channels();
  // Initialize the parameters that will updated by back propagation.
  if (this->blobs_.size() > 0) {
    LOG(INFO)<< "DenseCRFMeanfieldLayer layer skipping parameter initialization.";
  } else {
    LOG(INFO)<< "DenseCRFMeanfieldLayer layer default parameter initialization.";
    // blobs_[0] - spatial kernel weights, blobs_[1] compatability matrix
    this->blobs_.resize(3);

    // Allocate space for kernel weights.
//    this->blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_));
//    this->blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_));
//    caffe_set<Dtype>(channels_ * channels_, Dtype(0.), this->blobs_[0]->mutable_cpu_data());
//    caffe_set<Dtype>(channels_ * channels_, Dtype(0.), this->blobs_[1]->mutable_cpu_data());

    // Initialize the kernels weights.
//    for (int i = 0; i < channels_; ++i) {
//      this->blobs_[0]->mutable_cpu_data()[i * channels_ + i] = spatial_filter_weight_;
//    }
//
//    for (int i = 0; i < channels_; ++i) {
//      this->blobs_[1]->mutable_cpu_data()[i * channels_ + i] = bilateral_filter_weight_;
//    }

    this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
    this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, 1));

    // Initialize the kernels weights.
    CHECK_GE(spatial_filter_weight_, 0);
    this->blobs_[0]->mutable_cpu_data()[0] = spatial_filter_weight_;
    CHECK_GE(bilateral_filter_weight_, 0);
    this->blobs_[1]->mutable_cpu_data()[0] = bilateral_filter_weight_;

    // Initialize the compatibility matrix to have the Potts model.
    this->blobs_[2].reset(new Blob<Dtype>(1, 1, channels_, channels_));
    caffe_set<Dtype>(this->blobs_[2]->count(), (Dtype)0.0, this->blobs_[2]->mutable_cpu_data());
    for (int c = 0; c < channels_; ++c) {
      (this->blobs_[2]->mutable_cpu_data())[c * channels_ + c] = Dtype(-1.);
    }
  }

  // Make blobs to store outputs of each meanfield iteration. Output of the last iteration is stored in top[0].
  // So we need only (num_iterations_ - 1) blobs.
  iteration_output_blobs_.resize(num_iterations_ - 1);
  for (int i = 0; i < num_iterations_ - 1; ++i) {
    iteration_output_blobs_[i].reset(new Blob<Dtype>());
    iteration_output_blobs_[i]->ReshapeLike(*bottom[0]);
  }

  softmax_layers_.resize(num_iterations_);
  softmax_bottom_vec_vec_.resize(num_iterations_);
  softmax_top_vec_vec_.resize(num_iterations_);

  spatial_out_blobs_.resize(num_iterations_);
  bilateral_out_blobs_.resize(num_iterations_);
  probs_.resize(num_iterations_);
  message_passings_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; ++i) {
    spatial_out_blobs_[i].reset(new Blob<Dtype>());
    bilateral_out_blobs_[i].reset(new Blob<Dtype>());
    probs_[i].reset(new Blob<Dtype>());
    probs_[i]->ReshapeLike(*bottom[0]);

    message_passings_[i].reset(new Blob<Dtype>());
  }

  softmax_layers_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; ++i) {
    softmax_bottom_vec_vec_[i].clear();
    softmax_bottom_vec_vec_[i].push_back(
        i == 0 ? bottom[1] : iteration_output_blobs_[i - 1].get());

    softmax_top_vec_vec_[i].clear();
    softmax_top_vec_vec_[i].push_back(probs_[i].get());

    LayerParameter softmax_param;
    softmax_layers_[i].reset(new SoftmaxLayer<Dtype>(softmax_param));
    softmax_layers_[i]->SetUp(softmax_bottom_vec_vec_[i],
        softmax_top_vec_vec_[i]);
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void DenseCRFMeanfieldLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  CHECK_EQ(channels_, bottom[0]->channels());
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_pixels_ = height_ * width_;

  spatial_norm_.Reshape(1, 1, height_, width_);
  bilateral_norms_.Reshape(num_, 1, height_, width_);

  if (spatial_kernel_.count() == 0
      || !(spatial_kernel_.shape(0) == height_
          && spatial_kernel_.shape(1) == width_)) {
    spatial_kernel_.Reshape(1, height_, width_, 2);
    compute_spatial_kernel(spatial_kernel_.mutable_cpu_data());
    spatial_lattice_.reset(new ModifiedPermutohedral<Dtype>());

    norm_feed_.Reshape(1, 1, height_, width_);
    caffe_set<Dtype>(norm_feed_.count(), Dtype(1.0),
        norm_feed_.mutable_cpu_data());

    bilateral_kernel_.Reshape(1, height_, width_, 5);

    // Initialize the spatial lattice. This does not need to be computed for every image because we use a fixed size.
    switch (Caffe::mode()) {
    case Caffe::CPU:
      spatial_lattice_->init(spatial_kernel_.cpu_data(), 2, width_, height_);
      spatial_lattice_->compute(spatial_norm_.mutable_cpu_data(),
          norm_feed_.cpu_data(), 1);
      init_cpu = true;
      break;
#ifndef CPU_ONLY
    case Caffe::GPU:
      spatial_lattice_->init(spatial_kernel_.gpu_data(), 2, width_, height_);
      spatial_lattice_->compute(spatial_norm_.mutable_gpu_data(),
          norm_feed_.gpu_data(), 1);
      init_gpu = true;
      break;
#endif
    default:
      LOG(FATAL)<< "Unknown caffe mode.";
    }

    Dtype* norm_data = spatial_norm_.mutable_cpu_data();
    for (int i = 0; i < num_pixels_; ++i) {
      norm_data[i] = 1.0f / (norm_data[i] + 1e-20f);
    }

    for (int i = 0; i < num_iterations_ - 1; ++i) {
      iteration_output_blobs_[i]->Reshape(num_, channels_, height_, width_);
    }
    top[0]->Reshape(num_, channels_, height_, width_);
    pairwise_.Reshape(num_, channels_, height_, width_);

    for (int i = 0; i < num_iterations_; ++i) {
      spatial_out_blobs_[i]->Reshape(num_, channels_, height_, width_);
      bilateral_out_blobs_[i]->Reshape(num_, channels_, height_, width_);
      probs_[i]->Reshape(num_, channels_, height_, width_);
      message_passings_[i]->Reshape(num_, channels_, height_, width_);
    }
  }
}

/**
 * Performs filter-based mean field inference given the image and unaries.
 *
 * bottom[0] - Unary terms
 * bottom[1] - Softmax input/Output from the previous iteration (a copy of the unary terms if this is the first stage).
 * bottom[2] - RGB images
 *
 * top[0] - Output of the mean field inference (not normalized).
 */
template<typename Dtype>
void DenseCRFMeanfieldLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Initialize the bilateral lattices.
  bilateral_lattices_.resize(num_);
  for (int n = 0; n < num_; ++n) {
    compute_bilateral_kernel(bottom[2], n,
        bilateral_kernel_.mutable_cpu_data());
    bilateral_lattices_[n].reset(new ModifiedPermutohedral<Dtype>());
    bilateral_lattices_[n]->init(bilateral_kernel_.cpu_data(), 5, width_,
        height_);
    // Calculate bilateral filter normalization factors.
    Dtype* norm_output_data = bilateral_norms_.mutable_cpu_data()
        + bilateral_norms_.offset(n);
    bilateral_lattices_[n]->compute(norm_output_data, norm_feed_.cpu_data(), 1);
    for (int i = 0; i < num_pixels_; ++i) {
      norm_output_data[i] = 1.f / (norm_output_data[i] + 1e-20f);
    }
  }

  for (int i = 0; i < num_iterations_; ++i) {
    softmax_layers_[i]->Forward(softmax_bottom_vec_vec_[i],
        softmax_top_vec_vec_[i]);
    for (int n = 0; n < num_; ++n) {
      Dtype* spatial_out_data = spatial_out_blobs_[i]->mutable_cpu_data()
          + spatial_out_blobs_[i]->offset(n);
      const Dtype* prob_input_data = probs_[i]->cpu_data()
          + probs_[i]->offset(n);
      spatial_lattice_->compute(spatial_out_data, prob_input_data, channels_,
          false);

      // Pixel-wise normalization.
      for (int channel_id = 0; channel_id < channels_; ++channel_id) {
        caffe_mul<Dtype>(num_pixels_, spatial_norm_.cpu_data(),
            spatial_out_data + channel_id * num_pixels_,
            spatial_out_data + channel_id * num_pixels_);
      }

      Dtype* bilateral_out_data = bilateral_out_blobs_[i]->mutable_cpu_data()
          + bilateral_out_blobs_[i]->offset(n);

      bilateral_lattices_[n]->compute(bilateral_out_data, prob_input_data,
          channels_, false);
      // Pixel-wise normalization.
      for (int channel_id = 0; channel_id < channels_; ++channel_id) {
        caffe_mul<Dtype>(num_pixels_,
            bilateral_norms_.cpu_data() + bilateral_norms_.offset(n),
            bilateral_out_data + channel_id * num_pixels_,
            bilateral_out_data + channel_id * num_pixels_);
      }
    }

    caffe_set<Dtype>(count_, Dtype(0.),
        message_passings_[i]->mutable_cpu_data());

    // for spatial kernel, give a different weight to each label
    caffe_cpu_axpby<Dtype>(num_ * num_pixels_ * channels_, this->blobs_[0]->cpu_data()[0],
        spatial_out_blobs_[i]->cpu_data(), 0.,
        message_passings_[i]->mutable_cpu_data());
//    for (int n = 0; n < num_; ++n) {
//      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
//          channels_, (Dtype) 1., this->blobs_[0]->cpu_data(),
//          spatial_out_blobs_[i]->cpu_data() + spatial_out_blobs_[i]->offset(n),
//          (Dtype) 0.,
//          message_passings_[i]->mutable_cpu_data()
//              + message_passings_[i]->offset(n));
//    }

    // for bilateral kernel, give a different weight to each label
    caffe_cpu_axpby<Dtype>(num_ * num_pixels_ * channels_, this->blobs_[1]->cpu_data()[0],
         bilateral_out_blobs_[i]->cpu_data(), 1.,
         message_passings_[i]->mutable_cpu_data());
//    for (int n = 0; n < num_; ++n) {
//      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
//          channels_, (Dtype) 1., this->blobs_[1]->cpu_data(),
//          bilateral_out_blobs_[i]->cpu_data()
//              + bilateral_out_blobs_[i]->offset(n), (Dtype) 1.,
//          message_passings_[i]->mutable_cpu_data()
//              + message_passings_[i]->offset(n));
//    }

    //--------------------------- Compatibility multiplication ----------------
    //Result from message passing needs to be multiplied with compatibility values.
    for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
          channels_, (Dtype) 1., this->blobs_[2]->cpu_data(),
          message_passings_[i]->cpu_data() + message_passings_[i]->offset(n),
          (Dtype) 0., pairwise_.mutable_cpu_data() + pairwise_.offset(n));
    }

    // Adding unary and pairwise terms
    Dtype *top_data =
        i == num_iterations_ - 1 ?
            top[0]->mutable_cpu_data() :
            iteration_output_blobs_[i]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), 0, top_data);
    caffe_axpy<Dtype>(bottom[0]->count(), 1.0, bottom[0]->cpu_data(), top_data);
    caffe_axpy<Dtype>(bottom[0]->count(), -1.0, pairwise_.cpu_data(), top_data);
  }
}

/**
 * Backprop through filter-based mean field inference.
 */
template<typename Dtype>
void DenseCRFMeanfieldLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  caffe_set<Dtype>(bottom[0]->count(), Dtype(0.),
      bottom[0]->mutable_cpu_diff());
  for (int i = 0; i < this->blobs_.size(); ++i) {
    caffe_set<Dtype>(this->blobs_[i]->count(), Dtype(0.),
        this->blobs_[i]->mutable_cpu_diff());
  }

  for (int i = (num_iterations_ - 1); i >= 0; --i) {
    const Dtype *top_diff =
        i == num_iterations_ - 1 ?
            top[0]->cpu_diff() : iteration_output_blobs_[i]->cpu_diff();
    caffe_axpy<Dtype>(top[0]->count(), 1.0, top_diff,
        bottom[0]->mutable_cpu_diff());
    caffe_cpu_axpby<Dtype>(top[0]->count(), -1.0, top_diff, 0,
        pairwise_.mutable_cpu_diff());

    //---------------------------- Update compatibility diffs ------------------
    for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
          num_pixels_, (Dtype) 1., pairwise_.cpu_diff() + pairwise_.offset(n),
          message_passings_[i]->cpu_data() + message_passings_[i]->offset(n),
          (Dtype) 1., this->blobs_[2]->mutable_cpu_diff());
    }

    //-------------------------- Gradient after compatibility transform--- -----
    for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
          channels_, (Dtype) 1., this->blobs_[2]->cpu_data(),
          pairwise_.cpu_diff() + pairwise_.offset(n), (Dtype) 0.,
          message_passings_[i]->mutable_cpu_diff()
              + message_passings_[i]->offset(n));
    }

    // ------------------------- Gradient w.r.t. kernels weights ------------
    Dtype *spatial_filter_weight_diff = this->blobs_[0]->mutable_cpu_diff();

    spatial_filter_weight_diff[0] += caffe_cpu_dot<Dtype>(num_ * num_pixels_ * channels_,
        message_passings_[i]->cpu_diff(), spatial_out_blobs_[i]->cpu_data());

    for (int n = 0; n < num_; ++n) {
//      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
//          num_pixels_, (Dtype) 1.,
//          message_passings_[i]->cpu_diff() + message_passings_[i]->offset(n),
//          spatial_out_blobs_[i]->cpu_data() + spatial_out_blobs_[i]->offset(n),
//          (Dtype) 1., this->blobs_[0]->mutable_cpu_diff());
    }

    Dtype *bilateral_filter_weight_diff = this->blobs_[1]->mutable_cpu_diff();

    bilateral_filter_weight_diff[0] += caffe_cpu_dot<Dtype>(num_ * num_pixels_ * channels_,
        message_passings_[i]->cpu_diff(),
        bilateral_out_blobs_[i]->cpu_data());
//    for (int n = 0; n < num_; ++n) {
//      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
//          num_pixels_, (Dtype) 1.,
//          message_passings_[i]->cpu_diff() + message_passings_[i]->offset(n),
//          bilateral_out_blobs_[i]->cpu_data()
//              + bilateral_out_blobs_[i]->offset(n), (Dtype) 1.,
//          this->blobs_[1]->mutable_cpu_diff());
//    }

    // Gradient w.r.t spatial and bilateral output blob
    caffe_cpu_axpby<Dtype>(num_ * num_pixels_ * channels_, this->blobs_[0]->cpu_data()[0],
        message_passings_[i]->cpu_diff(),
        0., spatial_out_blobs_[i]->mutable_cpu_diff());
//    for (int n = 0; n < num_; ++n) {
//      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
//          channels_, (Dtype) 1., this->blobs_[0]->cpu_data(),
//          message_passings_[i]->cpu_diff() + message_passings_[i]->offset(n),
//          (Dtype) 0.,
//          spatial_out_blobs_[i]->mutable_cpu_diff()
//              + spatial_out_blobs_[i]->offset(n));
//    }

    caffe_cpu_axpby<Dtype>(num_* num_pixels_*channels_, this->blobs_[1]->cpu_data()[0],
        message_passings_[i]->cpu_diff(),
        0.,bilateral_out_blobs_[i]->mutable_cpu_diff());
    for (int n = 0; n < num_; ++n) {
//      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
//          channels_, (Dtype) 1., this->blobs_[1]->cpu_data(),
//          message_passings_[i]->cpu_diff() + message_passings_[i]->offset(n),
//          (Dtype) 0.,
//          bilateral_out_blobs_[i]->mutable_cpu_diff()
//              + bilateral_out_blobs_[i]->offset(n));
    }

    //---------------------------- BP thru normalization --------------------------
    for (int n = 0; n < num_; ++n) {
      Dtype *spatial_out_diff = spatial_out_blobs_[i]->mutable_cpu_diff()
          + spatial_out_blobs_[i]->offset(n);
      for (int channel_id = 0; channel_id < channels_; ++channel_id) {
        caffe_mul<Dtype>(num_pixels_, spatial_norm_.cpu_data(),
            spatial_out_diff + channel_id * num_pixels_,
            spatial_out_diff + channel_id * num_pixels_);
      }

      Dtype *bilateral_out_diff = bilateral_out_blobs_[i]->mutable_cpu_diff()
          + bilateral_out_blobs_[i]->offset(n);
      for (int channel_id = 0; channel_id < channels_; ++channel_id) {
        caffe_mul<Dtype>(num_pixels_,
            bilateral_norms_.cpu_data() + bilateral_norms_.offset(n),
            bilateral_out_diff + channel_id * num_pixels_,
            bilateral_out_diff + channel_id * num_pixels_);
      }
    }

    //--------------------------- Gradient for message passing ---------------
    for (int n = 0; n < num_; ++n) {
      spatial_lattice_->compute(
          probs_[i]->mutable_cpu_diff() + probs_[i]->offset(n),
          spatial_out_blobs_[i]->cpu_diff() + spatial_out_blobs_[i]->offset(n),
          channels_, true, false);

      bilateral_lattices_[n]->compute(
          probs_[i]->mutable_cpu_diff() + probs_[i]->offset(n),
          bilateral_out_blobs_[i]->cpu_diff()
              + bilateral_out_blobs_[i]->offset(n), channels_, true, true);
    }

    vector<bool> propagate_down(2, true);
    softmax_layers_[i]->Backward(softmax_top_vec_vec_[i], propagate_down,
        softmax_bottom_vec_vec_[i]);

  }
}

template<typename Dtype>
DenseCRFMeanfieldLayer<Dtype>::~DenseCRFMeanfieldLayer() {
  if (init_cpu) {
//    delete[] bilateral_kernel_buffer_;
//    delete[] norm_feed_;
  }
#ifndef CPU_ONLY
  if (init_gpu) {
//    CUDA_CHECK(cudaFree(bilateral_kernel_buffer_));
//    CUDA_CHECK(cudaFree(norm_feed_));
  }
#endif
}

template<typename Dtype>
void DenseCRFMeanfieldLayer<Dtype>::compute_bilateral_kernel(
    const Blob<Dtype>* const rgb_blob, const int n,
    Dtype* const output_kernel) {

  for (int p = 0; p < num_pixels_; ++p) {
    output_kernel[5 * p] = static_cast<Dtype>(p % width_) / theta_alpha_;
    output_kernel[5 * p + 1] = static_cast<Dtype>(p / width_) / theta_alpha_;

    const Dtype * const rgb_data_start = rgb_blob->cpu_data()
        + rgb_blob->offset(n);
    output_kernel[5 * p + 2] = static_cast<Dtype>(rgb_data_start[p]
        / theta_beta_);
    output_kernel[5 * p + 3] =
        static_cast<Dtype>((rgb_data_start + num_pixels_)[p] / theta_beta_);
    output_kernel[5 * p + 4] = static_cast<Dtype>((rgb_data_start
        + num_pixels_ * 2)[p] / theta_beta_);
  }
}

template<typename Dtype>
void DenseCRFMeanfieldLayer<Dtype>::compute_spatial_kernel(
    Dtype* output_kernel) {

  for (int p = 0; p < num_pixels_; ++p) {
    output_kernel[2 * p] = static_cast<Dtype>(p % width_) / theta_gamma_;
    output_kernel[2 * p + 1] = static_cast<Dtype>(p / width_) / theta_gamma_;
  }
}

INSTANTIATE_CLASS(DenseCRFMeanfieldLayer);
REGISTER_LAYER_CLASS(DenseCRFMeanfield);
}
// namespace caffe

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/dense_crf_meanfield_layer.hpp"

namespace caffe {

// Avoid divergence by uncoalescing access
template<typename Dtype>
__global__ void computeBilateralKernel(const int num_pixels_,
    const Dtype* const rgb_blob, const int width_, const int height_,
    const int channels_, Dtype theta_alpha_, Dtype theta_beta_, const int n,
    Dtype* const output_kernel) {
  int offset = ((n * channels_) * height_) * width_;
  CUDA_KERNEL_LOOP(p, num_pixels_)
  {
    output_kernel[5 * p] = (Dtype) (p % width_) / theta_alpha_;
    output_kernel[5 * p + 1] = (Dtype) (p / width_) / theta_alpha_;
    const Dtype * const rgb_data_start = rgb_blob + offset;
    output_kernel[5 * p + 2] = (Dtype) (rgb_data_start[p] / theta_beta_);
    output_kernel[5 * p + 3] = (Dtype) ((rgb_data_start + num_pixels_)[p]
        / theta_beta_);
    output_kernel[5 * p + 4] = (Dtype) ((rgb_data_start + num_pixels_ * 2)[p]
        / theta_beta_);
  }
}

template<typename Dtype>
__global__ void computeNorm(Dtype* norm_output_data, int num_pixels) {
  CUDA_KERNEL_LOOP(i, num_pixels)
  {
    norm_output_data[i] = 1.f / (norm_output_data[i] + 1e-20f);
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
void DenseCRFMeanfieldLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (init_cpu) {
    LOG(FATAL)<< ("You initialize your network on CPU, please initialize it on GPU.");
  }
  const Dtype* bottom_data = bottom[2]->gpu_data();

// Initialize the bilateral lattices.
  bilateral_lattices_.resize(num_);
  for (int n = 0; n < num_; ++n) {
    computeBilateralKernel<Dtype><<<CAFFE_GET_BLOCKS(num_pixels_), CAFFE_CUDA_NUM_THREADS>>>(
        num_pixels_, bottom_data, width_, height_, channels_,
        theta_alpha_, theta_beta_, n,
        bilateral_kernel_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

    bilateral_lattices_[n].reset(new ModifiedPermutohedral<Dtype>());
    bilateral_lattices_[n]->init(bilateral_kernel_.gpu_data(), 5, width_, height_);
    // Calculate bilateral filter normalization factors.
    Dtype* norm_output_data = bilateral_norms_.mutable_gpu_data() + bilateral_norms_.offset(n);
    bilateral_lattices_[n]->compute(norm_output_data, norm_feed_.gpu_data(), 1);
    computeNorm<Dtype><<<CAFFE_GET_BLOCKS(num_pixels_), CAFFE_CUDA_NUM_THREADS>>>(norm_output_data, num_pixels_);
    CUDA_POST_KERNEL_CHECK;
  }
  // mean field iterations start
  for (int i = 0; i < num_iterations_; ++i) {
    softmax_layers_[i]->Forward(softmax_bottom_vec_vec_[i],
        softmax_top_vec_vec_[i]);
    for (int n = 0; n < num_; ++n) {
      Dtype* spatial_out_data = spatial_out_blobs_[i]->mutable_gpu_data()
      + spatial_out_blobs_[i]->offset(n);
      const Dtype* prob_input_data = probs_[i]->gpu_data()
      + probs_[i]->offset(n);
      spatial_lattice_->compute(spatial_out_data, prob_input_data, channels_,
          false);
      // Pixel-wise normalization.
      for (int channel_id = 0; channel_id < channels_; ++channel_id) {
        caffe_gpu_mul<Dtype>(num_pixels_, spatial_norm_.gpu_data(),
            spatial_out_data + channel_id * num_pixels_,
            spatial_out_data + channel_id * num_pixels_);
      }

      Dtype* bilateral_out_data = bilateral_out_blobs_[i]->mutable_gpu_data()
      + bilateral_out_blobs_[i]->offset(n);
      bilateral_lattices_[n]->compute(bilateral_out_data, prob_input_data,
          channels_, false);
      // Pixel-wise normalization.
      for (int channel_id = 0; channel_id < channels_; ++channel_id) {
        caffe_gpu_mul<Dtype>(num_pixels_,
            bilateral_norms_.gpu_data() + bilateral_norms_.offset(n),
            bilateral_out_data + channel_id * num_pixels_,
            bilateral_out_data + channel_id * num_pixels_);
      }
    }

    caffe_gpu_set<Dtype>(count_, Dtype(0.),
        message_passings_[i]->mutable_gpu_data());

    // for spatial kernel, give a different weight to each label
    caffe_gpu_axpby<Dtype>(num_ * num_pixels_ * channels_, this->blobs_[0]->cpu_data()[0],
        spatial_out_blobs_[i]->gpu_data(), 0., message_passings_[i]->mutable_gpu_data());
//    for (int n = 0; n < num_; ++n) {
//      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
//          channels_, (Dtype) 1., this->blobs_[0]->gpu_data(),
//          spatial_out_blobs_[i]->gpu_data() + spatial_out_blobs_[i]->offset(n),
//          (Dtype) 0.,
//          message_passings_[i]->mutable_gpu_data()
//          + message_passings_[i]->offset(n));
//    }

    // for bilateral kernel, give a different weight to each label
//    Dtype bilteral_filter_weight = 1.0 - this->blobs_[0]->cpu_data()[0];
    caffe_gpu_axpby<Dtype>(num_ * num_pixels_ * channels_, this->blobs_[1]->cpu_data()[0],
        bilateral_out_blobs_[i]->gpu_data(), 1., message_passings_[i]->mutable_gpu_data());

//    for (int n = 0; n < num_; ++n) {
//      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
//          channels_, (Dtype) 1., this->blobs_[1]->gpu_data(),
//          bilateral_out_blobs_[i]->gpu_data()
//          + bilateral_out_blobs_[i]->offset(n), (Dtype) 1.,
//          message_passings_[i]->mutable_gpu_data()
//          + message_passings_[i]->offset(n));
//    }

    //--------------------------- Compatibility multiplication ----------------
    //Result from message passing needs to be multiplied with compatibility values.
    for (int n = 0; n < num_; ++n) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
          channels_, (Dtype) 1., this->blobs_[2]->gpu_data(),
          message_passings_[i]->gpu_data() + message_passings_[i]->offset(n),
          (Dtype) 0., pairwise_.mutable_gpu_data() + pairwise_.offset(n));
    }
    // Adding unary and pairwise terms
    Dtype *top_data =
    i == num_iterations_ - 1 ?
    top[0]->mutable_gpu_data() :
    iteration_output_blobs_[i]->mutable_gpu_data();
    caffe_gpu_set<Dtype>(top[0]->count(), 0, top_data);
    caffe_gpu_axpy<Dtype>(bottom[0]->count(), 1.0, bottom[0]->gpu_data(),
        top_data);
    caffe_gpu_axpy<Dtype>(bottom[0]->count(), -1.0, pairwise_.gpu_data(),
        top_data);
  }
}

/**
 ** Backprop through filter-based mean field inference.
 **/

template<typename Dtype>
void DenseCRFMeanfieldLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (init_cpu) {
    LOG(FATAL)<< ("You initialize your network on CPU, please initialize it on GPU.");
  }
  caffe_gpu_set<Dtype>(bottom[0]->count(), Dtype(0.),
      bottom[0]->mutable_gpu_diff());
  caffe_set<Dtype>(this->blobs_[0]->count(), Dtype(0.),
      this->blobs_[0]->mutable_cpu_diff());
  caffe_set<Dtype>(this->blobs_[1]->count(), Dtype(0.),
      this->blobs_[1]->mutable_cpu_diff());
  caffe_gpu_set<Dtype>(this->blobs_[2]->count(), Dtype(0.),
      this->blobs_[2]->mutable_gpu_diff());


  for (int i = (num_iterations_ - 1); i >= 0; --i) {
    const Dtype *top_diff =
    i == num_iterations_ - 1 ?
    top[0]->gpu_diff() : iteration_output_blobs_[i]->gpu_diff();
    caffe_gpu_axpy<Dtype>(top[0]->count(), 1.0, top_diff,
        bottom[0]->mutable_gpu_diff());
    caffe_gpu_axpby<Dtype>(top[0]->count(), -1.0, top_diff, 0,
        pairwise_.mutable_gpu_diff());

    //---------------------------- Update compatibility diffs ------------------
    for (int n = 0; n < num_; ++n) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
          num_pixels_, (Dtype) 1., pairwise_.gpu_diff() + pairwise_.offset(n),
          message_passings_[i]->gpu_data() + message_passings_[i]->offset(n),
          (Dtype) 1., this->blobs_[2]->mutable_gpu_diff());
    }

    //-------------------------- Gradient after compatibility transform--- -----
    for (int n = 0; n < num_; ++n) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
          channels_, (Dtype) 1., this->blobs_[2]->gpu_data(),
          pairwise_.gpu_diff() + pairwise_.offset(n), (Dtype) 0.,
          message_passings_[i]->mutable_gpu_diff()
          + message_passings_[i]->offset(n));
    }

    // ------------------------- Gradient w.r.t. kernels weights ------------
    Dtype *spatial_filter_weight_diff = this->blobs_[0]->mutable_cpu_diff();
//    Dtype *bilateral_filter_weight_diff = this->blobs_[1]->mutable_cpu_diff();

    Dtype diff;
    caffe_gpu_dot<Dtype>(num_ * num_pixels_ * channels_,
        message_passings_[i]->gpu_diff(),
        spatial_out_blobs_[i]->gpu_data(), &diff);
    spatial_filter_weight_diff[0] += diff;


//    for (int n = 0; n < num_; ++n) {
//      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
//          num_pixels_, (Dtype) 1.,
//          message_passings_[i]->gpu_diff() + message_passings_[i]->offset(n),
//          spatial_out_blobs_[i]->gpu_data() + spatial_out_blobs_[i]->offset(n),
//          (Dtype) 1., this->blobs_[0]->mutable_gpu_diff());
//    }

    Dtype *bilateral_filter_weight_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_gpu_dot<Dtype>(num_ * num_pixels_ * channels_,
        message_passings_[i]->gpu_diff(), bilateral_out_blobs_[i]->gpu_data(),
        &diff);
    bilateral_filter_weight_diff[0] += diff;
//    for (int n = 0; n < num_; ++n) {
//      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
//          num_pixels_, (Dtype) 1.,
//          message_passings_[i]->gpu_diff() + message_passings_[i]->offset(n),
//          bilateral_out_blobs_[i]->gpu_data()
//          + bilateral_out_blobs_[i]->offset(n), (Dtype) 1.,
//          this->blobs_[1]->mutable_gpu_diff());
//    }

    // Gradient w.r.t spatial and bilateral output blob
    caffe_gpu_axpby<Dtype>(num_ * num_pixels_ * channels_,
        this->blobs_[0]->cpu_data()[0],
        message_passings_[i]->gpu_diff(),
        0., spatial_out_blobs_[i]->mutable_gpu_diff());
    for (int n = 0; n < num_; ++n) {
//      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
//          channels_, (Dtype) 1., this->blobs_[0]->gpu_data(),
//          message_passings_[i]->gpu_diff() + message_passings_[i]->offset(n),
//          (Dtype) 0.,
//          spatial_out_blobs_[i]->mutable_gpu_diff()
//          + spatial_out_blobs_[i]->offset(n));
    }
//    Dtype bilteral_filter_weight = 1.0 - this->blobs_[0]->cpu_data()[0];
    caffe_gpu_axpby<Dtype>(num_* num_pixels_*channels_,
        this->blobs_[1]->cpu_data()[0],
        message_passings_[i]->gpu_diff(),
        0.,bilateral_out_blobs_[i]->mutable_gpu_diff());
    for (int n = 0; n < num_; ++n) {
//      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
//          channels_, (Dtype) 1., this->blobs_[1]->gpu_data(),
//          message_passings_[i]->gpu_diff() + message_passings_[i]->offset(n),
//          (Dtype) 0.,
//          bilateral_out_blobs_[i]->mutable_gpu_diff()
//          + bilateral_out_blobs_[i]->offset(n));
    }

    //---------------------------- BP thru normalization --------------------------
    for (int n = 0; n < num_; ++n) {
      Dtype *spatial_out_diff = spatial_out_blobs_[i]->mutable_gpu_diff()
      + spatial_out_blobs_[i]->offset(n);
      for (int channel_id = 0; channel_id < channels_; ++channel_id) {
        caffe_gpu_mul<Dtype>(num_pixels_, spatial_norm_.gpu_data(),
            spatial_out_diff + channel_id * num_pixels_,
            spatial_out_diff + channel_id * num_pixels_);
      }

      Dtype *bilateral_out_diff = bilateral_out_blobs_[i]->mutable_gpu_diff()
      + bilateral_out_blobs_[i]->offset(n);
      for (int channel_id = 0; channel_id < channels_; ++channel_id) {
        caffe_gpu_mul<Dtype>(num_pixels_,
            bilateral_norms_.gpu_data() + bilateral_norms_.offset(n),
            bilateral_out_diff + channel_id * num_pixels_,
            bilateral_out_diff + channel_id * num_pixels_);
      }
    }

    //--------------------------- Gradient for message passing ---------------
    for (int n = 0; n < num_; ++n) {
      spatial_lattice_->compute(
          probs_[i]->mutable_gpu_diff() + probs_[i]->offset(n),
          spatial_out_blobs_[i]->gpu_diff() + spatial_out_blobs_[i]->offset(n),
          channels_, true, false);

      bilateral_lattices_[n]->compute(
          probs_[i]->mutable_gpu_diff() + probs_[i]->offset(n),
          bilateral_out_blobs_[i]->gpu_diff()
          + bilateral_out_blobs_[i]->offset(n), channels_, true, true);
    }

    vector<bool> propagate_down(2, true);
    softmax_layers_[i]->Backward(softmax_top_vec_vec_[i], propagate_down,
        softmax_bottom_vec_vec_[i]);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DenseCRFMeanfieldLayer);
}  // namespace caffe

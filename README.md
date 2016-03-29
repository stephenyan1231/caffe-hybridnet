# Caffe-HybridNet

Caffe-HybridNet implements the hybrid network as described in paper '[Combining the Best of Convolutional Layers and Recurrent Layers: A Hybrid Network for Semantic Segmentation](http://arxiv.org/abs/1603.04871)'.

Check out the [project site](https://sites.google.com/site/homepagezhichengyan/home/hybridnet) for more details.

## Code Installation
- git clone the branch 'hybridnet' into your local repository
- Follow the general instructions in the original Caffe and build up the Caffe-HybridNet. Specifically, type 'make -j;make pycaffe' in the root folder.
- A demo python script is provided at the path 'examples/voc12_SBD/python/demo.ipynb'. 

## License and Citation

Caffe-HybridNet is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).

Please cite the paper below in your publications if you find Caffe-HybridNet is helpful in your research:

    @article{yan2016combining,
      Title={Combining the Best of Convolutional Layers and Recurrent Layers: A Hybrid Network for Semantic Segmentation},
      Author = {Yan, Zhicheng and Zhang, Hao and Jia, Yangqing and Breuel, Thomas and Yu, Yizhou},
      Journal = {arXiv preprint arXiv:1603.04871},
      Year = {2016}
    }

# Diverse Feature Visualizations

This is a Python3 / [Tensorflow](https://www.tensorflow.org/) implementation of the methods proposed
in the paper:

[Diverse feature visualizations reveal invariances in early layers of deep neural networks](https://arxiv.org/abs/1807.10589),
by Santiago Cadena, Marissa Weis, Leon Gatys, Matthias Bethge, and Alexander Ecker.

Take a look at the two sample notebooks for the Diverse Visualizations of a paricular feature
map of VGG19, and the shift-invariance test propsed in the paper.

## Setup

To run this code you need the following:
- Python3
- Matplotlib
- Tensorflow 
- Download the checkpoint weights of the normalized VGG network [here](https://drive.google.com/open?id=1TvVGf2ClDARfSNfjbHLZLTtgHNe_jLVo) (80MB), as well as the pixelcnn++ [here](http://alpha.openai.com/pxpp.zip) or [here](https://www.dropbox.com/s/we5cltujdlhuxr8/pxpp.zip?dl=0) if the later is broken (656MB),
and store them in the networks/ folder

Our code uses the open-AI implementation of [PixelCNN++](https://openreview.net/pdf?id=BJrFC6ceg)
that can be found [here](https://github.com/openai/pixel-cnn).

## Citation

If you find our code useful please cite us in your work:

```
@article{cadena2018diverse,
  title={Diverse feature visualizations reveal invariances in early layers of deep neural networks},
  author={Cadena, Santiago A and Weis, Marissa A and Gatys, Leon A and Bethge, Matthias and Ecker, Alexander S},
  journal={arXiv preprint arXiv:1807.10589},
  year={2018}
}
```





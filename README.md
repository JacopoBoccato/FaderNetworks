# FaderNetworks for biological sequences

This repository is an adaptation of https://github.com/facebookresearch/FaderNetworks which allows to generate biological sequences using the same architecture as the one implemented in the original repository

## Model

<p align="center"><a href=https://github.com/facebookresearch/FaderNetworks/blob/master/images/v3.png?raw=true><img width="70%" src="./images/v3.png" /></a></p>

The main branch of the model (Inference Model), is an autoencoder of images. Given an image `x` and an attribute `y` (e.g. male/female), the decoder is trained to reconstruct the image from the latent state `E(x)` and `y`. The other branch (Adversarial Component), is composed of a discriminator trained to predict the attribute from the latent state. The encoder of the Inference Model is trained not only to reconstruct the image, but also to fool the discriminator, by removing from `E(x)` the information related to the attribute. As a result, the decoder needs to consider `y` to properly reconstruct the image. During training, the model is trained using real attribute values, but at test time, `y` can be manipulated to generate variations of the original image.

## Dependencies
* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [PyTorch](http://pytorch.org/)
* OpenCV
* CUDA

## References

If you find this code useful, please consider citing:

[*Fader Networks: Manipulating Images by Sliding Attributes*](https://arxiv.org/pdf/1706.00409.pdf) - G. Lample, N. Zeghidour, N. Usunier, A. Bordes, L. Denoyer, M'A. Ranzato

```
@inproceedings{lample2017fader,
  title={Fader Networks: Manipulating Images by Sliding Attributes},
  author={Lample, Guillaume and Zeghidour, Neil and Usunier, Nicolas and Bordes, Antoine and DENOYER, Ludovic and others},
  booktitle={Advances in Neural Information Processing Systems},
  pages={5963--5972},
  year={2017}
}
```

Contact: [jacopo.boccato@ipht.fr]

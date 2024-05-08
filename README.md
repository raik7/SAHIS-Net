# SAHIS-Net

SAHIS-Net: a spectral attention and feature enhancement network for microscopic hyperspectral cholangiocarcinoma image segmentation

Citation:  
@article{zhang2024sahis,
  title={SAHIS-Net: a spectral attention and feature enhancement network for microscopic hyperspectral cholangiocarcinoma image segmentation},
  author={Zhang, Yunchu and Dong, Jianfei},
  journal={Biomedical Optics Express},
  volume={15},
  number={5},
  pages={3147--3162},
  year={2024},
  publisher={Optica Publishing Group}
}

Requirements:  
TensorFlow = 2.4.0  
numpy = 1.19.5  
tqdm = 4.58.0  

1. Run HMUNetGenerate.py to generate the preproceessed hyperspectral images. The pre-trained weights of HM-UNet are provided as HMUNet.h5.
2. Run HMUNetTrain.py to train the HM-UNet.

# Multi-scale Convolutional Neural Networks (NCNN)
---

MCNN extends the functionality of the hidden layers in the decoder of a U-Net by connecting them to additional convolution layers to produce coarse outputs, in attempt to match the low-frequency components.
This greatly accelerates the convergence and enhances the stability of the neural-network. The convergence curve with U-net is shown in the figure blow.

![architectures](./misc/images/mcnn_architecture.jpg)

### quick tutorial
For people who is interested in applying MCNN to their own project, check out the `tutorial` folder.


### phase retrieval application
For the phase retrieval applications, please check out folder `phase_retrieval`;

### imageing objects from diffuse reflection
For the imaging objects from diffusive reflection application, please check out folder `diffuse_reconstruction`;

### denoising STEM images
For the STEM images denoising application, please check out folder `denoising`;



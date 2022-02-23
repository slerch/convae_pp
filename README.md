# Convolutional autoencoders for spatially-informed ensemble post-processing

This repository provides code accompanying the paper

> Lerch, S. and Polsterer, K.L. (2022).
> Convolutional autoencoders for spatially-informed ensemble post-processing.
> International Conference on Learning Represenatations (ICLR) 2022, AI for Earth and Space Science Workshop

Ensemble weather predictions typically show systematic errors that have to be corrected via post-processing. Even state-of-the-art post-processing methods based on neural networks often solely rely on location-specific predictors that require an interpolation of the physical weather model's spatial forecast fields to the target locations. However, potentially useful predictability information contained in large-scale spatial structures within the input fields is potentially lost in this interpolation step. Therefore, we propose the use of convolutional autoencoders to learn compact representations of spatial input fields which can then be used to augment location-specific information as additional inputs to post-processing models. The benefits of including this spatial information is demonstrated in a case study of 2-m temperature forecasts at surface stations in Germany. 

Here is a schematic illustration of the DRN+ConvAE model we propose:
![Schematic illustration of the DRN+ConvAE model.](https://github.com/slerch/convae_pp/blob/main/model_schematic.png?raw=true)

## Data

The data needed to reproduce the results consists of two input and one observation dataset. The station-based NWP predictors and the correspodinding observations are available in the follwoing: 

> Rasp, Stephan (2021): PPNN full data (feather format). figshare. Dataset. https://doi.org/10.6084/m9.figshare.13516301.v1 

The spatial input fields are too large to share in a straightforward way, they can be obtained from the TIGGE dataset (https://confluence.ecmwf.int/display/TIGGE), following the instructions provided in https://github.com/slerch/ppnn/tree/master/data_retrieval.



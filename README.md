# Overview


### Supervised Learning for Predicting Particle Matter Concentration :
In this task, we build convolutional neural network (CNN) model to predict Particle Matter (PM) values from webcam images. We build a unique CNN model for each unique location.

### Motivation
1) Investigate if haze features are relevant in predicting air qualty.
2) Investigate if time-series of images contain enough signal for deep learning models to detect during training and evaluation.

### Dataset :

1) Example:

<img src="https://github.com/cemanuel/air_pollution/blob/master/dataset_examples.png" width="350" height="250">

2) Location(s) 20+:

![Alt Text](https://github.com/cemanuel/air_pollution/blob/master/dataset_locations.png)

3) Normalizated Distrubtion of Particle Matter (PM) values:
<img src="https://github.com/cemanuel/air_pollution/blob/master/dataset_distribution.png" width="275" height="250">


### Workflow :
![Alt Text](https://github.com/cemanuel/air_pollution/blob/master/workflow.png)

1) Using a pretrained (CNN) model to output a transmission map from
2) Run the transmission map on a convolutional neural network module and output a prediction.
3) Compare the predictions from the CNN model to predictions made from doing a simple linear regression on hand-engineered features

### Hand Engineered Features (Related to Haze Detection) for Simple Linear Regression Analysis
1) Transmission Values:  Because a high concentration pollutants can affect the amount of sunlight we see in the sky, lower transmission values might correlate with higher Particle Matter (PM) values.

2) Dark Channel: Converts a RGB channel to a one-channel (grayscale) image where each pixel is the channel-wise maximum. Then, 1D kernel is spatially applied on the one-channel (grayscale image) A convolution applied spatially across the imaget that takes the median value of the patch.

3) Saturation: Contrast in image processing is usually de-fined as a ratio between the darkest and the brightest spotsof an image. We hope that images that is highly polluted, the ratio is low  and relatively high for less polluted images

4) Power Spectrum: We apply a fouier-transform on each of the images. In short, the smaller the value of the atmospheric light transmission function, the denser the fog (which can be composed of pollutants), the higher the frequency of the image, and the lower the spectral energy, and vice versa. Therefore, frequency-domain features can be used to judge if there is haze in the image.

5) Time: hour, day, month

6) Weather Features

### Results:


### Discussion:
1) For most of the sites, the CNN architecture did not perform as well as
2) For both CNN architecture and simple linear regression, we achieved higher accuracy on sites that had lower PM values than sites with higher PM values.

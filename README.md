# Overview


### Supervised Learning for Predicting Particle Matter Concentration :
In this task, we build convolutional neural network (CNN) model to predict Particle Matter (PM) values from webcam images. We build a unique CNN model for each unique location.

### Motivation
1) Investigate if haze features are relevant in predicting air qualty.
2) Investigate if time-series of images contain enough signal for deep learning models to detect during training and evaluation.

### Dataset :

1) Example(s):

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

### Results

![Alt Text](https://github.com/cemanuel/air_pollution/blob/master/results_air_quality.png)

### Discussion

1.) Pretrained models, VGG-16 and Resnet, achieved  higher R-2 coefficient scores than the baseline methods and our model, DehazeNet. As a result, there are other features besides haze features that is helping the model to quantify pollution in the images. However, these pretrained models might be overfitting on specific sites. A model was trained on an Alaskan site, achieving an R-2 coefficient of 0.38. When this same model was evaluated on a site in St.Louis, Missouri, the R-2 coefficient was 0.03.

2.) All models achieved better predictions for sites that had lower particle matter (PM) values. This can be explained by the distribution of PM values in the United States. Other studies in air pollution focused on countries with high pollution. As a result, the results in those studies can not be compared with our results due to differences in the distribution of PM values.

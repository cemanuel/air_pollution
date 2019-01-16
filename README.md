# Overview


### Supervised Learning for Predicting Particle Matter Concentration :
In this task, we build convolutional neural network (CNN) model to predict the particle matter concentration from webcam images. We build a unique CNN model for each cite we received from webcam images.

### Dataset :

### Workflow :
1) Using a pretrained (CNN) model to output a transmission map from
2) Run the transmission map on a convolutional neural network module and output a prediction.
3) Compare the predictions from the CNN model to predictions made from doing a simple linear regression on hand-engineering features

### Hand Engineered Features for Simple
1) Saturation
2) Transmission Value
3)


### Results:


### Discussion:
1) For most of the sites
2) We predict that our model can perform even better if we have the computing power to train our model on the full images, as a lot of information are lost during compression. We also expect improvements by including more training data or transferring learned models from similar works, such was ChestXNet.
3) If improved to human level performance, our weakly supervised model can not only automate pneumonia location annotation and classification tasks, but can also be used to localize other diseases. 

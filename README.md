# NeuralNetwork

In the previous part of this exercise, you implemented multi-class logistic regression to recognize handwritten digits. However, logistic regression cannot
form more complex hypotheses as it is only a linear classifier.3
In this part of the exercise, you will implement a neural network to recognize handwritten digits using the same training set as before. The neural
network will be able to represent complex models that form non-linear hypotheses. For this week, you will be using parameters from a neural network
that we have already trained. Your goal is to implement the feedforward
propagation algorithm to use our weights for prediction. In next week’s exercise, you will write the backpropagation algorithm for learning the neural
network parameters

## Model representation

Our neural network is shown in Figure 2. It has 3 layers { an input layer, a
hidden layer and an output layer. Recall that our inputs are pixel values of
digit images. Since the images are of size 20×20, this gives us 400 input layer
units (excluding the extra bias unit which always outputs +1). As before,
the training data will be loaded into the variables X and y.
You have been provided with a set of network parameters (Θ(1); Θ(2))
already trained by us. These are stored in ex3weights.mat and will be
loaded by ex3 nn into Theta1 and Theta2 The parameters have dimensions
that are sized for a neural network with 25 units in the second layer and 10
output units (corresponding to the 10 digit classes)

## Example of Neural Network

![screenshot at 2018-02-04 15-55-46](https://user-images.githubusercontent.com/30857998/35779438-e8834ffa-09c4-11e8-8a59-bae23e35ad48.png)

## Examples of data
![figure_1](https://user-images.githubusercontent.com/30857998/35779525-3d3c985c-09c6-11e8-833a-42276a1b020c.png)

![figure_2](https://user-images.githubusercontent.com/30857998/35779526-3d601a5c-09c6-11e8-8f7a-3d694f3cd96f.png)

## Feedforward Propagation and Prediction

Now you will implement feedforward propagation for the neural network. You
will need to complete the code in predict() to return the neural network’s
prediction.
You should implement the feedforward computation that computes hθ(x(i))
for every example i and returns the associated predictions. Similar to the
one-vs-all classification strategy, the prediction from the neural network will
be the label that has the largest output (hθ(x))k.
You should see that the accuracy is about 97.5%. After that, an interactive sequence will launch displaying images from the training set one at a time, while the console prints
out the predicted label for the displayed image. 


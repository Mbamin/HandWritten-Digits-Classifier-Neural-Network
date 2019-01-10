# HandWritten Digits Classifier Neural-Network
Can a Neural Network be used to accurately classify handwritten digits?
This project uses a 3 layer Neural Network along with back propagation to address the aforementioned problem. It works by observing the greyscale intensity values of various digits and using gradient descent along with logistic regression to learn what each digit looks like. This project was trained and tested on a subset of the MNIST data set, (5000 images) and is ~ 96% accurate. This project was inspired by an assignment from a Machine Learning course from Stanford.

The architecture of the Neural Network used in this project.
<p align="center">
  <img width="569" height="419" src="https://user-images.githubusercontent.com/32972284/50938746-9d8ce180-1447-11e9-940b-aa607e9d216c.png">
</p>

The formula for the regularized cost function used for back propagation
![image](https://user-images.githubusercontent.com/32972284/50938786-cad98f80-1447-11e9-8321-772c2be46f41.png)

A sample of 100 random images pulled from the dataset
<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/32972284/50940630-ae415580-144f-11e9-81b7-40e6902cff0e.png">
</p>

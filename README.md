A simple neural network trained on the MNIST dataset.
demo.py file can be used to test the network after training the model with train.py
Problem:
  demo.py's drawing symulation generates images looking differently from how the mnist images look, giving way worse results than testing on mnist.
  Could be solved by adding a convolution on the output of the drawing program before passing to network, making it more blurry (more like MNIST).

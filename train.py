import mnist_loader
import pickle
from network1 import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper(
    "/home/maksiu/neural-networks-and-deep-learning/data/mnist.pkl.gz"
)

net = network([784, 30, 10])
net.sgd(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

with open("trained_network.pkl", "wb") as f:
    pickle.dump(net, f)

print("Training complete. Network saved to trained_network.pkl")

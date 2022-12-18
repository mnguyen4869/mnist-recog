# Summary

Small neural network library able to create simple neural networks of varying size.

To build and train a neural network on the MNIST dataset run

```
make
./bin/main 3 0.15 784 32 16 10
```

This trains a neural network with 3 hidden layers and a learning rate of 0.15.
The input layer is made of 784 nodes and the subsequent integers are the number
of nodes in each hidden layer after.

# Summary

Small neural network library able to create simple neural networks of varying size.

To build and train a neural network on the MNIST dataset run

```
make
./bin/main 0.15 160000 784 32 16 10

arguments: 
learning rate, max iterations, # nodes per layer
```


This trains a neural network with 3 hidden layers and a learning rate of 0.15.
The neural net will train with a maximum of 160,000 iterations on the dataset.
The input layer is made of 784 nodes and the subsequent integers are the number
of nodes in each hidden layer after.

Relies on my [small matrix library](https://github.com/mnguyen4869/gram).
Modifications to the Makefile is needed to match the .so filepath.

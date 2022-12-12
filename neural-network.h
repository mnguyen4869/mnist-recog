#pragma once

#include <stdio.h>
#include <math.h>

typedef struct {
	double *weight;
	size_t size;
} node;

typedef struct {
	size_t num_layers; 		// # of layers 
	size_t *layer_size; 	// array holding # nodes per layer
	node **layers;
	node *output_layer;		// # the resulting output layer
} neural_net;

double ReLU(double value);

double ReLU_activation(double *values);

double ReLU_d(double value);

double ReLU_d_activation(double *values);

void update_weight(double *values, double alpha, double delta);

void set_rand_weights(void);

node *create_node(double *weights, size_t size);

neural_net *create_nn(size_t num_layers, size_t *layer_size);

double *feed_forward(double *data);

double *back_prop(double *data, double *expected, double alpha);

void save_nn(neural_net *nn, char *filename);

neural_net *load_nn(char *filename);

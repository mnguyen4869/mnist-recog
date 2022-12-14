#pragma once

#include "matrix.h"

typedef struct {
	matrix *weights;
	matrix *biases;
} layer;

typedef struct {
	size_t num_layers; 		// # of layers (including input and output)
	size_t *layer_size; 	// array holding # layers per layer
	layer **layers;
	layer *output_layer;		// # the resulting output layer
} neural_net;

double ReLU(double value);

matrix *ReLU_activation(matrix *input);

double ReLU_d(double value);

matrix *ReLU_d_activation(matrix *input);

layer *create_layer(matrix *weights, matrix *biases);

layer *create_rand_layer(size_t weight_rows, size_t weight_cols);

void free_layer(layer **l);

neural_net *create_nn(size_t num_layers, size_t *layer_size, layer *input);

void free_nn(neural_net **nn);

double *feed_forward(double *data);

double *back_prop(double *data, double *expected, double alpha);

void update_weight(double *values, double alpha, double delta);

void save_nn(neural_net *nn, char *filename);

neural_net *load_nn(char *filename);

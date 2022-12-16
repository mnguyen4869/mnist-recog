#pragma once

#include "matrix.h"
#include "matops.h"

typedef struct {
	matrix *weights;
	matrix *biases;
} layer;

typedef struct {
	size_t num_h_layers; 		// # of hidden layers
	size_t *layer_size; 	// # nodes per layer (including input and excluding output)
	layer **layers;
} neural_net;

double ReLU(double value);

double ReLU_d(double value);

layer *create_layer(matrix *weights, matrix *biases);

layer *create_rand_layer(size_t weight_rows, size_t weight_cols);

void free_layer(layer **l);

neural_net *create_nn(size_t num_layers, size_t *layer_size);

void free_nn(neural_net **nn);

matrix **feed_forward(neural_net *nn, matrix *input);

void back_prop(neural_net *nn, matrix *data, matrix *expected, double alpha, size_t data_s);

void update_weight(double *values, double alpha, double delta);

void save_nn(neural_net *nn, char *filename);

neural_net *load_nn(char *filename);

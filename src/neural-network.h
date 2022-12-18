#pragma once

#include <math.h>

#include "matrix.h"
#include "matops.h"

typedef struct {
	matrix *weights;
	matrix *biases;
} layer;

typedef struct {
	size_t num_h_layers; 		// # of hidden layers
	size_t *layer_size; 		// # nodes per layer (including input and excluding output)
	layer **layers;				// hidden layers
} neural_net;

double ReLU(double value);

double ReLU_d(double value);

double sigmoid(double value);

double sigmoid_d(double value);

matrix *softmax(matrix *input);

matrix *apply_activation(layer *layer, matrix *input, double (*act)(double));

layer *create_layer(matrix *weights, matrix *biases);

layer *create_rand_layer(size_t weight_rows, size_t weight_cols);

void free_layer(layer **l);

neural_net *create_nn(size_t num_layers, size_t *layer_size);

void free_nn(neural_net **nn);

matrix **feed_forward_no_act(neural_net *nn, matrix *input);

matrix **feed_forward(neural_net *nn, matrix *input, double (*activation)(double));

double back_prop(neural_net *nn, matrix *data, matrix *expected, double alpha,
				double (*act)(double), double (*act_d)(double));

void update_weight_biases(neural_net *nn, double alpha, matrix *input,
				matrix **out_layers, matrix **deltas);

void save_nn(neural_net *nn, char *filename);

neural_net *load_nn(char *filename);

#include "idx_parse.h"
#include "neural-network.h"

double ReLU(double value)
{
	if (value > 0) {
		return value;
	}
	return 0;
}

double ReLU_d(double value)
{
	return (value > 0);
}

layer *create_layer(matrix *weights, matrix *biases)
{
	layer *l;
	if ((l = malloc(sizeof(*l))) == NULL) {
		fprintf(stderr, "Malloc on layer creation failed");
		exit(1);
	}
	l->biases = biases;
	l->weights = weights;
	return l;
}

layer *create_rand_layer(size_t rows, size_t cols)
{
	layer *l;
	if ((l = malloc(sizeof(*l))) == NULL) {
		fprintf(stderr, "Malloc on layer creation failed");
		exit(1);
	}
	l->biases = grm_create_mat(rows, 1);
	grm_fill_mat(l->biases, 1);
	l->weights = grm_create_rand_mat(rows, cols, -1, 1);
	return l;
}

void free_layer(layer **l)
{
	grm_free_mat(&((*l)->weights));
	grm_free_mat(&((*l)->biases));
	free(*l);
	*l = NULL;
}

neural_net *create_nn(size_t num_h_layers, size_t *layer_size)
{
	neural_net *nn;
	if ((nn = malloc(sizeof(*nn))) == NULL) {
		fprintf(stderr, "Malloc on neural net failed");
		exit(1);
	}
	nn->num_h_layers = num_h_layers;
	nn->layer_size = layer_size;
	if ((nn->layers = malloc(sizeof(*nn->layers) * num_h_layers)) == NULL) {
		fprintf(stderr, "Malloc on layers of neural net failed");
		exit(1);
	}
	for (size_t i = 0; i < num_h_layers; i++) {
		nn->layers[i] = create_rand_layer(layer_size[i + 1], layer_size[i]);
	}
	return nn;
}

void free_nn(neural_net **nn)
{
	for (size_t i = 0; i < (*nn)->num_h_layers; i++) {
		free_layer(&((*nn)->layers[i]));
	}
	free((*nn)->layers);
	free(*nn);
	*nn = NULL;
}

matrix **feed_forward(neural_net *nn, matrix *input)
{
	matrix **output;
	if ((output = malloc(sizeof(*output) * nn->num_h_layers)) == NULL) {
		fprintf(stderr, "Malloc on feed_forward output failed");
		exit(1);
	}

	for (size_t i = 0; i < nn->num_h_layers; i++) {
		matrix *w_l;
		// input layer
		if (i == 0) {
			w_l = grm_multiply(nn->layers[i]->weights, input);
			grm_free_mat(&input);
		}
		else {
			w_l = grm_multiply(nn->layers[i]->weights, output[i - 1]);
		}
		matrix *z = grm_add(w_l, nn->layers[i]->biases);
		output[i] = grm_apply(ReLU, z);
		grm_free_mat(&w_l);
		grm_free_mat(&z);
	}
	return output;
}

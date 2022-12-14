#include "idx_parse.h"
#include "neural-network.h"

double ReLU(double value)
{
	if (value > 0) {
		return value;
	}
	return 0;
}

// matrix *ReLU_activation(matrix *input)
// {
// 	return 0;
// }

double ReLU_d(double value)
{
	return (value > 0);
}

// matrix *ReLU_d_activation(matrix *input)
// {
// 	return 0;
// }

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

neural_net *create_nn(size_t num_layers, size_t *layer_size, layer *input)
{
	neural_net *nn;
	if ((nn = malloc(sizeof(*nn))) == NULL) {
		fprintf(stderr, "Malloc on neural net failed");
		exit(1);
	}
	nn->num_layers = num_layers;
	nn->layer_size = layer_size;
	if ((nn->layers = malloc(sizeof(*nn->layers) * num_layers)) == NULL) {
		fprintf(stderr, "Malloc on layers of neural net failed");
		exit(1);
	}
	nn->layers[0] = input;
	for (size_t i = 1; i < num_layers; i++) {
		nn->layers[i] = create_rand_layer(layer_size[i], layer_size[i - 1]);
	}
	return nn;
}

void free_nn(neural_net **nn)
{
	for (size_t i = 0; i < (*nn)->num_layers; i++) {
		free_layer(&((*nn)->layers[i]));
	}
	free((*nn)->layers);
	free(*nn);
	*nn = NULL;
}

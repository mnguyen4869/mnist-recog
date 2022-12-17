#include "idx_parse.h"
#include "neural-network.h"

#define UNUSED_PARAMETER(param) (void)(param)

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

double sigmoid(double value)
{
	return 1 / (1 + exp(-value));
}

double sigmoid_d(double value)
{
	return sigmoid(value) * (1 - sigmoid(value));
}

matrix *apply_activation(layer *layer, matrix *input, double (*act)(double))
{
	// apply weights on input
	matrix *weights_input = grm_dot(layer->weights, input);
	// apply bias on result of previous
	matrix *z = grm_add_scalar(layer->bias, weights_input);
	// activation
	matrix *out = grm_apply(act, z);
	grm_free_mat(&weights_input);
	grm_free_mat(&z);
	return out;
}

layer *create_layer(matrix *weights, double bias)
{
	layer *l;
	if ((l = malloc(sizeof(*l))) == NULL) {
		fprintf(stderr, "Malloc on layer creation failed");
		exit(1);
	}
	l->bias = bias;
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
	l->bias = 1;
	l->weights = grm_create_rand_mat(rows, cols, -1, 1);
	return l;
}

void free_layer(layer **l)
{
	grm_free_mat(&((*l)->weights));
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

matrix **feed_forward(neural_net *nn, matrix *input, double (*activation)(double))
{
	matrix **output;
	if ((output = malloc(sizeof(*output) * nn->num_h_layers)) == NULL) {
		fprintf(stderr, "Malloc on feed_forward output failed");
		exit(1);
	}

	for (size_t i = 0; i < nn->num_h_layers; i++) {
		// input layer
		if (i == 0) {
			output[i] = apply_activation(nn->layers[i], input, activation);
		}
		else {
			output[i] = apply_activation(nn->layers[i], output[i - 1], activation);
		}
	}
	return output;
}

void back_prop(neural_net *nn, matrix *data, matrix *expected, double alpha,
				double (*act)(double), double (*act_d)(double))
{
	size_t num_h_layers = nn->num_h_layers;
	matrix **output_layers = feed_forward(nn, data, act);
	matrix **deltas;
	double *bias_deltas;

	if ((deltas = malloc(sizeof(*deltas) * num_h_layers)) == NULL) {
		fprintf(stderr, "Malloc on deltas failed");
		exit(1);
	}

	if ((bias_deltas = malloc(sizeof(*bias_deltas) * num_h_layers)) == NULL) {
		fprintf(stderr, "Malloc on deltas failed");
		exit(1);
	}

	matrix *error = grm_subtract(output_layers[num_h_layers - 1], expected);

	// derivative of previous layer
	matrix *input_d = apply_activation(nn->layers[num_h_layers - 1],
			output_layers[num_h_layers - 2], act_d);

	deltas[num_h_layers - 1] = grm_multiply(error, input_d);

	// output of previous layer
	// matrix *d = grm_create_mat(1, 2);
	// grm_fill_mat(d, 1);
	// deltas[num_h_layers - 1] = grm_dot(err_times_in, d);
	// grm_free_mat(&d);

	printf("result\n");
	grm_print_mat(deltas[num_h_layers - 1]);
	printf("\n");

	// continue for all other hidden layers in reverse order
	for (size_t i = num_h_layers - 1; i > 0; i--) {
		printf("cur mat\n");
		grm_print_mat(nn->layers[i - 1]->weights);
		printf("\n");
		if (i == 1) {
			matrix *g_prime = apply_activation(nn->layers[i - 1], data, act_d);
			matrix *t = grm_transpose(nn->layers[i]->weights);
			matrix *summation = grm_dot(t, deltas[i]);
			matrix *delta_i = grm_multiply(g_prime, summation);
			printf("delta\n");
			grm_print_mat(delta_i);
			grm_free_mat(&g_prime);
			grm_free_mat(&t);
			grm_free_mat(&summation);
			deltas[i - 1] = delta_i;
		}
		// else {

		// }
		printf("\n");
	}

	update_weight_biases(nn, alpha, output_layers, deltas);

	for (size_t i = 0; i < nn->num_h_layers; i++) {
		grm_free_mat(&(output_layers[i]));
	}
}

// void update_weight_biases(neural_net *nn, double alpha,
// 		matrix **out_layers, matrix **deltas)
// {
// 	// update weights
// 	for (size_t i = 0; i < nn->num_h_layers; i++) {
// 		matrix *diff = grm_scale(alpha, deltas[i]);
// 		nn->layers[i]->weights = grm_subtract(nn->layers[i]->weights, diff);
// 		grm_free_mat(&diff);
// 	}
// 	// update biases
// }

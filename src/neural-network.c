#include "idx_parse.h"
#include "neural-network.h"

#define MAXSIZE 150

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

matrix *softmax(matrix *input)
{
	matrix *n = grm_create_mat(input->num_rows, input->num_cols);
	size_t dim = input->num_rows * input->num_cols;
	double sum = 0;

	for (size_t i = 0; i < dim; i++) {
		sum += exp(input->data[i]);
	}
	for (size_t i = 0; i < dim; i++) {
		n->data[i] += exp(input->data[i]) / sum;
	}
	return n;
}

matrix *apply_activation(layer *layer, matrix *input, double (*act)(double))
{
	// apply weights on input
	matrix *weights_input = grm_dot(layer->weights, input);
	// apply bias on result of previous
	matrix *z = grm_add(layer->biases, weights_input);
	// activation
	matrix *out = grm_apply(act, z);
	grm_free_mat(&weights_input);
	grm_free_mat(&z);
	return out;
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
	l->biases = grm_create_rand_mat(rows, 1, -0.5, 0.5);
	l->weights = grm_create_rand_mat(rows, cols, -0.5, 0.5);
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

double back_prop(neural_net *nn, matrix *data, matrix *expected, double alpha,
				double (*act)(double), double (*act_d)(double))
{
	size_t num_h_layers = nn->num_h_layers;
	matrix **output_layers = feed_forward(nn, data, act);
	matrix **deltas;
	double abs_error = 0;

	if ((deltas = malloc(sizeof(*deltas) * num_h_layers)) == NULL) {
		fprintf(stderr, "Malloc on deltas failed");
		exit(1);
	}

	// derivative of previous layer
	matrix *input_d = apply_activation(nn->layers[num_h_layers - 1],
			output_layers[num_h_layers - 2], act_d);

	matrix *error = grm_subtract(expected, output_layers[num_h_layers - 1]);
	for (size_t i = 0; i < error->num_rows * error->num_cols; i++) {
		abs_error += (error->data[i] * error->data[i]) / 2.0;
	}

	deltas[num_h_layers - 1] = grm_multiply(error, input_d);

	// continue for all other hidden layers in reverse order
	for (size_t i = num_h_layers - 1; i > 0; i--) {
		matrix *g_prime;
		if (i == 1) {
			g_prime = apply_activation(nn->layers[i - 1], data, act_d);
		}
		else {
			g_prime = apply_activation(nn->layers[i - 1], output_layers[i - 2], act_d);
		}
		matrix *t = grm_transpose(nn->layers[i]->weights);
		matrix *summation = grm_dot(t, deltas[i]);
		matrix *delta_i = grm_multiply(g_prime, summation);

		grm_free_mat(&g_prime);
		grm_free_mat(&t);
		grm_free_mat(&summation);
		deltas[i - 1] = delta_i;
	}

	update_weight_biases(nn, alpha, data, output_layers, deltas);

	for (size_t i = 0; i < nn->num_h_layers; i++) {
		grm_free_mat(&(output_layers[i]));
		grm_free_mat(&(deltas[i]));
	}

	grm_free_mat(&input_d);
	grm_free_mat(&error);
	free(output_layers);
	free(deltas);
	return abs_error;
}

void update_weight_biases(neural_net *nn, double alpha, matrix *input,
		matrix **out_layers, matrix **deltas)
{
	matrix *t;
	for (size_t i = 0; i < nn->num_h_layers; i++) {

		// update biases
		matrix *alpha_scaled_delta = grm_scale(alpha, deltas[i]);
		matrix *new_biases = grm_add(nn->layers[i]->biases, alpha_scaled_delta);
		grm_free_mat(&(nn->layers[i]->biases));
		nn->layers[i]->biases = new_biases;

		if (i == 0) {
			t = grm_transpose(input);
		}
		else {
			t = grm_transpose(out_layers[i - 1]);
		}

		// update weights
		matrix *before_a = grm_dot(deltas[i], t);
		matrix *alpha_mat = grm_scale(alpha, before_a);
		matrix *new_weights = grm_add(nn->layers[i]->weights, alpha_mat); 

		grm_free_mat(&alpha_scaled_delta);
		grm_free_mat(&t);
		grm_free_mat(&before_a);
		grm_free_mat(&alpha_mat);
		grm_free_mat(&(nn->layers[i]->weights));
		nn->layers[i]->weights = new_weights;
	}
}

void save_nn(neural_net *nn, char *filename)
{
	FILE *file = fopen(filename, "w");
	if (file == NULL) {
		fprintf(stderr, "Opening file failed");
		return;
	}
	fprintf(file, "%ld\n", nn->num_h_layers);
	for (size_t i = 0; i < nn->num_h_layers + 1; i++) {
		fprintf(file, "%ld\n", nn->layer_size[i]);
	}
	for (size_t i = 0; i < nn->num_h_layers; i++) {
		layer *curr = nn->layers[i];
		fprintf(file, "%ld\n", curr->weights->num_rows);
		fprintf(file, "%ld\n", curr->weights->num_cols);
		size_t curr_dim = curr->weights->num_rows * curr->weights->num_cols;
		for (size_t j = 0; j < curr_dim; j++) {
			fprintf(file, "%.10f\n", curr->weights->data[j]);
		}
		for (size_t j = 0; j < curr->weights->num_rows; j++) {
			fprintf(file, "%.10f\n", curr->biases->data[j]);
		}
	}
	fclose(file);
}

neural_net *load_nn(char *filename)
{
	FILE *file = fopen(filename, "r");
	if (file == NULL) {
		fprintf(stderr, "Opening file failed");
		return NULL;
	}
	char entry[MAXSIZE]; 
	fgets(entry, MAXSIZE, file);
	size_t num_h_layers = atoi(entry);

	size_t *layer_size;
	if ((layer_size = malloc(sizeof(*layer_size) * (num_h_layers + 1))) == NULL) {
		fprintf(stderr, "Malloc on layer load failed");
		exit(1);
	}
	for (size_t i = 0; i < num_h_layers + 1; i++) {
		fgets(entry, MAXSIZE, file);
		layer_size[i] = atoi(entry);
	}
	neural_net *nn = create_nn(num_h_layers, layer_size);
	for (size_t i = 0; i < num_h_layers; i++) {
		fgets(entry, MAXSIZE, file);
		size_t num_rows = atoi(entry);
		fgets(entry, MAXSIZE, file);
		size_t num_cols = atoi(entry);

		matrix *weights = grm_create_mat(num_rows, num_cols);
		matrix *biases = grm_create_mat(num_rows, 1);
		for (size_t j = 0; j < num_rows * num_cols; j++) {
			fgets(entry, MAXSIZE, file);
			double data = strtod(entry, NULL);
			weights->data[j] = data;
		}
		for (size_t j = 0; j < num_rows; j++) {
			fgets(entry, MAXSIZE, file);
			double data = strtod(entry, NULL);
			biases->data[j] = data;
		}
		free_layer(&(nn->layers[i]));
		nn->layers[i] = create_layer(weights, biases);
	}
	fclose(file);
	return nn;
}

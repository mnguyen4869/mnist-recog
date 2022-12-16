#include "neural-net-tests.h"

bool test_ReLU(void)
{
	double x = ReLU(10);
	double y = ReLU(-1);
	return (x == 10 && y == 0);
}

bool test_ReLU_activation(void)
{
	matrix *m = grm_create_mat(4, 1);
	double data[4] = {-2, -1, 1, 4};
	double expected[4] = {0, 0, 1, 4};
	grm_copy_data(m, data, 4);
	matrix *n = grm_apply(ReLU, m);
	size_t dim = m->num_rows * m->num_cols;
	for (size_t i = 0; i < dim; i++) {
		if (n->data[i] != expected[i]) {
			return false;
		}
	}
	grm_free_mat(&m);
	grm_free_mat(&n);
	return true;
}

bool test_ReLU_d(void)
{
	double x = ReLU_d(10);
	double y = ReLU_d(-1);
	return (x == 1 && y == 0);
}

bool test_ReLU_d_activation(void)
{
	return true;
}

bool test_free_layer(void)
{
	layer *l = create_rand_layer(10, 10);
	free_layer(&l);
	return (l == NULL);
}

bool test_free_nn(void)
{
	size_t n_layers = 2;
	size_t layer_s[3] = {2, 16, 10};
	neural_net *nn = create_nn(n_layers, layer_s);
	if (nn->layers[0]->weights->num_cols != 2 || nn->layers[0]->weights->num_rows != 16) {
		return false;
	}
	if (nn->layers[1]->weights->num_cols != 16 || nn->layers[1]->weights->num_rows != 10) {
		return false;
	}
	free_nn(&nn);
	return (nn == NULL);
}

bool test_feed_forward(void)
{
	size_t n_layers = 2;
	size_t layer_s[3] = {2, 3, 2};

	// input layer
	matrix *in = grm_create_mat(2, 1);
	double data[2] = {1, 0};
	grm_copy_data(in, data, 2);

	// hidden layer
	matrix *hi = grm_create_mat(3, 2);
	double hi_weights[6] = {-0.5, -0.2, 0.2, 0.1, 0.8, 0.7};
	grm_copy_data(hi, hi_weights, 6);
	matrix *hi_empty = grm_create_mat(3, 1);
	grm_fill_mat(hi_empty, 0);
	layer *hi_layer = create_layer(hi, hi_empty);

	// output layer
	matrix *ou = grm_create_mat(2, 3);
	double ou_weights[6] = {0.5, -0.4, -0.1, 0.3, 0.6, 0.1};
	grm_copy_data(ou, ou_weights, 6);
	matrix *ou_one = grm_create_mat(2, 1);
	grm_fill_mat(ou_one, 1);
	layer *ou_layer = create_layer(ou, ou_one);

	// neural_net *nn = create_nn(n_layers, layer_s, in_layer);
	neural_net *nn = create_nn(n_layers, layer_s);
	free_layer(&(nn->layers[0]));
	nn->layers[0] = hi_layer;
	free_layer(&(nn->layers[1]));
	nn->layers[1] = ou_layer;

	matrix **result = feed_forward(nn, in);
	double expected_out1[3] = {0, 0.2, 0.8};
	double expected_out2[2] = {0.84, 1.2};

	for (size_t i = 0; i < result[0]->num_rows * result[0]->num_cols; i++) {
		if (result[0]->data[i] != expected_out1[i]) {
			return false;
		}
	}
	for (size_t i = 0; i < result[1]->num_rows * result[1]->num_cols; i++) {
		if (result[1]->data[i] != expected_out2[i]) {
			return false;
		}
	}

	for (size_t i = 0; i < nn->num_h_layers; i++) {
		grm_free_mat(&(result[i]));
	}

	free(result);
	free_nn(&nn);

	return true;
}

bool test_back_prop(void)
{
	return true;
}

bool test_update_weight(void)
{
	return true;
}

bool test_save_load_nn(void)
{
	return true;
}

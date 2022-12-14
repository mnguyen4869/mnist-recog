#include "neural-net-tests.h"

bool test_ReLU(void)
{
	double x = ReLU(10);
	double y = ReLU(-1);
	return (x == 10 && y == 0);
}

bool test_ReLU_activation(void)
{
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
	size_t n_layers = 3;
	size_t layer_s[3] = {2, 16, 10};
	layer *in_layer = create_rand_layer(2, 1);
	neural_net *nn = create_nn(n_layers, layer_s, in_layer);
	// check layer data is okay
	if (nn->layers[0]->weights->num_cols != 1 || nn->layers[0]->weights->num_rows != 2) {
		return false;
	}
	if (nn->layers[1]->weights->num_cols != 2 || nn->layers[1]->weights->num_rows != 16) {
		return false;
	}
	if (nn->layers[2]->weights->num_cols != 16 || nn->layers[2]->weights->num_rows != 10) {
		return false;
	}
	free_nn(&nn);
	return (nn == NULL);
}

bool test_feed_forward(void)
{
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

#include "neural-net-tests.h"

int main ()
{
	srand(time(0)); // random seed for rand_mat

	assert(test_ReLU());

	assert(test_ReLU_activation());

	assert(test_ReLU_d());

	assert(test_ReLU_d_activation());

	assert(test_free_layer());

	assert(test_free_nn());

	assert(test_feed_forward());

	assert(test_back_prop());

	assert(test_update_weight());

	printf("Passed all neural net tests\n");

	return 0;
}

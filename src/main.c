#include <time.h>

#include "neural-network.h"
#include "idx_parse.h"

int main (int argc, char *argv[])
{
	if ((size_t) argc <= 1) {
		fprintf(stderr, "Incorrect amount of arguments.\n"
				"learning rate, max iterations, # of nodes per layer including input\n");
		exit(1);
	}

	srand(time(0));
	
	// MNIST dataset
	matrix **training_images = (matrix**) parse_idxfile("data/train/train-images-idx3-ubyte");
	matrix *training_labels = (matrix*) parse_idxfile("data/train/train-labels-idx1-ubyte");
	matrix **testing_images = (matrix**) parse_idxfile("data/test/t10k-images-idx3-ubyte");
	matrix *testing_labels = (matrix*) parse_idxfile("data/test/t10k-labels-idx1-ubyte");

	// KMNIST dataset
	// matrix **training_images = (matrix**) parse_idxfile("data/train/kmnist-train-images-idx3-ubyte");
	// matrix *training_labels = (matrix*) parse_idxfile("data/train/kmnist-train-labels-idx1-ubyte");
	// matrix **testing_images = (matrix**) parse_idxfile("data/test/kmnist-t10k-images-idx3-ubyte");
	// matrix *testing_labels = (matrix*) parse_idxfile("data/test/kmnist-t10k-labels-idx1-ubyte");

	for (int i = 0; i < argc; i++) {
		printf("arg %d: %s \n", i, argv[i]);
	}

	size_t num_layers = argc - 4;
	double learning_rate = strtod(argv[1], NULL);
	size_t max_iters = atoi(argv[2]);

	if ((size_t) argc != num_layers + 4) {
		fprintf(stderr, "Incorrect amount of arguments.\n"
				"learning rate, max iterations, # of nodes per layer including input\n");
		exit(1);
	}

	size_t *layer_size;
	if ((layer_size = malloc(sizeof(*layer_size) * (num_layers + 1))) == NULL) {
		fprintf(stderr, "Malloc on layer size failed\n");
		exit(1);
	}

	printf("Neural Network\n"
			"Learning Rate: %1.4f\n"
			"# Hidden Layers: %ld\n"
			"Layer Sizes:", learning_rate, num_layers);

	for (size_t i = 0; i < num_layers + 1; i++) {
		layer_size[i] = atoi(argv[i + 3]);
		printf(" %ld", layer_size[i]);
	}
	printf("\n\n");

	double training_error = 0;
	size_t test_correct = 0;
	size_t test_total = 0;

	neural_net *nn = create_nn(num_layers, layer_size);

	for (size_t i = 0; i < max_iters; i++) {
		matrix *expected = grm_create_mat(10, 1);
		grm_fill_mat(expected, 0);
		expected->data[(int) training_labels->data[i % 60000]] = 1;
		matrix *input = grm_flatten_mat(training_images[i % 60000], 1);
		// scaling data to be [0, 1]
		matrix *input_s = grm_scale((double) 1 / (double) 255, input);
		training_error += (back_prop(nn, input_s, expected, learning_rate, sigmoid, sigmoid_d));
		// training_error += (back_prop(nn, input_s, expected, learning_rate, ReLU, ReLU_d));
		grm_free_mat(&input);
		grm_free_mat(&input_s);
		grm_free_mat(&expected);
		// grm_free_mat(&(training_images[i % 60000]));
		if (i % 1000 == 0 && i != 0) {
			printf("Iteration %ld\n", i);
			printf("Accuracy: %f\n\n", 1.0 - (training_error / i));
		}
	}

	// freeing stored training images
	for (size_t i = 0; i < 60000; i++) {
		grm_free_mat(&(training_images[i % 60000]));
	}

	printf("Training is done :D\n");
	printf("Training Accuracy: %f\n\n", 1.0 - (training_error / 160000));

	for (size_t i = 0; i < 10000; i++) {
		matrix *input_test = grm_flatten_mat(testing_images[i], 1);
		matrix *input_test_s = grm_scale((double )1 / (double) 255, input_test);
		// scaling data to be [0, 1]
		matrix **result = feed_forward(nn, input_test_s, sigmoid);
		// matrix **result = feed_forward(nn, input_test_s, ReLU);

		if (grm_argmax_mat(result[num_layers - 1]) == testing_labels->data[i]) {
			test_correct++;
		}
		test_total++;

		for (size_t i = 0; i < num_layers; i++) {
			grm_free_mat(&(result[i]));
		}
		free(result);
		grm_free_mat(&input_test);
		grm_free_mat(&input_test_s);
		grm_free_mat(&(testing_images[i]));
	}

	double test_acc = (double) test_correct / test_total;
	printf("Test Accuracy %f\n", test_acc);
	printf("Test Accuracy %ld / %ld\n", test_correct, test_total);

	free(training_images);
	grm_free_mat(&training_labels);
	free(testing_images);
	grm_free_mat(&testing_labels);
	free_nn(&nn);
	return 0;
}

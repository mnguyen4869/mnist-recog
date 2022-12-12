#include "idx_parse.h"

void print_image(u8 *image, int height, int width)
{
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%.3d ", image[(i * width) +  j]);
		}
		printf("\n");
	}
}

void *parse_idxfile(char *filestr)
{
	void *result = {0};
	FILE *file = fopen(filestr, "r");

	// getting magic number
	int32_t magic_num_p[1];
	if (fread(magic_num_p, sizeof(*magic_num_p), 1, file) != 1) {
		exit(1);
	}
	int32_t magic_num = bswap_32(*magic_num_p);

	u8 datatype = (u8)((magic_num & (255 << 8)) >> 8);
	u8 dimensions = (u8)(magic_num & (255));

	int32_t *sizeof_dims = malloc(dimensions);
	if (fread(sizeof_dims, sizeof(*sizeof_dims), dimensions, file) != dimensions) {
		exit(1);
	}

	printf("%d\n", magic_num);
	printf("%d\n", datatype);
	printf("%d\n", dimensions);

	for (u8 i = 0; i < dimensions; i++) {
		sizeof_dims[i] = bswap_32(sizeof_dims[i]);
		printf("dimension %d size %d\n", i, sizeof_dims[i]);
	}

	// MAKE THIS MORE GENERIC LATER

	if (dimensions == 1) {
		u8 *data;
		if ((data = malloc(sizeof(*data) * sizeof_dims[0])) == NULL) {
			fprintf(stderr, "Malloc failed");
			exit(1);
		}
		if (fread(data, sizeof(*data), sizeof_dims[0], file) != sizeof_dims[0]) {
			exit(1);
		}
		// for (u_int32_t i = 0; i < 3; i++) {
		// 	printf("%d\n", data[i]);
		// }
		result = data;
	}

	if (dimensions == 3) {
		u8 *data; 
		if ((data = malloc(sizeof(*data) * sizeof_dims[0] * sizeof_dims[1] * sizeof_dims[2])) == NULL) {
			fprintf(stderr, "Malloc failed");
			exit(1);
		}
		if (fread(data, sizeof(u8) * sizeof_dims[0] * sizeof_dims[1] * sizeof_dims[2], 1, file) != 1) {
			fprintf(stderr, "fread failed");
			exit(1);
		}

		// for (u_int32_t i = 0; i < 3; i++) {
		// 	print_image(&data[i * sizeof_dims[1] * sizeof_dims[2], sizeof_dims[2], sizeof_dims[1]]);
		// 	printf("\n");
		// 	printf("\n");
		// 	printf("\n");
		// }
		result = data;
	}
	fclose(file);
	return result;
}

// int main (int argc, char *argv[])
// {
// 	parse_idxfile("./test/t10k-labels-idx1-ubyte");
// 	parse_idxfile("./test/t10k-images-idx3-ubyte");
// 	return 0;
// }

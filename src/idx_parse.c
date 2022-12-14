#include "idx_parse.h"

void *parse_idxfile(char *filestr)
{
	matrix *result = NULL;
	FILE *file = fopen(filestr, "r");
	if (file == NULL) {
		printf("Invalid file path, %s\n", filestr);
		exit(1);
	}

	// getting magic number
	int32_t magic_num_p[1];
	if (fread(magic_num_p, sizeof(*magic_num_p), 1, file) != 1) {
		exit(1);
	}
	int32_t magic_num = bswap_32(*magic_num_p);

	// u8 datatype = (u8)((magic_num & (255 << 8)) >> 8);
	u8 dimensions = (u8)(magic_num & (255));

	int32_t sizeof_dims[dimensions];
	if (fread(sizeof_dims, sizeof(*sizeof_dims), dimensions, file) != dimensions) {
		exit(1);
	}

	for (u8 i = 0; i < dimensions; i++) {
		sizeof_dims[i] = bswap_32(sizeof_dims[i]);
	}

	// MAKE THIS MORE GENERIC LATER

	if (dimensions == 1) {
		u8 *data;
		if ((data = malloc(sizeof(*data) * sizeof_dims[0])) == NULL) {
			fprintf(stderr, "Malloc failed");
			exit(1);
		}
		if (fread(data, sizeof(*data), sizeof_dims[0], file) != (size_t) sizeof_dims[0]) {
			exit(1);
		}
		matrix *result = grm_create_mat(sizeof_dims[0], 1);
		for (int32_t i = 0; i < sizeof_dims[0]; i++) {
			result->data[i] = (double) data[i];
		}
		free(data);
		fclose(file);
		return result; // returns matrix *
	}

	if (dimensions == 3) {
		u8 *data; 
		int32_t chunk_size = sizeof_dims[0] * sizeof_dims[1] * sizeof_dims[2];
		if ((data = malloc(sizeof(*data) * chunk_size)) == NULL) {
			fprintf(stderr, "Malloc failed");
			exit(1);
		}
		if (fread(data, sizeof(*data) * chunk_size, 1, file) != 1) {
			fprintf(stderr, "fread failed");
			exit(1);
		}

		matrix **result;
		if ((result = malloc(sizeof(*result) * sizeof_dims[0])) == NULL) {
			fprintf(stderr, "Malloc failed");
			exit(1);
		}
		int32_t dim_matrix = sizeof_dims[1] * sizeof_dims[2];

		for (int32_t i = 0; i < sizeof_dims[0]; i++) {
			matrix *curr = grm_create_mat(sizeof_dims[1], sizeof_dims[2]);
			for (int32_t j = 0; j < dim_matrix; j++) {
				curr->data[j] = (double) data[i * dim_matrix + j];
			}
			result[i] = curr;
		}
		free(data);
		fclose(file);
		return result; // returns matrix **
	}
	fclose(file);
	return result;
}

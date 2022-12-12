#ifndef IDX_PARSE

#include <stdio.h>
#include <stdlib.h>
#include <byteswap.h>

/*
	The basic format is
	
	magic number
	size in dimension 0
	size in dimension 1
	size in dimension 2
	.....
	size in dimension N
	data
	
	The third byte codes the type of the data:
	0x08: unsigned byte
	0x09: signed byte
	0x0B: short (2 bytes)
	0x0C: int (4 bytes)
	0x0D: float (4 bytes)
	0x0E: double (8 bytes)

	The 4-th byte codes the # of dimensions of the vector/matrix: 1 for vectors, 2 for matrices

 */

typedef unsigned char u8;

void print_image(u8 *image, int width, int height);
void *parse_idxfile(char *filename);

#endif /* IDX_PARSE */

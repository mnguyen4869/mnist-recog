#pragma once

#include <assert.h>
#include <stdbool.h>
#include <time.h>

#include "../src/neural-network.h"

bool test_ReLU(void);

bool test_ReLU_activation(void);

bool test_ReLU_d(void);

bool test_ReLU_d_activation(void);

bool test_free_layer(void);

bool test_free_nn(void);

bool test_feed_forward(void);

bool test_back_prop(void);

bool test_update_weight(void);

bool test_save_load_nn(void);

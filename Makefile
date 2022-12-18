CC := gcc

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
TEST_DIR := tests

EXE := $(BIN_DIR)/main
SRC := $(wildcard $(SRC_DIR)/*.c)
OBJ := $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

TEST_EXE := $(BIN_DIR)/run-tests
TEST_SRC := $(wildcard $(TEST_DIR)/*.c)
VAL := $(wildcard $(TEST_DIR)/valgrind-out.txt*)

CPPFLAGS := -I/home/$(USER)/.local/include -MMD -MP
CFLAGS   := -Wall -pedantic -Wextra -Werror
LDFLAGS  := -Wl,-rpath=/home/$(USER)/.local/lib -L/home/$(USER)/.local/lib 
LDLIBS   := -lgramlinalg -lm

.PHONY: all clean run-tests run-valgrind

all: $(EXE)

$(EXE): $(OBJ) | $(BIN_DIR)
	$(CC) -O2 $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) -O2 $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

run-tests: $(TEST_EXE)
	./$<

run-valgrind: $(TEST_EXE)
	valgrind --leak-check=full \
    	--show-leak-kinds=all \
    	--track-origins=yes \
    	--verbose \
    	--log-file="$(TEST_DIR)/valgrind-out.txt" \
    	./$<
	cat "$(TEST_DIR)/valgrind-out.txt"

$(TEST_EXE): $(BIN_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -ggdb3 $(LDFLAGS) $(LDLIBS) src/idx_parse.c src/neural-network.c $(TEST_SRC) $(TEST) -o $@

clean:
	rm -rfv $(OBJ_DIR)
	rm -rfv $(BIN_DIR)
	rm -rfv $(VAL)
	rm -rfv "$(TEST_DIR)/test_file"

-include $(OBJ:.o=.d)

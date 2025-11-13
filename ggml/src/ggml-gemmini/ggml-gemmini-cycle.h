#pragma once

#include <stdio.h>

#if CYCLE_LOG
#define PRINT_CYCLE(...) \
    fprintf(stderr, "[layer=%s][%s] start = %lu end = %lu elapsed = %lu\n", ##__VA_ARGS__)
#else
#define PRINT_CYCLE(...) ((void)0)
#endif
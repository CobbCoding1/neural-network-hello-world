#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

float train_single[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

#define MODEL_SIZE 3
typedef float sample[MODEL_SIZE];

sample train_and[] = {
    {1, 1, 1},
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 0},
};

sample train_or[] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

sample train_nand[] = {
    {0, 0, 1},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

sample train_xor[] = {
    {1, 1, 0},
    {1, 0, 1},
    {0, 1, 1},
    {0, 0, 0},
};

sample *train = train_xor;
size_t SIZE = 4;

#define WEIGHTS 2

float rand_float() {
    return (float)rand()/(float)RAND_MAX;
}

float relu(float x) {
    return (x + fabsf(x))/2;
}

float sigmoid(float x) {
    return 1/(1 + expf(-x));
}

float forward(sample weights[WEIGHTS], size_t size, size_t i, float bias) {
    float output = 0;
    for(size_t k = 0; k < WEIGHTS; k++) {
        for(size_t j = 0; j < MODEL_SIZE; j++) {
            float input = train[i][j];
            output += input * weights[k][j];
        }
    }
    return(sigmoid(output + bias));
}

float cost(sample weights[WEIGHTS], size_t size, float bias) {
    float result = 0.0f;
    for(size_t i = 0; i < SIZE; i++) {
        float output = forward(weights, size, i, bias);
        float d = output - train[i][size];
        result += d*d;
    }
    result /= SIZE;
    return result;
}

void init_weights(sample weights[WEIGHTS]) {
    for(size_t i = 0; i < WEIGHTS; i++) {
        for(size_t j = 0; j < MODEL_SIZE; j++) {
            weights[i][j] = rand_float()*sqrt(2.0 / ((float)MODEL_SIZE + WEIGHTS));
        }
    }
}

int main(void) {
    srand(69);
    float eps = 1e-3;
    float rate = 1e-2;
    float weights[WEIGHTS][MODEL_SIZE];
    init_weights(weights);
    float bias = rand_float()*10;
    
    float result = cost(weights, WEIGHTS, bias);
    printf("initial result: %f\n", result);
    size_t epochs = 50000;
    
    for(size_t i = 0; i < epochs; i++) {
        for(size_t k = 0; k < WEIGHTS; k++) {
            for(size_t j = 0; j < MODEL_SIZE; j++) {
                float sum_gradient = 0.0f;
                for(size_t m = 0; m < SIZE; m++) {
                    float pre_cost = cost(weights, WEIGHTS, bias);
                    float original_weight = weights[k][j];
                    weights[k][j] += eps;
                    float new_cost = cost(weights, WEIGHTS, bias);
                    float gradient = (new_cost - pre_cost) / eps;
                    weights[k][j] = original_weight - gradient * rate;
                    sum_gradient += gradient;
                }
                weights[k][j] -= sum_gradient / SIZE;
            }
        }
        
        float pre_cost = cost(weights, WEIGHTS, bias);
        float new_cost = cost(weights, WEIGHTS, bias+eps);
        float gradient = (new_cost - pre_cost)/eps;
        bias -= gradient * rate;
        
        result = cost(weights, WEIGHTS, bias);
        
        //printf("%f\n", result);
    }
    
    for(size_t i = 0; i < WEIGHTS; i++) {
        for(size_t j = 0; j < MODEL_SIZE; j++) {
            printf("w%zu = %f ", i, weights[i][j]);
        }
    }
    printf("bias = %f, result = %f\n", bias, result);

    for(size_t i = 0; i < SIZE; i++) {
        printf("%f | %f = %f\n", train[i][0], train[i][1], forward(weights, WEIGHTS, i, bias));
    }
    return 0;
}

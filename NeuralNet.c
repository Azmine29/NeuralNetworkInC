#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Simple neural network tutorial to help me better understand 
double sigmoid(double x) {return 1/(1+exp(-x));}
double dsigmoid(double x) {return x * (1-x);}

double initialWeights() {return ((double)rand()) / ((double)RAND_MAX);}

void shuffer(int *array, size_t n) {
    if (n > 1) {
        size_t i; 
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define numInputs 2 
#define numHiddenNodes 2 
#define numOutputs 1 
#define numTrainingSets 4 

int main() {
    const double lr = 0.1f; 

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    double trainingInputs[numTrainingSets][numInputs] = {{0.0f, 0.0f},
                                                         {1.0f, 0.0f},           
                                                         {0.0f, 1.0f},           
                                                         {1.0f, 1.0f}};

    double trainingOutputs[numTrainingSets][numOutputs] = {{0.0f},
                                                           {1.0f},           
                                                           {1.0f},           
                                                           {0.0f}};

    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            hiddenWeights[i][j] = initialWeights();
        }
    }

    for (int i = 0; i < numHiddenNodes; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i][j] = initialWeights();
        }
    }

    for (int i = 0; i < numHiddenNodes; i++) {
        hiddenLayerBias[i] = initialWeights();
    }

    for (int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = initialWeights();
    }

    int trainingSetOrder[] = {0, 1, 2, 3};

    int numEpochs = 10000;

    // Train Neural Net for numbers of epochs 
    for (int epochs = 0; epochs < numEpochs; epochs++) {
        shuffer(trainingSetOrder, numTrainingSets);
        for (int x = 0; x < numTrainingSets; x++) {
            int i = trainingSetOrder[x];
            
            // Forward Pass 
            // Compute Hidden Layer Activation
            for (int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++) {
                    activation += trainingInputs[i][k] * hiddenWeights[k][j];     
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            // Compute Output Layer Activation
            for (int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation); 
            }

            // Print inputs, predicted output, and expected output
            printf("Epoch %d, Training Set %d:\n", epochs + 1, i + 1);
            printf("Inputs: %.1f, %.1f\n", trainingInputs[i][0], trainingInputs[i][1]);
            printf("Predicted Output: %.4f\n", outputLayer[0]);
            printf("Expected Output: %.1f\n", trainingOutputs[i][0]);
            printf("------------------------------\n");

            // Backprop
            // Compute change in output weights 
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++) {
                double error = (trainingOutputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dsigmoid(outputLayer[j]);   
            }    

            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++) {
                double error = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dsigmoid(hiddenLayer[j]); 
            }

            // Apply changes to output weights
            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k = 0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            // Apply changes to hidden weights
            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; k++) {
                    hiddenWeights[k][j] += trainingInputs[i][k] * deltaHidden[j] * lr; 
                }
            }
        }

        // Optional: Print a summary after each epoch
        if ((epochs + 1) % 1000 == 0) {
            printf("\n=== Epoch %d completed ===\n\n", epochs + 1);
        }
    }

    // After training, test the network
    printf("\nTraining completed. Testing the network:\n");
    for (int i = 0; i < numTrainingSets; i++) {
        // Forward pass for testing
        for (int j = 0; j < numHiddenNodes; j++) {
            double activation = hiddenLayerBias[j];
            for (int k = 0; k < numInputs; k++) {
                activation += trainingInputs[i][k] * hiddenWeights[k][j];
            }
            hiddenLayer[j] = sigmoid(activation);
        }
        for (int j = 0; j < numOutputs; j++) {
            double activation = outputLayerBias[j];
            for (int k = 0; k < numHiddenNodes; k++) {
                activation += hiddenLayer[k] * outputWeights[k][j];
            }
            outputLayer[j] = sigmoid(activation);
        }

        // Print test results
        printf("Test %d:\n", i + 1);
        printf("Inputs: %.1f, %.1f\n", trainingInputs[i][0], trainingInputs[i][1]);
        printf("Predicted Output: %.4f\n", outputLayer[0]);
        printf("Expected Output: %.1f\n", trainingOutputs[i][0]);
        printf("------------------------------\n");
    }

    return 0;
}
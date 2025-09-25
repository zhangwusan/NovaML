#include "mldl/modules/sequential.hpp"
#include "mldl/layers/dense.hpp"
#include "mldl/layers/relu.hpp"
#include "mldl/losses/mse.hpp"
#include <iostream>

using namespace ml;

int main() {
    // Input tensor (3 features)
    Tensor<double> input(3); 
    input[0] = 1.0;
    input[1] = 2.0;
    input[2] = 3.0;

    // Target tensor (for training)
    Tensor<double> target(1);
    target[0] = 10.0;  // Example target

    // Build Model
    Sequential<double> model;
    model.add(std::make_shared<Dense<double>>(3, 4));
    model.add(std::make_shared<ReLU<double>>());
    model.add(std::make_shared<Dense<double>>(4, 1));
    model.add(std::make_shared<ReLU<double>>());

    std::cout << model << std::endl;

    // Define Loss
    MSELoss<double> loss_fn;

    double learning_rate = 0.01;

    // Training loop (example with 100 iterations)
    for (int epoch = 0; epoch < 100; ++epoch) {
        // Forward pass
        Tensor<double> output = model.forward(input);

        // Compute loss
        double loss_value = loss_fn.forward(output, target);

        // Backward pass
        Tensor<double> grad = loss_fn.backward(output, target);
        model.backward(grad);

        // Update weights
        model.update(learning_rate);

        if (epoch % 10 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << loss_value << "\n";
    }

    // Final prediction
    Tensor<double> final_output = model.forward(input);

    std::cout << final_output << std::endl;
    std::cout << "Finished" << std::endl;

    return 0;
}
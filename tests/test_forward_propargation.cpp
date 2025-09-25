#include "mldl/model.hpp"
#include "mldl/layer.hpp"
#include "mldl/activation.hpp"
#include "mldl/factory.hpp"

using namespace ml;

int main() {
    Tensor<double> input(3); 

    input[0] = 1.0f;
    input[1] = 2.0f;
    input[2] = 3.0f;

    // Build Model

    Model<double> model;
    model.add(std::make_shared<Dense<double>>(3, 4)); // 3 in features and 4 out features
    model.add(std::make_shared<ReLU<double>>());
    model.add(std::make_shared<Dense<double>>(4, 1));
    model.add(std::make_shared<ReLU<double>>());

    std::cout << model;

    // Forward pass

    Tensor<double> output = model.forward(input);

    std::cout << "Final output : " << output;
    

    return 0;
}
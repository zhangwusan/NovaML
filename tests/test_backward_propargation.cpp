#include "NovaML/Core/Core.hpp"
#include <iostream>

using namespace NovaML::Core;

int main() {
    TensorModule::Tensor<double> input(3);
    input[0] = 1.0;
    input[1] = 2.0;
    input[2] = 3.0;

    TensorModule::Tensor<double> target(1);
    target[0] = 10.0;

    auto model = std::make_shared<Module::Sequential<double>>();
    model->add(std::make_shared<LayerModule::Dense<double>>(3, 4));
    model->add(std::make_shared<ActivationModule::ReLU<double>>());
    model->add(std::make_shared<LayerModule::Dense<double>>(4, 1));
    model->add(std::make_shared<ActivationModule::ReLU<double>>());

    std::cout << *model << std::endl;

    LossModule::MSELoss<double> loss_fn;
    double learning_rate = 0.01;

    for (int epoch = 0; epoch < 100; ++epoch) {
        TensorModule::Tensor<double> output = model->forward(input);
        double loss_value = loss_fn.forward(output, target);
        TensorModule::Tensor<double> grad = loss_fn.backward();
        model->backward(grad);
        model->update(learning_rate);

        if (epoch % 10 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << loss_value << "\n";
    }

    TensorModule::Tensor final_output = model->forward(input);
    std::cout << final_output << std::endl;
    std::cout << "Finished" << std::endl;
}
#include "NovaML/core/tensor/tensor.hpp"
#include "NovaML/core/tensor/utils.hpp"
#include "NovaML/core/module/sequential.hpp"
#include "NovaML/core/layer/dense.hpp"
#include <iostream>
#include <memory>
#include <sstream>

using namespace NovaML::Core;

int main()
{
    Sequential<double> model;
    model.add_module("dense1", std::make_shared<Dense<double>>(3, 2));
    model.add_module("dense2", std::make_shared<Dense<double>>(2, 1));

    auto x = std::make_shared<Tensor<double>>(std::vector<double>{1.0, 2.0, 3.0}, true);

    auto y = model.forward(x);
    std::cout << "Forward output: " << *y << "\n";

    // Backward: simulate scalar gradient of 1
    y->backward();

    std::cout << "\nGradients after backward:\n";

    for (size_t i = 0; i < model.get_modules().size(); i++)
    {
        auto dense = std::dynamic_pointer_cast<Dense<double>>(model.get_modules()[i]);
        if (dense)
        {
            std::cout << "Layer " << i << " weights grad: " << vector_to_string(dense->get_weights()->get_grad()) << "\n";
            if (dense->get_bias())
                std::cout << "Layer " << i << " bias grad: " << vector_to_string(dense->get_bias()->get_grad()) << "\n";
        }
    }
}
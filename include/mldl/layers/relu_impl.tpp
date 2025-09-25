#pragma once
#include "relu.hpp"

namespace ml {

template <typename T>
ReLU<T>::ReLU() : last_input(0) {}

// Forward: store input and apply ReLU
template <typename T>
Tensor<T> ReLU<T>::forward(const Tensor<T> &input) {
    last_input = input;
    Tensor<T> output(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        output[i] = input[i] > T(0) ? input[i] : T(0);
    return output;
}

// Backward: gradient passes only where input > 0
template <typename T>
Tensor<T> ReLU<T>::backward(const Tensor<T> &grad_output) {
    Tensor<T> grad(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); ++i)
        grad[i] = last_input[i] > T(0) ? grad_output[i] : T(0);
    return grad;
}

} // namespace ml
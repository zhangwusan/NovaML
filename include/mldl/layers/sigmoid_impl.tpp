#pragma once
#include "sigmoid.hpp"

namespace ml
{

    template <typename T> Sigmoid<T>::Sigmoid() : last_output(0) {}
    // Forward: apply sigmoid function
    template <typename T>
    Tensor<T> Sigmoid<T>::forward(const Tensor<T> &input)
    {
        last_output = Tensor<T>(input.size());
        for (size_t i = 0; i < input.size(); ++i)
            last_output[i] = T(1) / (T(1) + std::exp(-input[i]));
        return last_output;
    }

    // Backward: gradient = grad_output * sigmoid(x) * (1 - sigmoid(x))
    template <typename T>
    Tensor<T> Sigmoid<T>::backward(const Tensor<T> &grad_output)
    {
        Tensor<T> grad(grad_output.size());
        for (size_t i = 0; i < grad_output.size(); ++i)
            grad[i] = grad_output[i] * last_output[i] * (T(1) - last_output[i]);
        return grad;
    }

}
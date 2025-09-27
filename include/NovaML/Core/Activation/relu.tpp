#pragma once
#include "relu.hpp"

namespace NovaML::Core::ActivationModule
{

    template <typename T>
    NovaML::Core::TensorModule::Tensor<T> ReLU<T>::forward(
        const NovaML::Core::TensorModule::Tensor<T> &input)
    {
        last_input = input;
        NovaML::Core::TensorModule::Tensor<T> output(input.size());

        for (size_t i = 0; i < input.size(); ++i)
            output[i] = input[i] > T(0) ? input[i] : T(0);

        return output;
    }

    template <typename T>
    NovaML::Core::TensorModule::Tensor<T> ReLU<T>::backward(
        const NovaML::Core::TensorModule::Tensor<T> &grad_output)
    {
        NovaML::Core::TensorModule::Tensor<T> grad(grad_output.size());

        for (size_t i = 0; i < grad_output.size(); ++i)
            grad[i] = last_input[i] > T(0) ? grad_output[i] : T(0);

        return grad;
    }

}
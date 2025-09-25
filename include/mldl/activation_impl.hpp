#pragma once
#include "activation.hpp"

namespace ml
{
    // ---------------- ReLU ----------------
    template <typename T>
    ReLU<T>::ReLU() : last_input(0) {}

    template <typename T>
    Tensor<T> ReLU<T>::forward(const Tensor<T> &input)
    {
        Tensor<T> output(input.size());
        for (size_t i = 0; i < input.size(); ++i)
            output[i] = std::max(T(0), input[i]);
        last_input = input; // store for backward
        return output;
    }

    template <typename T>
    Tensor<T> ReLU<T>::backward(const Tensor<T> &grad_output)
    {
        Tensor<T> grad_input(grad_output.size());
        for (size_t i = 0; i < grad_output.size(); ++i)
            grad_input[i] = (last_input[i] > T(0)) ? grad_output[i] : T(0);
        return grad_input;
    }

    template <typename T>
    std::string ReLU<T>::get_info() const
    {
        return "ReLU()";
    }

    template <typename T>
    size_t ReLU<T>::num_params() const
    {
        return 0;
    }

    // ---------------- Sigmoid ----------------
    template <typename T>
    Sigmoid<T>::Sigmoid() : last_output(0) {}
    
    template <typename T>
    Tensor<T> Sigmoid<T>::forward(const Tensor<T> &input)
    {
        Tensor<T> output(input.size());
        last_output = Tensor<T>(input.size());
        for (size_t i = 0; i < input.size(); ++i)
        {
            output[i] = T(1) / (T(1) + std::exp(-input[i]));
            last_output[i] = output[i]; // store for backward
        }
        return output;
    }

    template <typename T>
    Tensor<T> Sigmoid<T>::backward(const Tensor<T> &grad_output)
    {
        Tensor<T> grad_input(grad_output.size());
        for (size_t i = 0; i < grad_output.size(); ++i)
            grad_input[i] = grad_output[i] * last_output[i] * (T(1) - last_output[i]);
        return grad_input;
    }

    template <typename T>
    std::string Sigmoid<T>::get_info() const
    {
        return "Sigmoid()";
    }

    template <typename T>
    size_t Sigmoid<T>::num_params() const
    {
        return 0;
    }
}
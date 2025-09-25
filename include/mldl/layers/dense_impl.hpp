#pragma once
#include "dense.hpp"

namespace ml {

template <typename T>
Dense<T>::Dense(size_t in_features, size_t out_features)
    : weights(out_features, std::vector<T>(in_features)),
      bias(out_features, T(0)),
      grad_weights(out_features, std::vector<T>(in_features, T(0))),
      grad_bias(out_features, T(0)),
      last_input(in_features)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(T(-0.1), T(0.1));

    for (auto &row : weights)
        for (auto &w : row)
            w = dist(gen);
}

template <typename T>
Tensor<T> Dense<T>::forward(const Tensor<T> &input)
{
    last_input = input;
    Tensor<T> output(weights.size());

    for (size_t i = 0; i < weights.size(); ++i)
    {
        T sum = bias[i];
        for (size_t j = 0; j < input.size(); ++j)
            sum += weights[i][j] * input[j];
        output[i] = sum;
    }

    return output;
}

template <typename T>
Tensor<T> Dense<T>::backward(const Tensor<T> &grad_output)
{
    Tensor<T> grad_input(last_input.size());

    for (size_t i = 0; i < weights.size(); ++i)
    {
        grad_bias[i] = grad_output[i];
        for (size_t j = 0; j < last_input.size(); ++j)
        {
            grad_weights[i][j] = grad_output[i] * last_input[j];
            grad_input[j] += weights[i][j] * grad_output[i];
        }
    }

    return grad_input;
}

template <typename T>
void Dense<T>::update(T lr)
{
    for (size_t i = 0; i < weights.size(); ++i)
    {
        for (size_t j = 0; j < weights[0].size(); ++j)
            weights[i][j] -= lr * grad_weights[i][j];
        bias[i] -= lr * grad_bias[i];
    }
}

template <typename T>
std::string Dense<T>::get_info() const
{
    return "Dense(" + std::to_string(weights[0].size()) + "->" + std::to_string(weights.size()) + ")";
}

template <typename T>
size_t Dense<T>::num_params() const
{
    return weights.size() * weights[0].size() + bias.size();
}

} // namespace ml
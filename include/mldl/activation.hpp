#pragma once
#include "tensor.hpp"
#include "layer.hpp"
#include <string>
#include <algorithm>
#include <cmath>

namespace ml
{
    template <typename T = float>
    class ReLU : public Layer<T>
    {
    public:
        ReLU();
        Tensor<T> forward(const Tensor<T> &input) override;
        Tensor<T> backward(const Tensor<T> &grad_output) override;
        std::string get_info() const override;
        size_t num_params() const override;

    private:
        Tensor<T> last_input; // store input for backward
    };

    template <typename T = float>
    class Sigmoid : public Layer<T>
    {
    public:
        Sigmoid();
        Tensor<T> forward(const Tensor<T> &input) override;
        Tensor<T> backward(const Tensor<T> &grad_output) override;
        std::string get_info() const override;
        size_t num_params() const override;

    private:
        Tensor<T> last_output;
    };
}

#include "activation_impl.hpp"
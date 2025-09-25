#pragma once
#include "../core/module.hpp"
#include <string>
#include <cmath>

namespace ml {

template <typename T = float>
class Sigmoid : public Module<T> {
public:
    Sigmoid();

    // Forward pass
    Tensor<T> forward(const Tensor<T> &input) override;

    // Backward pass
    Tensor<T> backward(const Tensor<T> &grad_output) override;

    // Info
    std::string get_info() const override {
        return "Sigmoid";
    }

private:
    Tensor<T> last_output; // store output for backprop
};

} // namespace ml

#include "sigmoid_impl.hpp"
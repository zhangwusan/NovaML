#pragma once
#include "../core/module.hpp"
#include <string>

namespace ml
{

    template <typename T = float>
    class ReLU : public Module<T>
    {
    public:
        ReLU();

        // Forward pass
        Tensor<T> forward(const Tensor<T> &input) override;

        // Backward pass
        Tensor<T> backward(const Tensor<T> &grad_output) override;

        // Info
        std::string get_info() const override
        {
            return "ReLU";
        }

    private:
        Tensor<T> last_input;
    };

} // namespace ml

#include "relu_impl.tpp"
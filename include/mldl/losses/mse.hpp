#pragma once
#include "loss.hpp"

namespace ml
{

    template <typename T = float>
    class MSELoss : public Loss<T>
    {
    public:
        MSELoss() = default;
        ~MSELoss() override = default;

        // Compute loss value
        T forward(const Tensor<T> &prediction, const Tensor<T> &target) override;

        // Compute gradient w.r.t prediction
        Tensor<T> backward(const Tensor<T> &prediction, const Tensor<T> &target) override;
    };

} // namespace ml

// Include implementation for templates
#include "mse_loss.tpp"
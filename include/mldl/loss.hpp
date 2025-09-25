#pragma once
#include "tensor.hpp"

namespace ml
{

    template <typename T = float>
    class Loss
    {
    public:
        virtual ~Loss() = default;
        virtual T forward(const Tensor<T> &prediction, const Tensor<T> &target) = 0;
        virtual Tensor<T> backward(const Tensor<T> &prediction, const Tensor<T> &target) = 0;
    };

    // MSE Loss
    template <typename T = float>
    class MSELoss : public Loss<T>
    {
    public:
        T forward(const Tensor<T> &prediction, const Tensor<T> &target) override;
        Tensor<T> backward(const Tensor<T> &prediction, const Tensor<T> &target) override;
    };
} // namespace ml

#include "loss_impl.hpp"
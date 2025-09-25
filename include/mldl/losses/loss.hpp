#pragma once
#include "../core/tensor.hpp"

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
} // namespace ml
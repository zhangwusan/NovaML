#pragma once
#include "loss.hpp"
#include <cmath>

namespace ml
{
    // MSE Forward: mean squared error
    template <typename T>
    T MSELoss<T>::forward(const Tensor<T> &prediction, const Tensor<T> &target)
    {
        T loss = T(0);
        size_t n = prediction.size();
        for (size_t i = 0; i < n; ++i)
        {
            T diff = prediction[i] - target[i];
            loss += diff * diff;
        }
        return loss / static_cast<T>(n);
    }

    // MSE Backward: gradient w.r.t prediction
    template <typename T>
    Tensor<T> MSELoss<T>::backward(const Tensor<T> &prediction, const Tensor<T> &target)
    {
        size_t n = prediction.size();
        Tensor<T> grad(prediction.size());
        for (size_t i = 0; i < n; ++i)
        {
            grad[i] = T(2) * (prediction[i] - target[i]) / static_cast<T>(n);
        }
        return grad;
    }

} // namespace ml
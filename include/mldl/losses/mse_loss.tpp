#pragma once
#include "mse.hpp"

namespace ml
{

    template <typename T>
    T MSELoss<T>::forward(const Tensor<T> &prediction, const Tensor<T> &target)
    {
        T loss = 0;
        for (size_t i = 0; i < prediction.size(); ++i)
        {
            T diff = prediction[i] - target[i];
            loss += diff * diff;
        }
        return loss / prediction.size();
    }

    template <typename T>
    Tensor<T> MSELoss<T>::backward(const Tensor<T> &prediction, const Tensor<T> &target)
    {
        Tensor<T> grad(prediction.size());
        T n = static_cast<T>(prediction.size());
        for (size_t i = 0; i < prediction.size(); ++i)
            grad[i] = 2 * (prediction[i] - target[i]) / n;
        return grad;
    }

}
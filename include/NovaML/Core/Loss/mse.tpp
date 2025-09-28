#pragma once
#include "mse.hpp"

namespace NovaML::Core::LossModule
{
    template <typename T>
    MSELoss<T>::MSELoss()
        : last_pred(0), last_target(0)
    {}

    template <typename T>
    T MSELoss<T>::forward(
        const NovaML::Core::TensorModule::Tensor<T> &pred,
        const NovaML::Core::TensorModule::Tensor<T> &target)
    {
        last_pred = pred;
        last_target = target;

        T loss = 0;
        for (size_t i = 0; i < pred.size(); ++i)
            loss += (pred[i] - target[i]) * (pred[i] - target[i]);
        return loss / static_cast<T>(pred.size());
    }

    template <typename T>
    NovaML::Core::TensorModule::Tensor<T> MSELoss<T>::backward()
    {
        NovaML::Core::TensorModule::Tensor<T> grad(last_pred.size());
        T inv_size = T(1) / static_cast<T>(last_pred.size());

        for (size_t i = 0; i < last_pred.size(); ++i)
            grad[i] = 2 * (last_pred[i] - last_target[i]) * inv_size;

        return grad;
    }

}
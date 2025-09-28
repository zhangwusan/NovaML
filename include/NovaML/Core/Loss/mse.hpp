#pragma once
#include "../Tensor/tensor.hpp"
#include <cstddef>
#include <cmath>

namespace NovaML::Core::LossModule
{

    template <typename T = float>
    class MSELoss
    {
    public:
        MSELoss();

        // Forward: compute scalar loss
        T forward(
            const NovaML::Core::TensorModule::Tensor<T> &pred,
            const NovaML::Core::TensorModule::Tensor<T> &target);

        // Backward: compute gradient w.r.t. prediction
        NovaML::Core::TensorModule::Tensor<T> backward();

    private:
        NovaML::Core::TensorModule::Tensor<T> last_pred;
        NovaML::Core::TensorModule::Tensor<T> last_target;
    };

} // namespace NovaML::Core::LossModule

#include "mse.tpp"
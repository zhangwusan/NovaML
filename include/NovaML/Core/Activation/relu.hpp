#pragma once
#include "../Module/module.hpp"

namespace NovaML::Core::ActivationModule
{

    template <typename T = float>
    class ReLU : public NovaML::Core::Module::BaseModule<T>
    {
    public:
        ReLU() : last_input(0) {}

        // Forward pass
        NovaML::Core::TensorModule::Tensor<T> forward(const NovaML::Core::TensorModule::Tensor<T> &input) override;
        // Backward pass
        NovaML::Core::TensorModule::Tensor<T> backward(const NovaML::Core::TensorModule::Tensor<T> &grad_output) override;
        // Info
        std::string info(std::ostream &os) const override { return "ReLU"; }

    private:
        NovaML::Core::TensorModule::Tensor<T> last_input;
    };

}

#include "relu.tpp"
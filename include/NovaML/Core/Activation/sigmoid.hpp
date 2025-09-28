#pragma once
#include "../Module/module.hpp"
#include <string>
#include <cmath>

namespace NovaML::Core::ActivationModule {

template <typename T = float>
class Sigmoid : public NovaML::Core::Module::BaseModule<T> {
public:
    Sigmoid();

    // Forward pass
    NovaML::Core::TensorModule::Tensor<T> forward(
        const NovaML::Core::TensorModule::Tensor<T> &input) override;

    // Backward pass
    NovaML::Core::TensorModule::Tensor<T> backward(
        const NovaML::Core::TensorModule::Tensor<T> &grad_output) override;

    // Info
    std::string info(std::ostream &os) const override { return "Sigmoid"; }

private:
    NovaML::Core::TensorModule::Tensor<T> last_output;
};

}

#include "sigmoid.tpp"
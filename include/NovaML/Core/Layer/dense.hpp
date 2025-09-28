#pragma once
#include "../Module/module.hpp"
#include "../Tensor/tensor.hpp"
#include <vector>
#include <random>
#include <string>


namespace NovaML::Core::LayerModule
{

    template <typename T = float>
    class Dense : public NovaML::Core::Module::BaseModule<T>
    {
    public:
        Dense(size_t in_features, size_t out_features);

        NovaML::Core::TensorModule::Tensor<T> forward(const NovaML::Core::TensorModule::Tensor<T> &input) override;
        NovaML::Core::TensorModule::Tensor<T> backward(const NovaML::Core::TensorModule::Tensor<T> &grad_output) override;
        void update(T lr) override;
        std::string info(std::ostream &os) const override;
        size_t num_params() const override;

    private:
        std::vector<std::vector<T>> weights;
        std::vector<T> bias;
        std::vector<std::vector<T>> grad_weights;
        std::vector<T> grad_bias;
        NovaML::Core::TensorModule::Tensor<T> last_input;
    };

}

#include "dense.tpp"
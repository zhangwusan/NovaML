#pragma once
#include "../Module/module.hpp"
#include <vector>
#include <memory>
#include <sstream>

namespace NovaML::Core::Module {

template <typename T = float>
class Sequential : public BaseModule<T> {
public:
    Sequential() = default;

    void add(std::shared_ptr<BaseModule<T>> module);

    NovaML::Core::TensorModule::Tensor<T> forward(const NovaML::Core::TensorModule::Tensor<T> &input) override;
    NovaML::Core::TensorModule::Tensor<T> backward(const NovaML::Core::TensorModule::Tensor<T> &grad_output) override;
    void update(T lr) override;
    std::string info(std::ostream &os) const override;
    size_t num_params() const override;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Sequential<T>& seq);

} // namespace NovaML::Core::Module

#include "sequential.tpp"
#include "module.hpp"

namespace NovaML::Core::Module
{
    template <typename T>
    TensorModule::Tensor<T> BaseModule<T>::backward(const TensorModule::Tensor<T> &grad_output)
    {
        auto grad = grad_output;
        for (auto it = this->submodules.rbegin(); it != this->submodules.rend(); ++it)
            grad = (*it)->backward(grad);
        return grad;
    }

    template <typename T>
    void BaseModule<T>::update(T lr)
    {
        for (auto &m : this->submodules)
            m->update(lr);
    }

    template <typename T>
    size_t BaseModule<T>::num_params() const
    {
        size_t total = 0;
        for (auto &m : this->submodules)
            total += m->num_params();
        return total;
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const BaseModule<T> &module)
    {
        os << module.info();
        return os;
    }
}
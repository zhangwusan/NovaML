#include "sequential.hpp"

namespace NovaML::Core::Module
{

    template <typename T>
    void Sequential<T>::add(std::shared_ptr<BaseModule<T>> module)
    {
        this->submodules.push_back(module);
    }

    template <typename T>
    TensorModule::Tensor<T> Sequential<T>::forward(const TensorModule::Tensor<T> &input)
    {
        TensorModule::Tensor<T> x = input;
        for (auto &m : this->submodules)
            x = m->forward(x);
        return x;
    }

    template <typename T>
    TensorModule::Tensor<T> Sequential<T>::backward(const TensorModule::Tensor<T> &grad_output)
    {
        TensorModule::Tensor<T> grad = grad_output;
        for (auto it = this->submodules.rbegin(); it != this->submodules.rend(); ++it)
            grad = (*it)->backward(grad);
        return grad;
    }

    template <typename T>
    void Sequential<T>::update(T lr)
    {
        for (auto &m : this->submodules)
            m->update(lr);
    }

    template <typename T>
    std::string Sequential<T>::info(std::ostream &os) const
    {
        os << "Sequential with " << this->submodules.size() << " modules\n";
        for (size_t i = 0; i < this->submodules.size(); ++i)
            os << " [" << i << "] " << this->submodules[i]->info(os) << "\n";
        return ""; // or some string summary if you want
    }

    template <typename T>
    size_t Sequential<T>::num_params() const
    {
        size_t total = 0;
        for (auto &m : this->submodules)
            total += m->num_params();
        return total;
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const Sequential<T> &seq)
    {
        os << seq.info(os);
        return os;
    }

}
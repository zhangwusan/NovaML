#pragma once
#include "sequential.hpp"

namespace ml
{
    template <typename T>
    void Sequential<T>::add(std::shared_ptr<Module<T>> layer)
    {
        this->submodules.push_back(layer);
    }

    template <typename T>
    Tensor<T> Sequential<T>::forward(const Tensor<T> &input)
    {
        Tensor<T> x = input;
        for (auto &layer : this->submodules)
            x = layer->forward(x);
        return x;
    }

    template <typename T>
    std::string Sequential<T>::get_info() const
    {
        std::ostringstream oss;
        oss << "Sequential with " << this->submodules.size() << " layers\n";
        for (size_t i = 0; i < this->submodules.size(); ++i)
        {
            oss << " Layer " << i + 1 << ": " << this->submodules[i]->get_info()
                << " | Params: " << this->submodules[i]->num_params() << "\n";
        }
        oss << "Total params: " << this->num_params();
        return oss.str();
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const Sequential<T> &seq)
    {
        os << seq.get_info();
        return os;
    }
}
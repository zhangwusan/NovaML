#include "tensor.hpp"
namespace NovaML::Core::TensorModule
{

    template <typename T>
    Tensor<T>::Tensor(size_t size) : data(size, T(0)) {}

    template <typename T>
    T &Tensor<T>::operator[](size_t i) { return data[i]; }

    template <typename T>
    const T &Tensor<T>::operator[](size_t i) const { return data[i]; }

    template <typename T>
    size_t Tensor<T>::size() const { return data.size(); }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor)
    {
        os << "[";
        for (size_t i = 0; i < tensor.size(); ++i)
        {
            os << tensor[i];
            if (i != tensor.size() - 1)
                os << ", ";
        }
        os << "]";
        return os;
    }

}
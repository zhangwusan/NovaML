#include "model.hpp"

namespace ml
{
    template <typename T>
    void Model<T>::add(std::shared_ptr<Layer<T>> layer)
    {
        layers.push_back(layer);
    }

    template <typename T>
    Tensor<T> Model<T>::forward(const Tensor<T> &input)
    {
        Tensor<T> x = input;
        for (auto &layer : layers)
        {
            x = layer->forward(x);
        }
        return x;
    }

    template <typename T>
    Tensor<T> Model<T>::backward(const Tensor<T> &grad_output)
    {
        Tensor<T> grad = grad_output;
        // iterate layers in reverse order
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            grad = (*it)->backward(grad);
        }
        return grad; // gradient w.r.t input
    }

    template <typename T>
    void Model<T>::update(T lr)
    {
        for (auto &layer : layers)
        {
            auto dense = std::dynamic_pointer_cast<Dense<T>>(layer);
            if (dense)
                dense->update(lr);
        }
    }

    template <typename T>
    const std::vector<std::shared_ptr<Layer<T>>> &Model<T>::get_layers() const
    {
        return layers;
    }

    template <typename T>
    size_t Model<T>::total_params() const
    {
        size_t total = 0;
        for (auto &layer : layers)
            total += layer->num_params();
        return total;
    }

    template <typename T>
    size_t Model<T>::total_memory_bytes() const
    {
        return total_params() * sizeof(T); // assuming float
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const Model<T> &model)
    {
        os << "Model Summary:\n";
        const auto &layers = model.get_layers();
        for (size_t i = 0; i < layers.size(); ++i)
        {
            os << " Layer " << i + 1 << ": "
               << layers[i]->get_info()
               << " | Params: " << layers[i]->num_params() << "\n";
        }
        os << "Total parameters: " << model.total_params() << "\n";
        os << "Approx. memory usage: " << model.total_memory_bytes() << " bytes\n";
        return os;
    }
}
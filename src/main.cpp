#include "NovaML/Core/tensor.hpp"
#include "NovaML/Core/tensor_ops.hpp"
#include <iostream>

using namespace NovaML::Core;

int main()
{
    // Create two tensors with gradient tracking
    auto a = std::make_shared<Tensor<int>>(std::vector<int>{1, 2, 3}, true);
    auto b = std::make_shared<Tensor<int>>(std::vector<int>{4, 5, 6}, true);

    // Perform addition using operator+
    auto c = a + b;

    std::cout << "Tensor a: " << *a << std::endl;
    std::cout << "Tensor b: " << *b << std::endl;
    std::cout << "Tensor c = a + b: " << *c << std::endl;

    // Backprop
    c->backward();

    std::cout << "After backward:" << std::endl;
    std::cout << "Tensor a grad: " << vector_to_string(a->get_grad()) << std::endl;
    std::cout << "Tensor b grad: " << vector_to_string(b->get_grad()) << std::endl;

    return 0;
}
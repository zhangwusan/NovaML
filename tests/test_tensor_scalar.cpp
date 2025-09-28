#include <NovaML/core/tensor/tensor.hpp>
#include <NovaML/core/tensor/tensor_ops.hpp>
#include <iostream>

using namespace NovaML::Core;

int main() {
    auto a = std::make_shared<Tensor<double>>(std::vector<double>{1.0, 2.0, 3.0}, true);
    auto b = std::make_shared<Tensor<double>>(std::vector<double>{4.0, 5.0, 6.0}, true);

    double scalar = 10.0;

    // Tensor + Scalar
    auto r1 = a + scalar;
    std::cout << "a + scalar: " << *r1 << std::endl;
    r1->backward();
    std::cout << "a grad: " << vector_to_string(a->get_grad()) << "\n";
    a->zero_grad();

    // Scalar + Tensor
    auto r2 = scalar + b;
    std::cout << "scalar + b: " << *r2 << std::endl;
    r2->backward();
    std::cout << "b grad: " << vector_to_string(b->get_grad()) << "\n";
    b->zero_grad();

    // Tensor - Scalar
    auto r3 = a - scalar;
    std::cout << "a - scalar: " << *r3 << std::endl;
    r3->backward();
    std::cout << "a grad: " << vector_to_string(a->get_grad()) << "\n";
    a->zero_grad();

    // Scalar - Tensor
    auto r4 = scalar - b;
    std::cout << "scalar - b: " << *r4 << std::endl;
    r4->backward();
    std::cout << "b grad: " << vector_to_string(b->get_grad()) << "\n";
    b->zero_grad();

    // Tensor * Scalar
    auto r5 = a * scalar;
    std::cout << "a * scalar: " << *r5 << std::endl;
    r5->backward();
    std::cout << "a grad: " << vector_to_string(a->get_grad()) << "\n";
    a->zero_grad();

    // Scalar * Tensor
    auto r6 = scalar * b;
    std::cout << "scalar * b: " << *r6 << std::endl;
    r6->backward();
    std::cout << "b grad: " << vector_to_string(b->get_grad()) << "\n";
    b->zero_grad();

    return 0;
}
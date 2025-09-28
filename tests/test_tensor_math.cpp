#include "NovaML/core/tensor/tensor.hpp"
#include "NovaML/core/tensor/tensor_math.hpp"
#include <iostream>

using namespace NovaML::Core;

int main()
{
    auto a = std::make_shared<Tensor<double>>(std::vector<double>{1.0, 2.0, 3.0}, true);
    auto b = std::make_shared<Tensor<double>>(std::vector<double>{4.0, 5.0, 6.0}, true);

    auto s = sum(a);  // sum all elements
    auto m = mean(a); // mean
    auto e = exp(a);  // element-wise exp
    auto l = log(a);  // element-wise log

    std::cout << "sum(a): " << *s << std::endl;
    s->backward();
    std::cout << "a grad after sum: " << vector_to_string(a->get_grad()) << std::endl;
    a->zero_grad();

    std::cout << "mean(a): " << *m << std::endl;
    m->backward();
    std::cout << "a grad after mean: " << vector_to_string(a->get_grad()) << std::endl;
    a->zero_grad();

    std::cout << "exp(a): " << *e << std::endl;
    e->backward();
    std::cout << "a grad after exp: " << vector_to_string(a->get_grad()) << std::endl;
    a->zero_grad();

    std::cout << "log(a): " << *l << std::endl;
    l->backward();
    std::cout << "a grad after log: " << vector_to_string(a->get_grad()) << std::endl;
    a->zero_grad();

    return 0;
}
#pragma once
#include "../module/module.hpp"
#include <memory>
#include <vector>
#include <string>
#include <sstream>

namespace NovaML::Core
{
    template <typename T = float>
    class Sequential : public BaseModule<T>
    {
    public:
        Sequential() = default;

        void add_module(const std::string &name, std::shared_ptr<BaseModule<T>> module)
        {
            this->add_submodule(name, module);
            module_names.push_back(name);
            modules.push_back(module);
        }

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>> &input) override
        {
            auto x = input;
            for (auto &m : modules)
                x = m->forward(x);
            return x;
        }

        std::vector<std::shared_ptr<Tensor<T>>> parameters() const override
        {
            std::vector<std::shared_ptr<Tensor<T>>> params;
            for (auto &m : modules)
            {
                auto p = m->parameters();
                params.insert(params.end(), p.begin(), p.end());
            }
            return params;
        }

        std::string info(int level = 0) const override
        {
            std::ostringstream oss;
            std::string indent(level * 2, ' ');
            oss << indent << "Sequential (" << modules.size() << " modules)\n";
            for (size_t i = 0; i < modules.size(); i++)
                oss << indent << " [" << module_names[i] << "] " << modules[i]->info(level + 1) << "\n";
            return oss.str();
        }

        const std::vector<std::shared_ptr<BaseModule<T>>> &get_modules() const { return modules; }

    private:
        std::vector<std::shared_ptr<BaseModule<T>>> modules;
        std::vector<std::string> module_names;
    };
}
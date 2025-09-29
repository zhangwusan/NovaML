#pragma once
#include "../tensor/tensor.hpp"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <sstream>

namespace NovaML::Core
{
    template <typename T = float>
    class BaseModule : public std::enable_shared_from_this<BaseModule<T>>
    {
    public:
        virtual ~BaseModule() = default;

        virtual std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>> &input) = 0;

        std::shared_ptr<Tensor<T>> operator()(const std::shared_ptr<Tensor<T>> &input)
        {
            return forward(input);
        }

        void add_submodule(const std::string &name, std::shared_ptr<BaseModule<T>> module)
        {
            submodules_map[name] = module;
            submodules.push_back(module);
        }

        virtual std::vector<std::shared_ptr<Tensor<T>>> parameters() const
        {
            std::vector<std::shared_ptr<Tensor<T>>> params;
            for (auto &sub : submodules)
            {
                auto p = sub->parameters();
                params.insert(params.end(), p.begin(), p.end());
            }
            return params;
        }

        virtual void zero_grad()
        {
            for (auto &param : parameters())
                param->zero_grad();
        }

        virtual void update(T lr)
        {
            for (auto &param : parameters())
            {
                auto g = param->get_grad();
                auto &data = param->get_data();
                for (size_t i = 0; i < data.size(); i++)
                    const_cast<std::vector<T>&>(data)[i] -= lr * g[i];
            }
        }

        virtual std::string info(int level = 0) const
        {
            std::ostringstream oss;
            std::string indent(level * 2, ' ');
            oss << indent << typeid(*this).name() << " (" << submodules.size() << " submodules)\n";
            for (auto &sub : submodules)
                oss << sub->info(level + 1);
            return oss.str();
        }

        std::shared_ptr<BaseModule<T>> operator[](const std::string &name) const
        {
            auto it = submodules_map.find(name);
            return (it != submodules_map.end()) ? it->second : nullptr;
        }

    protected:
        std::vector<std::shared_ptr<BaseModule<T>>> submodules;
        std::map<std::string, std::shared_ptr<BaseModule<T>>> submodules_map;
    };
}
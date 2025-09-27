#pragma once

// Standard headers commonly used in Core
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <random>

// Core submodules
#include "Tensor/tensor.hpp"
#include "Module/module.hpp"
#include "Module/sequential.hpp"
#include "Layer/dense.hpp"
#include "Activation/relu.hpp"
#include "Activation/sigmoid.hpp"
#include "Loss/mse.hpp"

namespace TensorNS = NovaML::Core::TensorModule;
namespace ModuleNS = NovaML::Core::Module;
namespace LayerNS = NovaML::Core::LayerModule;
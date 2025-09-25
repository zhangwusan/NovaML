```bash
ml_project/
│── CMakeLists.txt         # or Makefile
│── README.md              # project description
│── data/                  # datasets (ignored in git)
│   ├── raw/
│   └── processed/
│── include/               # public headers
│   └── ml_project/
│       ├── tensor.hpp     # custom tensor class
│       ├── dataset.hpp    # dataset loading
│       ├── model.hpp      # base model class
│       ├── layers.hpp     # layers (dense, conv, etc.)
│       ├── optimizer.hpp  # SGD, Adam, etc.
│       ├── loss.hpp       # loss functions
│       └── utils.hpp      # math helpers, logging
│── src/                   # cpp implementation
│   ├── core/              # tensor + math ops
│   │   ├── tensor.cpp
│   │   └── math_ops.cpp
│   ├── data/              # dataset loaders
│   │   └── dataset.cpp
│   ├── models/            # ML/DL models
│   │   ├── linear_regression.cpp
│   │   ├── neural_network.cpp
│   │   └── cnn.cpp
│   ├── layers/            # layer implementations
│   │   ├── dense.cpp
│   │   ├── conv2d.cpp
│   │   └── activation.cpp
│   ├── optimizers/        # training algorithms
│   │   ├── sgd.cpp
│   │   └── adam.cpp
│   ├── losses/            # loss functions
│   │   ├── mse.cpp
│   │   └── cross_entropy.cpp
│   └── main.cpp           # entry point (training loop)
│── tests/                 # unit tests
│   ├── test_tensor.cpp
│   ├── test_layers.cpp
│   ├── test_models.cpp
│   └── test_optimizers.cpp
│── scripts/               # automation
│   ├── download_data.sh
│   ├── train.sh
│   └── evaluate.sh
│── notebooks/             # optional (C++ with Jupyter via xeus-cling)
│── build/                 # compiled binaries (ignored in git)
│── third_party/           # external libraries (Eigen, CUDA, OpenCV, etc.)
```
# Makefile for NovaML project

# Build directory
BUILD_DIR := build

# Executable name
TARGET := NovaML

# CMake generator
CMAKE := cmake

# Default target: configure and build
all: configure build

# Configure the project with CMake
configure:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(CMAKE) ..

# Build the project
build:
	cd $(BUILD_DIR) && $(CMAKE) --build . --config Release

# Run the main executable
run: build
	@echo "Running $(TARGET)..."
	@$(BUILD_DIR)/bin/$(TARGET)

# Clean the build directory
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)

# Rebuild: clean + all
rebuild: clean all

.PHONY: all configure build run clean rebuild
# NanoGrad

NanoGrad is a minimal implementation of automatic differentiation (autograd) from scratch,
similar to what PyTorch uses internally. This repository serves as a learning exercise
to understand how gradient computation and neural networks work under the hood.

This repo is based on: https://github.com/karpathy/micrograd
and you can watch this video to solve it:
https://youtu.be/VMj-3S1tku0?si=hJalzx83N7gcuFX4

## Purpose

This project is designed for educational purposes to help understand:
- How automatic differentiation works
- The mechanics of neural network computation
- Backpropagation algorithm implementation
- Gradient computation in deep learning frameworks

## Implementation Status

The core components have TODO placeholders that need to be implemented:

### Value Class (Automatic Differentiation Engine)
- ✅ Basic structure and convenience operators
- ❌ **TODO**: Addition, multiplication, power operations
- ❌ **TODO**: ReLU activation function
- ❌ **TODO**: Backpropagation algorithm

### Neural Network Modules
- ✅ Basic class structure
- ❌ **TODO**: Neuron implementation (weights, bias, forward pass)
- ❌ **TODO**: Layer implementation (multiple neurons)
- ❌ **TODO**: MLP implementation (multiple layers)

## Getting Started

### Setup Environment

```bash
python3 -m venv env
source env/bin/activate
pip3 install pytest
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Implementation Guide

**See [HINTS.md](HINTS.md) for detailed implementation guidance!**

The hints file provides:
- Step-by-step implementation guidance for each TODO
- Mathematical background for automatic differentiation
- Common pitfalls and debugging tips
- Testing strategies

### Running Tests

```bash
# Run all tests
python3 -m pytest

# Run with verbose output
python3 -m pytest test/test_value.py -v

# Run specific test
python3 -m pytest test/test_value.py::test_add -v
```

The tests compare your implementation against PyTorch to verify correctness.

## Files Structure

- `NanoGrad/value.py`: Core automatic differentiation implementation (**needs completion**)
- `NanoGrad/nn.py`: Neural network module system (**needs completion**)
- `test/test_value.py`: Test suite comparing against PyTorch
- `HINTS.md`: Detailed implementation guidance
- `README.md`: This documentation

## Learning Goals

This implementation helps understand:

1. **Computational Graphs**: How operations create a graph of dependencies
2. **Forward Pass**: How values flow through the computation
3. **Backward Pass**: How gradients flow backward through operations (backpropagation)
4. **Chain Rule**: How derivatives compose in complex expressions
5. **Neural Networks**: How simple operations combine to create learning systems

## Implementation Approach

1. **Start with Value class**: Implement basic operations (`__add__`, `__mul__`, `__pow__`, `relu`)
2. **Implement backpropagation**: Complete the `backward()` method with topological sorting
3. **Build neural components**: Implement `Neuron`, `Layer`, and `MLP` classes
4. **Test frequently**: Use the provided tests to verify correctness against PyTorch

## Key Concepts

- **Automatic Differentiation**: Computing derivatives automatically through code execution
- **Computational Graph**: DAG representing the flow of computations
- **Backpropagation**: Algorithm for computing gradients via the chain rule
- **Gradient Accumulation**: How gradients combine when variables are used multiple times

## Note

This is a simplified educational implementation. Production frameworks like PyTorch include many optimizations and features not present here, such as:
- Tensor operations (this only handles scalars)
- GPU acceleration
- Memory optimization
- Advanced optimizers
- Extensive operation library

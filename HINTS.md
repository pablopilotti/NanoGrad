# Implementation Hints for NanoGrad

This file provides detailed hints and context for implementing the TODO sections in `value.py` and `nn.py`.

## Value Class (`NanoGrad/value.py`)

### 1. Addition (`__add__` method)

**Context**: Addition is the simplest operation. The forward pass computes `z = x + y`, and the backward pass distributes the gradient equally to both inputs.

**Hints**:
- Forward: `out.data = self.data + other.data`
- Backward: Both `self.grad` and `other.grad` get the same gradient from `out.grad`
- Remember to set `out._prev = {self, other}` and `out._op = '+'`

**Mathematical background**: 
- If `z = x + y`, then `∂z/∂x = 1` and `∂z/∂y = 1`
- Chain rule: `∂L/∂x = ∂L/∂z * ∂z/∂x = ∂L/∂z * 1`

### 2. Multiplication (`__mul__` method)

**Context**: Multiplication follows the product rule. Each input's gradient is the other input's value times the output gradient.

**Hints**:
- Forward: `out.data = self.data * other.data`
- Backward: `self.grad += other.data * out.grad` and `other.grad += self.data * out.grad`
- Set `out._prev = {self, other}` and `out._op = '*'`

**Mathematical background**:
- If `z = x * y`, then `∂z/∂x = y` and `∂z/∂y = x`
- Chain rule: `∂L/∂x = ∂L/∂z * ∂z/∂x = ∂L/∂z * y`

### 3. Power (`__pow__` method)

**Context**: Power operation `z = x^n` where `n` is a constant. Uses the power rule for differentiation.

**Hints**:
- Forward: `out.data = self.data ** other`
- Backward: `self.grad += (other * self.data**(other-1)) * out.grad`
- Set `out._prev = {self}` and `out._op = f'**{other}'`

**Mathematical background**:
- If `z = x^n`, then `∂z/∂x = n * x^(n-1)`
- Chain rule: `∂L/∂x = ∂L/∂z * ∂z/∂x = ∂L/∂z * n * x^(n-1)`

### 4. ReLU (`relu` method)

**Context**: ReLU (Rectified Linear Unit) is `max(0, x)`. It's a piecewise function that's linear for positive inputs and zero for negative inputs.

**Hints**:
- Forward: `out.data = max(0, self.data)` or `out.data = self.data if self.data > 0 else 0`
- Backward: `self.grad += (out.data > 0) * out.grad` (gradient is 1 if input > 0, else 0)
- Set `out._prev = {self}` and `out._op = 'ReLU'`

**Mathematical background**:
- If `z = ReLU(x) = max(0, x)`, then `∂z/∂x = 1` if `x > 0`, else `∂z/∂x = 0`

### 5. Backward Pass (`backward` method)

**Context**: This implements backpropagation using topological sorting to ensure gradients flow in the correct order.

**Hints for `build_topo` function**:
- Use depth-first search (DFS) to visit all nodes
- Only visit each node once (check `visited` set)
- Add node to `topo` list after visiting all its children
- Recursively call `build_topo` on each node in `v._prev`

**Hints for backward pass**:
- Call `build_topo(self)` to get topological order
- Set `self.grad = 1.0` (this is the starting gradient)
- Iterate through `topo` in reverse order and call `_backward()` on each node

**Why topological sort?**: We need to process nodes in reverse topological order to ensure that when we compute a node's gradient, all nodes that depend on it have already been processed.

## Neural Network Modules (`NanoGrad/nn.py`)

### 1. Module Base Class

**Context**: Base class for all neural network components. Provides common functionality like zeroing gradients and collecting parameters.

**Hints for `parameters` method**:
- Should return an empty list by default
- Subclasses will override this to return their actual parameters

### 2. Neuron Class

**Context**: A single neuron computes a weighted sum of inputs plus bias, optionally followed by an activation function.

**Hints for `__init__`**:
- Initialize `self.weights` as a list of `Value` objects with random values (try `random.uniform(-1, 1)`)
- Initialize `self.bias` as a `Value` object with random value
- Store the `nonlin` flag

**Hints for `__call__`**:
- Compute weighted sum: `sum(w * x for w, x in zip(self.weights, x)) + self.bias`
- Apply ReLU if `self.nonlin` is True
- Return the result

**Hints for `parameters`**:
- Return `self.weights + [self.bias]`

### 3. Layer Class

**Context**: A layer contains multiple neurons, each taking the same input and producing one output.

**Hints for `__init__`**:
- Create `nout` neurons, each with `nin` inputs
- Store them in `self.neurons`

**Hints for `__call__`**:
- Apply each neuron to the input `x`
- Return list of outputs (one per neuron)
- If layer has only one neuron, you might want to return the single output instead of a list

**Hints for `parameters`**:
- Collect parameters from all neurons: `[p for neuron in self.neurons for p in neuron.parameters()]`

### 4. MLP (Multi-Layer Perceptron) Class

**Context**: An MLP chains multiple layers together. The output of one layer becomes the input to the next.

**Hints for `__init__`**:
- Create layers with appropriate input/output sizes
- First layer takes `nin` inputs
- Each subsequent layer takes the previous layer's output size as input
- Last layer should have `nonlin=False` (no activation on output layer)

**Hints for `__call__`**:
- Pass input through each layer sequentially
- Output of layer `i` becomes input to layer `i+1`
- Handle the case where a layer returns a single value vs. a list

**Hints for `parameters`**:
- Collect parameters from all layers: `[p for layer in self.layers for p in layer.parameters()]`

## General Implementation Tips

1. **Random Initialization**: Use `random.uniform(-1, 1)` for weight initialization
2. **Gradient Accumulation**: Always use `+=` when updating gradients, never `=`
3. **Type Handling**: Convert non-Value inputs to Value objects when needed
4. **Debugging**: Use the `_op` field to track operations for debugging
5. **Testing**: Run the tests frequently to check your implementation against PyTorch

## Common Pitfalls

1. **Forgetting to set `_prev` and `_op`**: These are needed for the computation graph
2. **Using `=` instead of `+=` for gradients**: Gradients should accumulate, not replace
3. **Wrong gradient computation**: Double-check the mathematical derivatives
4. **Topological sort errors**: Make sure to visit nodes in the correct order
5. **Not handling scalar vs. list outputs**: Be consistent about when to return single values vs. lists

## Testing Your Implementation

The tests in `test/test_value.py` compare your implementation against PyTorch. Key things to verify:
- Forward pass produces correct values
- Backward pass produces correct gradients
- Complex expressions work correctly
- Neural network components integrate properly

Run tests with: `python -m pytest test/test_value.py -v`

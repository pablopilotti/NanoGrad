import random
from NanoGrad.value import Value
from NanoGrad.nn import Module, Neuron, Layer, MLP

def test_module_base():
    """Test the base Module class functionality"""
    module = Module()
    
    # Test that parameters returns empty list by default
    assert module.parameters() == []
    
    # Test zero_grad doesn't crash on empty parameters
    module.zero_grad()

def test_neuron_initialization():
    """Test neuron initialization"""
    # Set seed for reproducible tests
    random.seed(42)
    
    neuron = Neuron(3, nonlin=True)
    
    # Check that weights and bias are Value objects
    assert len(neuron.w) == 3
    assert all(isinstance(w, Value) for w in neuron.w)
    assert isinstance(neuron.b, Value)
    
    # Check nonlinearity flag
    assert neuron.nonlin == True
    
    # Test without nonlinearity
    neuron_linear = Neuron(2, nonlin=False)
    assert neuron_linear.nonlin == False

def test_neuron_forward_pass():
    """Test neuron forward pass computation"""
    # Create a neuron with known weights for testing
    neuron = Neuron(2, nonlin=False)
    neuron.w = [Value(0.5), Value(-0.3)]
    neuron.b = Value(0.1)
    
    # Test forward pass
    x = [Value(1.0), Value(2.0)]
    output = neuron(x)
    
    # Expected: 0.5*1.0 + (-0.3)*2.0 + 0.1 = 0.5 - 0.6 + 0.1 = 0.0
    assert abs(output.data - 0.0) < 1e-10
    
    # Test with ReLU
    neuron.nonlin = True
    output_relu = neuron(x)
    assert output_relu.data == 0.0  # ReLU(0.0) = 0.0
    
    # Test with positive output
    neuron.b = Value(1.0)  # Now output should be 0.9
    output_positive = neuron(x)
    assert output_positive.data == 0.9  # ReLU(0.9) = 0.9

def test_neuron_parameters():
    """Test neuron parameters method"""
    neuron = Neuron(3)
    params = neuron.parameters()
    
    # Should return weights + bias
    assert len(params) == 4  # 3 weights + 1 bias
    assert all(isinstance(p, Value) for p in params)
    assert params[-1] is neuron.b  # Last parameter should be bias

def test_neuron_backward_pass():
    """Test neuron backward pass"""
    neuron = Neuron(2, nonlin=False)
    neuron.w = [Value(2.0), Value(3.0)]
    neuron.b = Value(1.0)
    
    x = [Value(1.0), Value(-1.0)]
    output = neuron(x)
    output.backward()
    
    # Check gradients
    assert neuron.w[0].grad == x[0].data  # dL/dw0 = x0
    assert neuron.w[1].grad == x[1].data  # dL/dw1 = x1
    assert neuron.b.grad == 1.0  # dL/db = 1
    assert x[0].grad == neuron.w[0].data  # dL/dx0 = w0
    assert x[1].grad == neuron.w[1].data  # dL/dx1 = w1

def test_layer_initialization():
    """Test layer initialization"""
    layer = Layer(3, 2, nonlin=True)
    
    # Check number of neurons
    assert len(layer.neurons) == 2
    assert all(isinstance(n, Neuron) for n in layer.neurons)
    
    # Check that each neuron has correct input size
    for neuron in layer.neurons:
        assert len(neuron.w) == 3
        assert neuron.nonlin == True

def test_layer_forward_pass():
    """Test layer forward pass"""
    layer = Layer(2, 3, nonlin=False)
    
    # Set known weights for testing
    for i, neuron in enumerate(layer.neurons):
        neuron.w = [Value(i + 1), Value(i + 2)]
        neuron.b = Value(i)
    
    x = [Value(1.0), Value(1.0)]
    outputs = layer(x)
    
    # Check outputs
    assert len(outputs) == 3
    assert all(isinstance(o, Value) for o in outputs)
    
    # Expected outputs:
    # neuron 0: 1*1 + 2*1 + 0 = 3
    # neuron 1: 2*1 + 3*1 + 1 = 6
    # neuron 2: 3*1 + 4*1 + 2 = 9
    expected = [3.0, 6.0, 9.0]
    for i, (output, exp) in enumerate(zip(outputs, expected)):
        assert abs(output.data - exp) < 1e-10, f"Neuron {i}: expected {exp}, got {output.data}"

def test_layer_parameters():
    """Test layer parameters method"""
    layer = Layer(2, 3)
    params = layer.parameters()
    
    # Should return all parameters from all neurons
    # Each neuron has 2 weights + 1 bias = 3 params
    # 3 neurons * 3 params = 9 total params
    assert len(params) == 9
    assert all(isinstance(p, Value) for p in params)

def test_mlp_initialization():
    """Test MLP initialization"""
    mlp = MLP(3, [4, 2, 1])
    
    # Check number of layers
    assert len(mlp.layers) == 3
    
    # Check layer sizes
    assert len(mlp.layers[0].neurons) == 4  # First layer: 4 neurons
    assert len(mlp.layers[1].neurons) == 2  # Second layer: 2 neurons
    assert len(mlp.layers[2].neurons) == 1  # Third layer: 1 neuron
    
    # Check input sizes
    assert len(mlp.layers[0].neurons[0].w) == 3  # First layer takes 3 inputs
    assert len(mlp.layers[1].neurons[0].w) == 4  # Second layer takes 4 inputs
    assert len(mlp.layers[2].neurons[0].w) == 2  # Third layer takes 2 inputs

def test_mlp_forward_pass():
    """Test MLP forward pass"""
    # Create a simple 2-layer MLP
    mlp = MLP(2, [2, 1])
    
    # Set known weights for testing
    # First layer
    mlp.layers[0].neurons[0].w = [Value(1.0), Value(0.0)]
    mlp.layers[0].neurons[0].b = Value(0.0)
    mlp.layers[0].neurons[1].w = [Value(0.0), Value(1.0)]
    mlp.layers[0].neurons[1].b = Value(0.0)
    
    # Second layer
    mlp.layers[1].neurons[0].w = [Value(1.0), Value(1.0)]
    mlp.layers[1].neurons[0].b = Value(0.0)
    
    # Test forward pass
    x = [Value(2.0), Value(3.0)]
    output = mlp(x)
    
    # First layer outputs: [2.0, 3.0] (identity transformation)
    # Second layer output: 2.0 + 3.0 = 5.0
    assert isinstance(output, Value)  # Single output neuron returns Value directly
    assert abs(output.data - 5.0) < 1e-10

def test_mlp_parameters():
    """Test MLP parameters method"""
    mlp = MLP(2, [3, 1])
    params = mlp.parameters()
    
    # First layer: 3 neurons * (2 weights + 1 bias) = 9 params
    # Second layer: 1 neuron * (3 weights + 1 bias) = 4 params
    # Total: 13 params
    assert len(params) == 13
    assert all(isinstance(p, Value) for p in params)

def test_mlp_backward_pass():
    """Test MLP backward pass"""
    mlp = MLP(2, [2, 1])
    
    # Simple forward pass
    x = [Value(1.0), Value(2.0)]
    output = mlp(x)
    
    # Backward pass
    output.backward()
    
    # Check that gradients are computed (non-zero for at least some parameters)
    params = mlp.parameters()
    grad_sum = sum(abs(p.grad) for p in params)
    assert grad_sum > 0, "No gradients computed"

def test_zero_grad():
    """Test zero_grad functionality"""
    mlp = MLP(2, [2, 1])
    
    # Forward and backward pass to set gradients
    x = [Value(1.0), Value(2.0)]
    output = mlp(x)
    output.backward()
    
    # Check that some gradients are non-zero
    params = mlp.parameters()
    grad_sum_before = sum(abs(p.grad) for p in params)
    assert grad_sum_before > 0
    
    # Zero gradients
    mlp.zero_grad()
    
    # Check that all gradients are zero
    grad_sum_after = sum(abs(p.grad) for p in params)
    assert grad_sum_after == 0

def test_string_representations():
    """Test string representations of neural network components"""
    neuron = Neuron(3, nonlin=True)
    layer = Layer(4, 2)
    mlp = MLP(3, [4, 2, 1])
    
    # Test that repr methods don't crash and return strings
    assert isinstance(repr(neuron), str)
    assert isinstance(repr(layer), str)
    assert isinstance(repr(mlp), str)
    
    # Test some basic content
    assert "3" in repr(neuron)  # Should mention input size
    assert "4" in repr(layer)  # Should mention input size
    assert "3" in repr(mlp)  # Should mention some layer size

def test_integration_with_value():
    """Test integration between neural network components and Value class"""
    # Create a simple network
    mlp = MLP(1, [2, 1])
    
    # Test with Value inputs
    x = [Value(1.5)]
    output = mlp(x)
    
    # Should return Value objects
    assert isinstance(output, Value)
    
    # Test backward pass
    output.backward()
    
    # Input should have gradient
    assert x[0].grad != 0

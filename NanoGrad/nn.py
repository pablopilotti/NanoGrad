import random
from NanoGrad.value import Value

class Module:
    """
    Base class for all neural network modules.
    
    TODO: Implement zero_grad method to reset gradients
    TODO: Implement parameters method to return all parameters
    """

    def zero_grad(self):
        """
        TODO: Reset all parameter gradients to zero
        """
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        """
        TODO: Return all parameters (Value objects) in this module
        """
        return []

class Neuron(Module):
    """
    A single neuron (neural unit).
    
    TODO: Implement initialization with input size and activation function
    TODO: Implement forward pass (__call__)
    TODO: Implement parameters method to return neuron's weights and bias
    TODO: Implement string representation
    """

    def __init__(self, nin, nonlin=True):
        """
        TODO: Initialize neuron with nin inputs
        TODO: Create weights and bias as Value objects
        TODO: Set non-linearity flag
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        """
        TODO: Implement forward pass computation
        TODO: Apply activation function if nonlin=True
        """
        # Compute weighted sum: w1*x1 + w2*x2 + ... + b
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        return act.relu() if self.nonlin else act

    def parameters(self):
        """
        TODO: Return neuron's weights and bias
        """
        return self.w + [self.b]

    def __repr__(self):
        """
        TODO: Return string representation of the neuron
        """
        return f"Neuron({len(self.w)}, nonlin={self.nonlin})"

class Layer(Module):
    """
    A layer of neurons.
    
    TODO: Implement initialization with input and output sizes
    TODO: Implement forward pass (__call__)
    TODO: Implement parameters method to return all layer parameters
    TODO: Implement string representation
    """

    def __init__(self, nin, nout, **kwargs):
        """
        TODO: Initialize layer with nin inputs and nout outputs
        TODO: Create neurons in the layer
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """
        TODO: Implement forward pass through all neurons in the layer
        """
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """
        TODO: Return all parameters from all neurons in the layer
        """
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        """
        TODO: Return string representation of the layer
        """
        return f"Layer({len(self.neurons[0].w)}, {len(self.neurons)})"

class MLP(Module):
    """
    Multi-Layer Perceptron.
    
    TODO: Implement initialization with input size and layer sizes
    TODO: Implement forward pass (__call__)
    TODO: Implement parameters method to return all MLP parameters
    TODO: Implement string representation
    """

    def __init__(self, nin, nouts):
        """
        TODO: Initialize MLP with input size and list of output sizes for each layer
        TODO: Create layers in the network
        """
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        """
        TODO: Implement forward pass through all layers
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """
        TODO: Return all parameters from all layers in the MLP
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """
        TODO: Return string representation of the MLP
        """
        layer_sizes = [len(self.layers[0].neurons[0].w)] + [len(layer.neurons) for layer in self.layers]
        return f"MLP({', '.join(str(size) for size in layer_sizes)})"


import random
from NanoGrad.value import Value


class Module:
    """
    Base class for all neural network modules.

    Implement zero_grad method to reset gradients
    Implement parameters method to return all parameters
    """

    def zero_grad(self):
        """
        Reset all parameter gradients to zero
        """
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        """
        Return all parameters (Value objects) in this module
        """
        return []


class Neuron(Module):
    """
    A single neuron (neural unit).

    Implement initialization with input size and activation function
    Implement forward pass (__call__)
    Implement parameters method to return neuron's weights and bias
    Implement string representation
    """

    def __init__(self, nin, nonlin=True):
        """
        Initialize neuron with nin inputs
        Create weights and bias as Value objects
        Set non-linearity flag
        """
        self.w = [Value(random.random()) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """
        Implement forward pass computation
        Apply activation function if nonlin=True
        """
        ret = sum([w * xx for w, xx in zip(self.w, x)], self.b)
        return ret.relu() if self.nonlin else ret

    def parameters(self):
        """
        Return neuron's weights and bias
        """
        return [w for w in self.w] + [self.b]

    def __repr__(self):
        """
        Return string representation of the neuron
        """
        return f"Neuron({len(self.w)}, nonlin={self.nonlin})"


class Layer(Module):
    """
    A layer of neurons.

    Implement initialization with input and output sizes
    Implement forward pass (__call__)
    Implement parameters method to return all layer parameters
    Implement string representation
    """

    def __init__(self, nin, nout, **kwargs):
        """
        Initialize layer with nin inputs and nout outputs
        Create neurons in the layer
        """
        self.nin = nin
        self.nout = nout
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        """
        Implement forward pass through all neurons in the layer
        """
        return [n(x) for n in self.neurons]

    def parameters(self):
        """
        Return all parameters from all neurons in the layer
        """
        return [p for x in self.neurons for p in x.parameters()]

    def __repr__(self):
        """
        Return string representation of the layer
        """
        return f"Layer({len(self.neurons[0].w)}, {len(self.neurons)})"


class MLP(Module):
    """
    Multi-Layer Perceptron.

    Implement initialization with input size and layer sizes
    Implement forward pass (__call__)
    Implement parameters method to return all MLP parameters
    Implement string representation
    """

    def __init__(self, nin, nouts):
        """
        Initialize MLP with input size and list of output sizes for each layer
        Create layers in the network
        """
        self.layers = []
        myin = nin
        for myout in nouts:
            self.layers.append(Layer(myin, myout))
            myin = myout

    def __call__(self, x):
        """
        Implement forward pass through all layers
        """
        ret = x
        for layer in self.layers:
            ret = layer(ret)
        return ret[0]

    def parameters(self):
        """
        Return all parameters from all layers in the MLP
        """
        return [p for x in self.layers for p in x.parameters()]

    def __repr__(self):
        """
        Return string representation of the MLP
        """
        layer_sizes = [len(self.layers[0].neurons[0].w)] + [
            len(layer.neurons) for layer in self.layers
        ]
        return f"MLP({', '.join(str(size) for size in layer_sizes)})"

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
        pass

    def parameters(self):
        """
        TODO: Return all parameters (Value objects) in this module
        """
        pass

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
        # self.w =
        # self.b = 
        # self.nonlin = 

    def __call__(self, x):
        """
        TODO: Implement forward pass computation
        TODO: Apply activation function if nonlin=True
        """
        # Compute weighted sum
        pass

    def parameters(self):
        """
        TODO: Return neuron's weights and bias
        """
        pass

    def __repr__(self):
        """
        TODO: Return string representation of the neuron
        """
        pass

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
        pass

    def __call__(self, x):
        """
        TODO: Implement forward pass through all neurons in the layer
        """
        pass

    def parameters(self):
        """
        TODO: Return all parameters from all neurons in the layer
        """
        pass

    def __repr__(self):
        """
        TODO: Return string representation of the layer
        """
        pass

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
        pass

    def __call__(self, x):
        """
        TODO: Implement forward pass through all layers
        """
        pass
    
    def parameters(self):
        """
        TODO: Return all parameters from all layers in the MLP
        """
        pass

    def __repr__(self):
        """
        TODO: Return string representation of the MLP
        """
        pass


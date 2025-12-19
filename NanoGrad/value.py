class Value:
    """
    Stores a single scalar value and its gradient.
    This is the core building block for automatic differentiation.
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data              # the actual scalar value
        self.grad = 0.0               # gradient of this value w.r.t. final output

        # --- Autograd internals ---
        self._backward = lambda: None # function to propagate gradients to parents
        self._prev = set(_children)   # previous Value nodes in the graph
        self._op = _op                # operation that produced this node (for debugging)

    def __add__(self, other):
        """
        Addition: z = x + y
        """
        other = other if isinstance(other, Value) else Value(other)

        # TODO: compute forward and backward pass
        out = Value()

        def _backward():
            pass

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Multiplication: z = x * y
        """
        other = other if isinstance(other, Value) else Value(other)

        # TODO: compute forward and backward pass
        out = Value()

        def _backward():
            pass

        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        Power: z = x ** n (n is a constant)
        """
        assert isinstance(other, (int, float)), "only int/float powers supported"

        # TODO: compute forward and backward pass
        out = Value()

        def _backward():
            pass

        out._backward = _backward
        return out

    def relu(self):
        """
        ReLU activation: max(0, x)
        """

        # TODO: compute forward and backward pass
        out = Value()

        def _backward():
            pass

        out._backward = _backward
        return out

    def backward(self):
        """
        Runs backpropagation starting from this node.
        """

        # TODO
        # --- Topological sort of the computation graph ---
        topo = []
        visited = set()

        def build_topo(v):
            pass

        # --- Backward pass ---
        # TODO
        # call build_topo(self)
        # make grads = 0 for all nodes in topo
        # set self.grad = 1.0


        for t in reversed(topo):
            t._backward()

    # ---- Convenience operators ----

    def __neg__(self):        # -x
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):   # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op={self._op})"

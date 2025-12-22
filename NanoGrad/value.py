class Value:
    """
    Stores a single scalar value and its gradient.
    This is the core building block for automatic differentiation.
    
    TODO: Implement the forward and backward passes for all operations
    This class should support basic arithmetic operations with automatic differentiation
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

        out = Value(
            self.data + other.data,
            (self, other),
            '+'
        )

        def _backward():
            self.grad += out.grad  
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Multiplication: z = x * y
        """
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            '*'
        )

        def _backward():
            self.grad += out.grad * other.data 
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        Power: z = x ** n (n is a constant)
        """

        assert isinstance(other, (int, float)), "only int/float powers supported"

        out = Value(
            self.data ** other,
            (self,),
            '^'
        )

        def _backward():
            self.grad += out.grad * other *  self.data ** (other-1)


        out._backward = _backward
        return out

    def relu(self):
        """
        ReLU activation: max(0, x)
        """
        out = Value(
            self.data if self.data > 0 else 0 ,
            (self,),
            'Relu'
        )

        def _backward():
            self.grad += out.grad if self.data > 0 else 0

        out._backward = _backward
        return out

    def backward(self):
        """
        Runs backpropagation starting from this node.
        """
        # --- Topological sort of the computation graph ---
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for n in v._prev:
                    build_topo(n)
                topo.append(v)

            

        # --- Backward pass ---
        build_topo(self)
        for g in topo:
            g.grad = 0

        self.grad = 1.0


        for t in reversed(topo):
            t._backward()

    # ---- Convenience operators ---

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

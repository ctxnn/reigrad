import math
class Val:
    """val will store a scalar value(data) and its gradient"""
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        self._prev = set(_children) # used for graph construction (to do topological sort etc)
        self._backward = lambda:None # write backward for only operations that are used in ANNs

    def __repr__(self):
        return f"Val(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Val) else Val(other)
        out = Val(self.data + other.data)


        def _backward():
            self.grad += 1.0 * out.grad # ∂L/∂x = (∂L/∂out) * (∂out/∂x) [out.grad is ∂L/∂out (upstream gradient), 1.0 is ∂out/∂x (local gradient)]
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Val) else Val(other)
        out = Val(self.data * other.data)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Val(0 if self.data < 0 else self.data, (self,))
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out  

    def exp(self):
            out = Val(math.exp(self.data), (self,))

            def _backward():
                self.grad += out.data * out.grad
            out._backward = _backward
            return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Val(self.data**other, (self,))

        def _backward():
                self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        out = Val(math.tanh(self.data), (self,))
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out
    def __truediv__(self, other):
        out = Val(self * other**-1)
        return out

    def __rmul__(self, other):
        other = other if isinstance(other, Val) else Val(other)
        out = Val(other.data *self.data)
        return out

    def __radd__(self, other):
      other = other if isinstance(other, Val) else Val(other)
      out = Val(other.data + self.data)
      return out

    def backward(self):
        # write topological sort (it sorts the graph(we constructed the graph using _prev) left to right)
        topo_order = []
        visited = set()
        def build_order(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_order(child)
                topo_order.append(v)
        build_order(self)

        # calculating grad one by one and applying the chain rule to get their gradient
        self.grad = 1
        for v in reversed(topo_order):
            v._backward()

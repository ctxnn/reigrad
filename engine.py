import math
class Val:
    """val will store a scalar value(data) and its gradient"""
    def __init__(self, data):
        self.data = data
        self.grad = 0

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Val) else Val(other)
        out = self.data + other.data
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Val) else Val(other)
        out = self.data * other.data
        return out

    def exp(self):
        out = math.exp(self.data)
        return out

    def __truediv__(self, other):
        out = self * other**-1
        return out

    def __rmul__(self, other):
        out = other.data *self.data

a = 2
b = Val(a)
print(b*2)

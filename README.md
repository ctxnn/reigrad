# Reigrad

A simple autograd engine I built to learn about automatic differentiation and neural networks from scratch. This is my personal learning project to understand how frameworks like PyTorch work under the hood.

<!-- ## 📝 What I'm Learning

- How automatic differentiation works
- Building neural networks from scratch
- Understanding backpropagation
- Implementing basic deep learning concepts -->

## Key Components

### The Val Class (engine.py)
The core class that handles automatic differentiation:
```python
# Simple example of how it works
x = Val(2.0)
y = Val(3.0)
z = x * y + x.tanh()
z.backward()  # Computes all gradients
```

### Neural Networks (nn.py)
Basic neural network implementation:
```python
# Creating a small network
model = MLP(nin=2, nouts=[16, 16, 1])  # 2 inputs, 2 hidden layers, 1 output
```

### Tests (test_engine.py)
Tests to make sure everything works:
```python
# Testing gradients
x = Val(2.0)
y = x * x
y.backward()
assert x.grad == 4.0  # dy/dx = 2x at x=2
```

## Running the Code

Just clone and run the tests:
```bash
git clone https://github.com/yourusername/reigrad.git
cd reigrad
python test_engine.py
```

## Learning Resources

These resources helped me understand the concepts:
- Andrej Karpathy's micrograd
- PyTorch documentation
- Various autograd tutorials

---

<div align="center">
  built with ❤️ by ctxnn
</div>

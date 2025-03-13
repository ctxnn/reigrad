import numpy as np
from engine import Val
from nn import MLP
import math

def test_basic_ops():
    """Test basic operations of the autograd engine"""
    print("\n=== Testing Basic Operations ===")
    
    # Test addition
    a = Val(2.0)
    b = Val(3.0)
    c = a + b
    c.backward()
    print(f"Addition Test: {a.grad == 1.0 and b.grad == 1.0}")
    print(f"  a.grad: {a.grad}, b.grad: {b.grad} (expected: 1.0, 1.0)")
    
    # Test multiplication
    a = Val(2.0)
    b = Val(3.0)
    c = a * b
    c.backward()
    print(f"Multiplication Test: {a.grad == 3.0 and b.grad == 2.0}")
    print(f"  a.grad: {a.grad}, b.grad: {b.grad} (expected: 3.0, 2.0)")
    
    # Test power
    a = Val(2.0)
    c = a ** 2
    c.backward()
    print(f"Power Test: {abs(a.grad - 4.0) < 1e-6}")
    print(f"  a.grad: {a.grad} (expected: 4.0)")
    
    # Test tanh
    x = Val(2.0)
    y = x.tanh()
    y.backward()
    expected_grad = 1 - math.tanh(2.0)**2
    print(f"Tanh Test: {abs(x.grad - expected_grad) < 1e-6}")
    print(f"  x.grad: {x.grad} (expected: {expected_grad})")

def test_complex_computation():
    """Test a more complex computation graph"""
    print("\n=== Testing Complex Computation ===")
    
    # Create a simple computation: f(x) = (x + 2)^2 * tanh(x)
    x = Val(1.5)
    term1 = (x + 2) ** 2
    term2 = x.tanh()
    y = term1 * term2
    y.backward()
    
    print(f"Complex computation test:")
    print(f"  x.data: {x.data}")
    print(f"  term1.data: {term1.data}")
    print(f"  term2.data: {term2.data}")
    print(f"  y.data: {y.data}")
    print(f"  x.grad: {x.grad}")

def test_neural_network():
    """Test the neural network implementation"""
    print("\n=== Testing Neural Network ===")
    
    # Create a simple dataset
    X = [[2.0, 3.0], [-1.0, -1.0], [0.0, 4.0], [1.0, -2.0]]
    y = [1, -1, 1, -1]  # Binary classification
    
    # Create a small neural network
    model = MLP(nin=2, nouts=[4, 1])
    
    # Test forward pass
    all_outputs = []
    for x in X:
        output = model(x)
        all_outputs.append(output.data)
    
    print(f"Forward pass outputs:")
    for i, (x, out) in enumerate(zip(X, all_outputs)):
        print(f"  Input {x} -> Output {out:.6f}")
    
    # Test backward pass
    x = X[0]
    output = model(x)
    output.backward()
    
    # Check gradients
    print("\nGradient statistics:")
    grads = [p.grad for p in model.parameters()]
    print(f"  Min gradient: {min(grads):.6f}")
    print(f"  Max gradient: {max(grads):.6f}")
    print(f"  Mean gradient: {sum(grads)/len(grads):.6f}")

def run_all_tests():
    """Run all test cases"""
    test_basic_ops()
    test_complex_computation()
    test_neural_network()

if __name__ == "__main__":
    run_all_tests()

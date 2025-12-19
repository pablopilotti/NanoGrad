import torch
from NanoGrad.value import Value

def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

def test_relu():
    # Test ReLU forward pass
    x = Value(-2.0)
    y = x.relu()
    assert y.data == 0.0
    
    x = Value(2.0)
    y = x.relu()
    assert y.data == 2.0
    
    # Test ReLU backward pass
    x = Value(-2.0)
    y = x.relu()
    y.backward()
    assert x.grad == 0.0
    
    x = Value(2.0)
    y = x.relu()
    y.backward()
    assert x.grad == 1.0

def test_pow():
    # Test power forward pass
    x = Value(2.0)
    y = x**3
    assert y.data == 8.0
    
    # Test power backward pass
    x = Value(2.0)
    y = x**3
    y.backward()
    # Gradient of x^3 is 3*x^2, so 3*2^2 = 12
    assert abs(x.grad - 12.0) < 1e-10

def test_add():
    # Test addition forward pass
    x = Value(2.0)
    y = Value(3.0)
    z = x + y
    assert z.data == 5.0
    
    # Test addition backward pass
    x = Value(2.0)
    y = Value(3.0)
    z = x + y
    z.backward()
    assert x.grad == 1.0
    assert y.grad == 1.0

def test_mul():
    # Test multiplication forward pass
    x = Value(2.0)
    y = Value(3.0)
    z = x * y
    assert z.data == 6.0
    
    # Test multiplication backward pass
    x = Value(2.0)
    y = Value(3.0)
    z = x * y
    z.backward()
    assert x.grad == 3.0
    assert y.grad == 2.0

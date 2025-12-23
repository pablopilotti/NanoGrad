
import torch
import torch.nn.functional as F
from pathlib import Path

# Import our new modules
from .config import ctoi, itoc, VOCAB_SIZE
from .data_loader import load_bigram_tensors

def bigram_model():
    """Bigram model using counting probabilities"""
    xs, ys, P, _ = load_bigram_tensors()
    
    # Calculate loss
    loss = -torch.log(P[xs, ys]).mean()
    return P, loss.item()

def generate_with_bigram(model, num_samples, seed=2147483647):
    """Generate names using the bigram model"""
    P = model
    g_cpu = torch.Generator().manual_seed(seed)
    
    names = []
    for _ in range(num_samples):
        select = 0  # Start with '.'
        chars = []
        while True:
            prob = P[select]
            select = torch.multinomial(prob, num_samples=1, replacement=True, generator=g_cpu).item()
            if select == ctoi['.']:  # End token
                break
            chars.append(itoc[select])
        names.append(''.join(chars))
    return names

def train_neural_net():
    """Train neural network bigram model"""
    xs, ys, _, _ = load_bigram_tensors()
    
    # Training setup
    learning_rate = 100.0
    g_cpu = torch.Generator().manual_seed(2147483647)
    W = torch.rand((VOCAB_SIZE, VOCAB_SIZE), dtype=torch.float, requires_grad=True, generator=g_cpu)
    
    for i in range(1000):
        # Forward pass
        xenc = F.one_hot(xs, num_classes=VOCAB_SIZE).float()
        logits = xenc @ W
        counts = logits.exp()
        prob = counts / counts.sum(1, keepdim=True)
        
        # Loss with L2 regularization
        loss = -prob[torch.arange(prob.shape[0]), ys].log().mean() + 0.001 * (W**2).mean()
        
        # Adjust learning rate
        if i % 100 == 0:
            learning_rate /= 2.0
            print(f"Step {i}, Loss: {loss.item()}")
        
        # Backward pass
        W.grad = None
        loss.backward()
        W.data += -learning_rate * W.grad
    
    return W

def train():
    xs, ys, _, _ = load_bigram_tensors()
    step = 100.0
    g_cpu = torch.Generator().manual_seed(2147483647)
    W = torch.rand((27,27),dtype=torch.float, requires_grad = True, generator=g_cpu)
    for i in range(1000):
        # Forward pass
        xenc = F.one_hot(xs, num_classes = len(itoc)).float()
        logits = xenc @ W
        counts = logits.exp()
        prob = counts / counts.sum(1,keepdim=True)

        loss = - prob[torch.arange(prob.shape[0]), ys].log().mean() + 0.001*(W**2).mean()
        if i % 100 == 0:
            step /= 2.0
            print(step, loss)

        # print(ys.shape)
        W.grad = None
        loss.backward()

        W.data += -step*W.grad

    return W     

def load_W():
    if not Path("W.pt").is_file():
        W = train()
    else:
        W = torch.load("W.pt")
    torch.save(W, "W.pt")
    return W

def load_or_train_W():
    """Load trained weights or train if not present"""
    if not Path("W.pt").is_file():
        W = train_neural_net()
        torch.save(W, "W.pt")
    else:
        W = torch.load("W.pt")
    return W

def generate_with_neural_net(W, num_samples, seed=2147483647):
    """Generate names using the neural network model"""
    generator = torch.Generator().manual_seed(seed)
    
    names = []
    for _ in range(num_samples):
        idx = 0  # Start with '.'
        chars = []
        while True:
            xenc = F.one_hot(torch.tensor([idx]), num_classes=VOCAB_SIZE).float()
            logits = xenc @ W
            counts = logits.exp()
            prob = counts / counts.sum(1, keepdim=True)
            
            idx = torch.multinomial(prob, num_samples=1, replacement=True, generator=generator).item()
            if idx == ctoi['.']:  # End token
                break
            chars.append(itoc[idx])
        names.append(''.join(chars))
    return names

def main():
    # Bigram model
    P, loss = bigram_model()
    print(f"Bigram model loss: {loss:.4f}")
    print("Generated names with bigram model:")
    bigram_names = generate_with_bigram(P, 10)
    for name in bigram_names:
        print(name)
    
    print("\n" + "="*50 + "\n")
    
    # Neural network model
    W = load_or_train_W()
    print("Generated names with neural network:")
    nn_names = generate_with_neural_net(W, 10)
    for name in nn_names:
        print(name)

if __name__=="__main__":
    main()

# Exercises:
# E01: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. 
#      Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model?
# E02: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. 
#      Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?
# E03: use the dev set to tune the strength of smoothing (or regularization) for the trigram model - 
#       i.e. try many possibilities and see which one works best based on the dev set loss. 
#       What patterns can you see in the train and dev set loss as you tune this strength? 
#       Take the best setting of the smoothing and evaluate on the test set once and at the end. 
#       How good of a loss do you achieve?
# E04: we saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. 
#      Can you delete our use of F.one_hot in favor of simply indexing into rows of W?
# E05: look up and use F.cross_entropy instead. You should achieve the same result. 
#       Can you think of why we'd prefer to use F.cross_entropy instead?
# E06: meta-exercise! Think of a fun/interesting exercise and complete it.

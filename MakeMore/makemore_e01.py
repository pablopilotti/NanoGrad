
import torch
import torch.nn.functional as F
from pathlib import Path
import random
import matplotlib.pyplot as plt 

# Character mapping configuration
START_TOKEN = '.'
END_TOKEN = '.'

# Build character to index and index to character mappings
chars = [START_TOKEN] + [chr(ord('a') + i) for i in range(26)]
ctoi = {ch: i for i, ch in enumerate(chars)}
itoc = {i: ch for i, ch in enumerate(chars)}
VOCAB_SIZE = len(chars)

def load_names():
    """
    Load and preprocess names from names.txt file.
    
    Returns:
        list: A list of names as strings, each split by whitespace from the file
        
    Raises:
        FileNotFoundError: If names.txt cannot be found in the expected locations
        
    The function tries to find names.txt in the current directory or in the MakeMore/
    subdirectory. Once found, it reads the file and splits the content by whitespace.
    """
    # Try different paths
    paths_to_try = ['names.txt', 'MakeMore/names.txt']
    for path in paths_to_try:
        if Path(path).exists():
            with open(path, 'r') as file:
                names = file.read().split()
            return names
    raise FileNotFoundError("Could not find names.txt in any of the expected locations")

def create_trigram_dataset(names):
    """
    Create a trigram dataset from a list of names.
    
    Args:
        names (list): List of names as strings
        
    Returns:
        tuple: (xs, ys) where both are torch.Tensor of integers
               xs contains indices of first characters in trigrams
               ys contains indices of second characters in trigrams
               
    Each name is surrounded with start and end tokens. For each adjacent pair
    of characters in the processed names, the first character index goes to xs
    and the second to ys. This creates the training data for the trigram model.
    """
    xs, ys = [], []
    for name in names:
        # Add start and end tokens
        name_tokens = START_TOKEN + name + END_TOKEN
        for c1, c2, c3 in zip(name_tokens, name_tokens[1:],name_tokens[2:]):
            xs.append((ctoi[c1],ctoi[c2]))
            ys.append(ctoi[c3])
    return torch.tensor(xs), torch.tensor(ys)

def save_trigram_tensors():
    """
    Generate and save trigram tensors to disk for faster loading.
    
    This function processes the names, computes the trigram statistics,
    and saves the following tensors:
    - xs.pt: Input character indices
    - ys.pt: Target character indices
    - P.pt: Probability matrix of transitions between characters
    - N.pt: Count matrix of transitions between characters
    
    The count matrix N starts with ones for smoothing and is incremented
    by each observed trigram transition. P is the normalized probability matrix.
    """
    names = load_names()
    xs, ys = create_trigram_dataset(names)
    
    # Create count matrix N
   
    # 1. Calculate the rank (number of dimensions)
    # If xs is [196113, 2], x_dim is 2. If xs is [196113], x_dim is 1.
    x_dim = xs.ndim 

    # 2. Create the shape for N
    # We need (x_dim + 1) dimensions in total to include ys
    siz = (VOCAB_SIZE,) * (x_dim + 1)
    N = torch.ones(siz, dtype=torch.int32)

    # 3. Iterate and index dynamically
    for x, y in zip(xs, ys):
        if x_dim > 1:
            # If x is a tensor [a, b], *x.tolist() turns it into indices a, b
            # Then y is added as the final index
            N[(*x.tolist(), y)] += 1
        else:
            # If x is just a scalar (integer)
            N[x.item(), y.item()] += 1
    
    # Create probability matrix P
    P = N / N.sum(-1, keepdim=True)
    
    # Save tensors
    torch.save(xs, "xs.pt")
    torch.save(ys, "ys.pt")
    torch.save(P, "P.pt")
    torch.save(N, "N.pt")

def load_trigram_tensors():
    """
    Load precomputed trigram tensors from disk, generating them if missing.
    
    Returns:
        tuple: (xs, ys, P, N) where:
            xs (torch.Tensor): Input character indices
            ys (torch.Tensor): Target character indices  
            P (torch.Tensor): Probability matrix
            N (torch.Tensor): Count matrix
            
    If the tensor files don't exist, they are generated and saved first.
    This provides a convenient way to cache the preprocessed data.
    """
    if not Path("xs.pt").is_file():
        save_trigram_tensors()
    return (
        torch.load("xs.pt"),
        torch.load("ys.pt"),
        torch.load("P.pt"),
        torch.load("N.pt")
    )

def trigram_model():
    """
    Create and evaluate a counting-based trigram model.
    
    Returns:
        tuple: (P, loss) where:
            P (torch.Tensor): The probability matrix
            loss (float): The negative log likelihood loss on the dataset
            
    This model uses maximum likelihood estimation via counting to learn
    character transition probabilities. The loss is computed as the average
    negative log likelihood of the true transitions in the dataset.
    """
    xs, ys, P, _ = load_trigram_tensors()
    
    # Calculate loss
    # 1. Split xs into its constituent columns (as a tuple)
    # 2. Append ys to that tuple
    indices = (*torch.unbind(xs, dim=1), ys) if xs.ndim > 1 else (xs, ys)

    # 3. Use the tuple to index P
    probs = P[indices]

    loss = -torch.log(probs).mean()    

    return P, loss.item()

def generate_with_trigram(model, num_samples, seed=2147483647):
    """
    Generate names using the trigram probability model.
    
    Args:
        model (torch.Tensor): Probability matrix P from trigram_model()
        num_samples (int): Number of names to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        list: A list of generated names as strings
        
    The generation starts with the start token and samples characters
    according to the transition probabilities until an end token is generated.
    """
    P = model
    g_cpu = torch.Generator().manual_seed(seed)
    
    names = []
    for _ in range(num_samples):
        select = (ctoi['.'],) * (P.ndim - 1)
        chars = []
        while True:
            prob = P[select]
            new = torch.multinomial(prob, num_samples=1, replacement=True, generator=g_cpu).item()
            select = select[1:] + (new,)
            if new == ctoi['.']:  # End token
                break
            chars.append(itoc[new])
        names.append(''.join(chars))
    return names

def train_neural_net():
    """
    Train a neural network-based trigram model using gradient descent.
    
    Returns:
        torch.Tensor: The trained weight matrix W with requires_grad=True
        
    The model uses a simple linear layer followed by softmax to predict
    the next character. Training includes L2 regularization to prevent overfitting.
    Weights are initialized with a fixed seed for reproducibility.
    """
    xs, ys, _, _ = load_trigram_tensors()
    
    # Use torch.randn for better initialization
    generator = torch.Generator().manual_seed(2147483647)
    num_clases = VOCAB_SIZE * xs.ndim
    W = torch.randn((num_clases, VOCAB_SIZE), generator=generator, requires_grad=True)

    learning_rate = 50.0
    num_epochs = 800
    regularization_strength = 0.01
    
    for epoch in range(num_epochs):
        # Forward pass
        # print("xs.shape", xs.shape)
        xenc = F.one_hot(xs, num_classes=VOCAB_SIZE).float()
        xenc = xenc.view(xenc.shape[0], -1)
        # print("xenc.shape", xenc.shape)
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        
        # Loss with L2 regularization
        data_loss = -probs[torch.arange(xs.size(0)), ys].log().mean()
        reg_loss = regularization_strength * (W**2).mean()
        loss = data_loss + reg_loss
        
        # Backward pass
        W.grad = None
        loss.backward()
        
        # Update weights
        W.data -= learning_rate * W.grad
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, learning_rate: {learning_rate}")
            learning_rate = learning_rate * 0.9
    
    return W

def load_or_train_W():
    """
    Load pre-trained weights or train a new model if weights don't exist.
    
    Returns:
        torch.Tensor: The weight matrix W, either loaded from disk or freshly trained
        
    This provides a caching mechanism to avoid retraining the model every time.
    The weights are saved to W.pt for future use.
    """
    if not Path("W.pt").is_file():
        W = train_neural_net()
        torch.save(W, "W.pt")
    else:
        W = torch.load("W.pt")
    return W

def generate_with_neural_net(W, num_samples, seed=2147483647):
    """
    Generate names using the trained neural network model.
    
    Args:
        W (torch.Tensor): The trained weight matrix
        num_samples (int): Number of names to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        list: A list of generated names as strings
        
    Similar to the trigram generation, but uses the neural network's predictions
    for the next character probabilities at each step.
    """
    generator = torch.Generator().manual_seed(seed)
    num_clases =  W.shape[0]
    names = []
    for _ in range(num_samples):
        idx = [ctoi['.']] * W.ndim 
        chars = []
        while True:
            xenc = F.one_hot(torch.tensor([idx]), num_classes=VOCAB_SIZE).float()
            xenc = xenc.view(xenc.shape[0], -1)
            logits = xenc @ W
            counts = logits.exp()
            prob = counts / counts.sum(1, keepdim=True)
            
            new = torch.multinomial(prob, num_samples=1, replacement=True, generator=generator).item()
            idx = idx[1:] + [new]
            if new == ctoi['.']:  # End token
                break
            chars.append(itoc[new])
        names.append(''.join(chars))
    return names

def main():
    """
    Main function to demonstrate both trigram and neural network models.
    
    Generates names using both approaches and prints their performance metrics
    and sample outputs for comparison.
    """
    # trigram model
    P, loss = trigram_model()
    print(f"trigram model loss: {loss:.4f}")
    print("Generated names with trigram model:")
    trigram_names = generate_with_trigram(P, 10)
    for name in trigram_names:
        print(name)
    
    print("\n" + "==="*12 + "\n")
    
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

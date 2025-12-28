import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Character mapping configuration
START_TOKEN = '.'
END_TOKEN = '.'

# Build character to index and index to character mappings
chars = [START_TOKEN] + [chr(ord('a') + i) for i in range(26)]
ctoi = {ch: i for i, ch in enumerate(chars)}
itoc = {i: ch for i, ch in enumerate(chars)}
VOCAB_SIZE = len(chars)

def load_names(filename="names.txt"):
    """
    Load and preprocess names from names.txt file.
    
    Args:
        filename (str): Path to the names file (default: "names.txt")
        
    Returns:
        list: A list of names as strings, each split by whitespace from the file
        
    Raises:
        FileNotFoundError: If names.txt cannot be found in the expected locations
        
    The function tries to find names.txt in the current directory or in the MakeMore/
    subdirectory. Once found, it reads the file and splits the content by whitespace.
    """
    # Try different paths
    paths_to_try = [filename, f"MakeMore/{filename}"]
    for path in paths_to_try:
        if Path(path).exists():
            with open(path, 'r') as file:
                names = file.read().split()
            return names
    raise FileNotFoundError(f"Could not find {filename} in any of the expected locations")

def create_bigram_dataset(names):
    """
    Create a bigram dataset from a list of names.
    
    Args:
        names (list): List of names as strings
        
    Returns:
        tuple: (xs, ys) where both are torch.Tensor of integers
               xs contains indices of first characters in bigrams
               ys contains indices of second characters in bigrams
             
    Each name is surrounded with start and end tokens. For each adjacent pair
    of characters in the processed names, the first character index goes to xs
    and the second to ys. This creates the training data for the bigram model.
    """
    xs, ys = [], []
    for name in names:
        # Add start and end tokens
        name_tokens = START_TOKEN + name + END_TOKEN
        for c1, c2 in zip(name_tokens, name_tokens[1:]):
            xs.append(ctoi[c1])
            ys.append(ctoi[c2])
    return torch.tensor(xs), torch.tensor(ys)

def save_bigram_tensors(filename="names.txt"):
    """
    Generate and save bigram tensors to disk for faster loading.
    
    Args:
        filename (str): Path to the names file (default: "names.txt")
    
    This function processes the names, computes the bigram statistics,
    and saves the following tensors:
    - xs.pt: Input character indices
    - ys.pt: Target character indices
    - P.pt: Probability matrix of transitions between characters
    - N.pt: Count matrix of transitions between characters
    
    The count matrix N starts with ones for smoothing and is incremented
    by each observed bigram transition. P is the normalized probability matrix.
    """
    names = load_names(filename)
    xs, ys = create_bigram_dataset(names)
    
    # Create count matrix N
    N = torch.ones((VOCAB_SIZE, VOCAB_SIZE), dtype=torch.int32)
    for x, y in zip(xs, ys):
        N[x, y] += 1
    
    # Create probability matrix P
    P = N / N.sum(1, keepdim=True)
    
    # Save tensors
    torch.save(xs, "xs.pt")
    torch.save(ys, "ys.pt")
    torch.save(P, "P.pt")
    torch.save(N, "N.pt")

def load_bigram_tensors(filename="names.txt"):
    """
    Load precomputed bigram tensors from disk, generating them if missing.
    
    Args:
        filename (str): Path to the names file (default: "names.txt")
        
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
        save_bigram_tensors(filename)
    return (
        torch.load("xs.pt"),
        torch.load("ys.pt"),
        torch.load("P.pt"),
        torch.load("N.pt")
    )

def bigram_model(filename="names.txt"):
    """
    Create and evaluate a counting-based bigram model.
    
    Args:
        filename (str): Path to the names file (default: "names.txt")
        
    Returns:
        tuple: (P, loss) where:
            P (torch.Tensor): The probability matrix
            loss (float): The negative log likelihood loss on the dataset
            
    This model uses maximum likelihood estimation via counting to learn
    character transition probabilities. The loss is computed as the average
    negative log likelihood of the true transitions in the dataset.
    """
    xs, ys, P, _ = load_bigram_tensors(filename)
    
    # Calculate loss
    loss = -torch.log(P[xs, ys]).mean()
    return P, loss.item()

def generate_with_bigram(model, num_samples, seed=2147483647, filename="names.txt"):
    """
    Generate names using the bigram probability model.
    
    Args:
        model (torch.Tensor): Probability matrix P from bigram_model()
        num_samples (int): Number of names to generate
        seed (int): Random seed for reproducibility
        filename (str): Path to the names file (default: "names.txt")
        
    Returns:
        list: A list of generated names as strings
        
    The generation starts with the start token and samples characters
    according to the transition probabilities until an end token is generated.
    """
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

def train_neural_net(filename="names.txt"):
    """
    Train a neural network-based bigram model using gradient descent.
    
    Args:
        filename (str): Path to the names file (default: "names.txt")
        
    Returns:
        torch.Tensor: The trained weight matrix W with requires_grad=True
        
    The model uses a simple linear layer followed by softmax to predict
    the next character. Training includes L2 regularization to prevent overfitting.
    Weights are initialized with a fixed seed for reproducibility.
    """
    xs, ys, _, _ = load_bigram_tensors(filename)
    
    # Use torch.randn for better initialization
    generator = torch.Generator().manual_seed(2147483647)
    W = torch.randn((VOCAB_SIZE, VOCAB_SIZE), generator=generator, requires_grad=True)
    
    # More reasonable training parameters
    learning_rate = 50.0
    num_epochs = 200
    regularization_strength = 0.01
    
    for epoch in range(num_epochs):
        # Forward pass
        xenc = F.one_hot(xs, num_classes=VOCAB_SIZE).float()
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
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return W

def load_or_train_W(filename="names.txt"):
    """
    Load pre-trained weights or train a new model if weights don't exist.
    
    Args:
        filename (str): Path to the names file (default: "names.txt")
        
    Returns:
        torch.Tensor: The weight matrix W, either loaded from disk or freshly trained
        
    This provides a caching mechanism to avoid retraining the model every time.
    The weights are saved to W.pt for future use.
    """
    if not Path("W.pt").is_file():
        W = train_neural_net(filename)
        torch.save(W, "W.pt")
    else:
        W = torch.load("W.pt")
    return W

def generate_with_neural_net(W, num_samples, seed=2147483647, filename="names.txt"):
    """
    Generate names using the trained neural network model.
    
    Args:
        W (torch.Tensor): The trained weight matrix
        num_samples (int): Number of names to generate
        seed (int): Random seed for reproducibility
        filename (str): Path to the names file (default: "names.txt")
        
    Returns:
        list: A list of generated names as strings
        
    Similar to the bigram generation, but uses the neural network's predictions
    for the next character probabilities at each step.
    """
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

def main(filename="names.txt"):
    """
    Main function to demonstrate both bigram and neural network models.
    
    Args:
        filename (str): Path to the names file (default: "names.txt")
        
    Generates names using both approaches and prints their performance metrics
    and sample outputs for comparison.
    """
    # Bigram model
    P, loss = bigram_model(filename)
    print(f"Bigram model loss: {loss:.4f}")
    print("Generated names with bigram model:")
    bigram_names = generate_with_bigram(P, 10, filename=filename)
    for name in bigram_names:
        print(name)
    
    print("\n" + "="*50 + "\n")
    
    # Neural network model
    W = load_or_train_W(filename)
    print("Generated names with neural network:")
    nn_names = generate_with_neural_net(W, 10, filename=filename)
    for name in nn_names:
        print(name)

if __name__=="__main__":
    # Allow command line argument for filename
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()

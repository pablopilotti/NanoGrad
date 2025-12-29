import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import random

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
    """        
    xs, ys = [], []
    for name in names:
        # Add start and end tokens
        name_tokens = START_TOKEN + name + END_TOKEN
        for c1, c2 in zip(name_tokens, name_tokens[1:]):
            xs.append(ctoi[c1])
            ys.append(ctoi[c2])
    return torch.tensor(xs), torch.tensor(ys)

def create_trigram_dataset(names):
    """
    Create a trigram dataset from a list of names.
    
    Args:
        names (list): List of names as strings
        
    Returns:
        tuple: (xs, ys) where both are torch.Tensor of integers
               xs contains indices of first characters in trigrams
               ys contains indices of second characters in trigrams
    """        
    xs, ys = [], []
    for name in names:
        # Add start and end tokens
        name_tokens = START_TOKEN + name + END_TOKEN
        for c1, c2, c3 in zip(name_tokens, name_tokens[1:],name_tokens[2:]):
            xs.append((ctoi[c1],ctoi[c2]))
            ys.append(ctoi[c3])
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
    
    # Save tensors using specific names to avoid overwriting
    out_dir = Path("weights") # Default or from args
    out_dir.mkdir(exist_ok=True)
    torch.save(xs, out_dir / "xs_bigram.pt")
    torch.save(ys, out_dir / "ys_bigram.pt")
    torch.save(P, out_dir / "P_bigram.pt")
    torch.save(N, out_dir / "N_bigram.pt")

def save_trigram_tensors(filename="names.txt", save_dir="."):
    """
    Generate and save trigram tensors to disk for faster loading.
    
    Args:
        filename (str): Path to the names file (default: "names.txt")
        save_dir (str): Directory to save the tensors (default: ".")
    
    This function processes the names, computes the trigram statistics,
    and saves the following tensors:
    - xs.pt: Input character indices
    - ys.pt: Target character indices
    - P.pt: Probability matrix of transitions between characters
    - N.pt: Count matrix of transitions between characters
    
    The count matrix N starts with ones for smoothing and is incremented
    by each observed trigram transition. P is the normalized probability matrix.
    """
    names = load_names(filename)
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
    
    # Save tensors using specific names to avoid overwriting
    save_path = Path(save_dir)
    torch.save(xs, save_path / "xs_trigram.pt")
    torch.save(ys, save_path / "ys_trigram.pt")
    torch.save(P, save_path / "P_trigram.pt")
    torch.save(N, save_path / "N_trigram.pt")

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
    if not Path("weights/P_bigram.pt").is_file():
        save_bigram_tensors(filename)
    return (
        torch.load("weights/xs_bigram.pt"),
        torch.load("weights/ys_bigram.pt"),
        torch.load("weights/P_bigram.pt"),
        torch.load("weights/N_bigram.pt")
    )

def load_trigram_tensors(filename="names.txt"):
    """
    Load precomputed trigram tensors from disk, generating them if missing.
    
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
    if not Path("weights/P_trigram.pt").is_file():
        save_trigram_tensors(filename, "weights")
    return (
        torch.load("weights/xs_trigram.pt"),
        torch.load("weights/ys_trigram.pt"),
        torch.load("weights/P_trigram.pt"),
        torch.load("weights/N_trigram.pt")
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

def trigram_model(filename="names.txt"):
    """
    Create and evaluate a counting-based trigram model.
    
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
    xs, ys, P, _ = load_trigram_tensors(filename)
    
    # Calculate loss
    # 1. Split xs into its constituent columns (as a tuple)
    # 2. Append ys to that tuple
    indices = (*torch.unbind(xs, dim=1), ys) if xs.ndim > 1 else (xs, ys)

    # 3. Use the tuple to index P
    probs = P[indices]

    loss = -torch.log(probs).mean()    

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

def generate_with_trigram(model, num_samples, seed=2147483647, filename="names.txt"):
    """
    Generate names using the trigram probability model.
    
    Args:
        model (torch.Tensor): Probability matrix P from trigram_model()
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

def train_bigram_neural_net(filename="names.txt"):
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

def train_trigram_neural_net(filename="names.txt"):
    """
    Train a neural network-based trigram model using gradient descent.
    
    Args:
        filename (str): Path to the names file (default: "names.txt")
        
    Returns:
        torch.Tensor: The trained weight matrix W with requires_grad=True
        
    The model uses a simple linear layer followed by softmax to predict
    the next character. Training includes L2 regularization to prevent overfitting.
    Weights are initialized with a fixed seed for reproducibility.
    """
    xs, ys, _, _ = load_trigram_tensors(filename)
    
    # Use torch.randn for better initialization
    generator = torch.Generator().manual_seed(2147483647)
    num_classes = VOCAB_SIZE * 2 # For trigram, input is 2 characters
    W = torch.randn((num_classes, VOCAB_SIZE), generator=generator, requires_grad=True)

    learning_rate = 50.0
    num_epochs = 800
    regularization_strength = 0.01
    
    for epoch in range(num_epochs):
        # Forward pass
        xenc = F.one_hot(xs, num_classes=VOCAB_SIZE).float()
        xenc = xenc.view(xenc.shape[0], -1)

        logits = xenc @ W
        # do not change this part
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

def load_or_train_bigram_W(filename="names.txt"):
    """
    Load pre-trained weights or train a new model if weights don't exist.
    
    Args:
        filename (str): Path to the names file (default: "names.txt")
        
    Returns:
        torch.Tensor: The weight matrix W, either loaded from disk or freshly trained
        
    This provides a caching mechanism to avoid retraining the model every time.
    The weights are saved to W.pt for future use.
    """
    if not Path("weights/W_bigram.pt").is_file():
        W = train_bigram_neural_net(filename)
        Path("weights").mkdir(exist_ok=True)
        torch.save(W, "weights/W_bigram.pt")
    else:
        W = torch.load("weights/W_bigram.pt")
    return W

def load_or_train_trigram_W(filename="names.txt"):
    """
    Load pre-trained weights or train a new model if weights don't exist.
    
    Args:
        filename (str): Path to the names file (default: "names.txt")
        
    Returns:
        torch.Tensor: The weight matrix W, either loaded from disk or freshly trained
        
    This provides a caching mechanism to avoid retraining the model every time.
    The weights are saved to W.pt for future use.
    """
    if not Path("weights/W_trigram.pt").is_file():
        W = train_trigram_neural_net(filename)
        Path("weights").mkdir(exist_ok=True)
        torch.save(W, "weights/W_trigram.pt")
    else:
        W = torch.load("weights/W_trigram.pt")
    return W

def generate_with_bigram_neural_net(W, num_samples, seed=2147483647, filename="names.txt"):
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

def generate_with_trigram_neural_net(W, num_samples, seed=2147483647, filename="names.txt"):
    """
    Generate names using the trained neural network model.
    
    Args:
        W (torch.Tensor): The trained weight matrix
        num_samples (int): Number of names to generate
        seed (int): Random seed for reproducibility
        filename (str): Path to the names file (default: "names.txt")
        
    Returns:
        list: A list of generated names as strings
        
    Similar to the trigram generation, but uses the neural network's predictions
    for the next character probabilities at each step.
    """
    generator = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(num_samples):
        # W is a 2D matrix [54, 27]. W.ndim is 2. 
        # Trigram context needs exactly 2 characters.
        idx = [ctoi['.'], ctoi['.']] 
        chars = []
        while True:
            # Must be [1, 2] shape for the linear layer (batch of 1)
            xenc = F.one_hot(torch.tensor([idx]), num_classes=VOCAB_SIZE).float()
            xenc = xenc.view(xenc.shape[0], -1)
            logits = xenc @ W
            # do not change this part
            counts = logits.exp()
            prob = counts / counts.sum(1, keepdim=True)
            
            new = torch.multinomial(prob, num_samples=1, replacement=True, generator=generator).item()
            idx = idx[1:] + [new]
            if new == ctoi['.']:  # End token
                break
            chars.append(itoc[new])
        names.append(''.join(chars))
    return names

def split_data(names):
    """Splits names into 80% train, 10% dev, and 10% test."""
    random.seed(42)
    random.shuffle(names)
    n1 = int(0.8 * len(names))
    n2 = int(0.9 * len(names))
    return names[:n1], names[n1:n2], names[n2:]

def get_trigram_counts(names):
    """Calculates raw counts for all trigrams in the provided list of names."""
    N = torch.zeros((VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE), dtype=torch.int32)
    for name in names:
        tokens = '.' + '.' + name + '.'
        for c1, c2, c3 in zip(tokens, tokens[1:], tokens[2:]):
            N[ctoi[c1], ctoi[c2], ctoi[c3]] += 1
    return N

def eval_trigram_loss(counts, smoothing, eval_names):
    """Computes NLL loss for a name set given a count matrix and smoothing strength."""
    P = (counts.float() + smoothing)
    P /= P.sum(2, keepdim=True)
    
    loss, n = 0, 0
    for name in eval_names:
        tokens = '.' + '.' + name + '.'
        for c1, c2, c3 in zip(tokens, tokens[1:], tokens[2:]):
            prob = P[ctoi[c1], ctoi[ctoi[c2]], ctoi[c3]]
            loss += -torch.log(prob)
            n += 1
    return (loss / n).item()

def evaluate_model(model_type, model, test_file, filename="names.txt"):
    """
    Evaluate a model on a test dataset.
    
    Args:
        model_type (str): Type of model ('bigram', 'trigram', 'bigram_nn', 'trigram_nn')
        model (torch.Tensor): The trained model
        test_file (str): Path to the test file
        filename (str): Path to the names file (default: "names.txt")
        
    Returns:
        float: The loss value
    """
    if model_type == 'bigram':
        _, loss = bigram_model(filename)
        return loss
    elif model_type == 'trigram':
        _, loss = trigram_model(filename)
        return loss
    elif model_type == 'bigram_nn':
        xs, ys = create_bigram_dataset(load_names(test_file))
        xenc = F.one_hot(xs, num_classes=VOCAB_SIZE).float()
        logits = xenc @ model
        probs = F.softmax(logits, dim=1)
        loss = -probs[torch.arange(xs.size(0)), ys].log().mean()
        return loss.item()
    elif model_type == 'trigram_nn':
        xs, ys = create_trigram_dataset(load_names(test_file))
        xenc = F.one_hot(xs, num_classes=VOCAB_SIZE).float().view(xs.shape[0], -1)
        logits = xenc @ model
        probs = F.softmax(logits, dim=1)
        loss = -probs[torch.arange(xs.size(0)), ys].log().mean()
        return loss.item()
    return 0.0

def main():
    parser = argparse.ArgumentParser(description="Makemore CLI: Train, Generate, and Evaluate n-gram models.")

    # 1. Model Selection Group (Mutually Exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--bigram', choices=['NN', 'table'], help="Use a bigram model (Neural Net or Lookup Table)")
    model_group.add_argument('--trigram', choices=['NN', 'table'], help="Use a trigram model (Neural Net or Lookup Table)")
    model_group.add_argument('--all', action='store_true', help="Operate on all available model types")

    # 2. Action Selection Group (Mutually Exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--train', metavar='PATH', help="Train models and save to this folder")
    action_group.add_argument('--generate', metavar='PATH', help="Generate names using models from this folder")
    action_group.add_argument('--evaluate', metavar='PATH', help="Evaluate models in this folder")

    # 3. Supplemental Arguments
    parser.add_argument('--file', type=str, help="Path to the .txt file for training or evaluation")

    args = parser.parse_args()

    # --- Logic Routing ---

    if args.train:
        if not args.file:
            parser.error("--train requires --file [dataset.txt]")
        print(f"Training {args.bigram or args.trigram or 'all'} model(s) using {args.file}..")
        
        if args.bigram:
            if args.bigram == 'table':
                save_bigram_tensors(args.file)
                print("Bigram table model trained and saved.")
            else:  # NN
                W = train_bigram_neural_net(args.file)
                Path(args.train).mkdir(exist_ok=True)
                torch.save(W, f"{args.train}/W_bigram.pt")
                print("Bigram neural net model trained and saved.")
                
        elif args.trigram:
            if args.trigram == 'table':
                save_trigram_tensors(args.file, "weights")
                print("Trigram table model trained and saved.")
            else:  # NN
                W = train_trigram_neural_net(args.file)
                Path(args.train).mkdir(exist_ok=True)
                torch.save(W, f"{args.train}/W_trigram.pt")
                print("Trigram neural net model trained and saved.")
                
        elif args.all:
            save_bigram_tensors(args.file)
            save_trigram_tensors(args.file, "weights")
            W1 = train_bigram_neural_net(args.file)
            W2 = train_trigram_neural_net(args.file)
            Path(args.train).mkdir(exist_ok=True)
            torch.save(W1, f"{args.train}/W_bigram.pt")
            torch.save(W2, f"{args.train}/W_trigram.pt")
            print("All models trained and saved.")
        
    elif args.generate:
        print(f"Generating sequences using {args.bigram or args.trigram or 'all'} model(s) from {args.generate}..")
        
        if args.bigram:
            if args.bigram == 'table':
                P, _ = bigram_model(args.file)
                names = generate_with_bigram(P, 10, filename=args.file)
                for name in names:
                    print(name)
            else:  # NN
                W = torch.load(f"{args.generate}/W_bigram.pt")
                names = generate_with_bigram_neural_net(W, 10, filename=args.file)
                for name in names:
                    print(name)
                    
        elif args.trigram:
            if args.trigram == 'table':
                P, _ = trigram_model(args.file)
                names = generate_with_trigram(P, 10, filename=args.file)
                for name in names:
                    print(name)
            else:  # NN
                W = torch.load(f"{args.generate}/W_trigram.pt")
                names = generate_with_trigram_neural_net(W, 10, filename=args.file)
                for name in names:
                    print(name)
                    
        elif args.all:
            # Generate with both models
            print("Bigram table:")
            P, _ = bigram_model(args.file)
            names = generate_with_bigram(P, 5, filename=args.file)
            for name in names:
                print(name)
                
            print("\nTrigram table:")
            P, _ = trigram_model(args.file)
            names = generate_with_trigram(P, 5, filename=args.file)
            for name in names:
                print(name)
                
            print("\nBigram neural net:")
            W = torch.load(f"{args.generate}/W_bigram.pt")
            names = generate_with_bigram_neural_net(W, 5, filename=args.file)
            for name in names:
                print(name)
                
            print("\nTrigram neural net:")
            W = torch.load(f"{args.generate}/W_trigram.pt")
            names = generate_with_trigram_neural_net(W, 5, filename=args.file)
            for name in names:
                print(name)
        
    elif args.evaluate:
        if not args.file:
            parser.error("--evaluate requires --file [test_set.txt]")
        print(f"Evaluating {args.bigram or args.trigram or 'all'} model(s) against {args.file}..")
        
        if args.bigram:
            if args.bigram == 'table':
                P, loss = bigram_model(args.file)
                print(f"Bigram table model loss: {loss:.4f}")
            else:  # NN
                W = torch.load(f"{args.evaluate}/W_bigram.pt")
                loss = evaluate_model('bigram_nn', W, args.file, args.file)
                print(f"Bigram neural net model loss: {loss:.4f}")
                
        elif args.trigram:
            if args.trigram == 'table':
                # E03 logic: Hyperparameter tuning
                names = load_names(args.file)
                train_n, dev_n, test_n = split_data(names)
                N_train = get_trigram_counts(train_n)
                
                best_s, best_loss = None, float('inf')
                # Try a range of smoothing possibilities (orders of magnitude)
                for s in [0.001, 0.01, 0.1, 1.0, 10.0]:
                    t_loss = eval_trigram_loss(N_train, s, train_n)
                    d_loss = eval_trigram_loss(N_train, s, dev_n)
                    print(f"Smoothing: {s:6.3f} | Train Loss: {t_loss:.4f} | Dev Loss: {d_loss:.4f}")
                    
                    if d_loss < best_loss:
                        best_loss = d_loss
                        best_s = s
                
                # Evaluate once on test set with best s
                final_test_loss = eval_trigram_loss(N_train, best_s, test_n)
                print(f"\nBest smoothing: {best_s}")
                print(f"Final Test Loss: {final_test_loss:.4f}")
            else:  # NN
                W = torch.load(f"{args.evaluate}/W_trigram.pt")
                loss = evaluate_model('trigram_nn', W, args.file, args.file)
                print(f"Trigram neural net model loss: {loss:.4f}")
                
        elif args.all:
            P, loss = bigram_model(args.file)
            print(f"Bigram table model loss: {loss:.4f}")
            
            P, loss = trigram_model(args.file)
            print(f"Trigram table model loss: {loss:.4f}")
            
            W = torch.load(f"{args.evaluate}/W_bigram.pt")
            loss = evaluate_model('bigram_nn', W, args.file, args.file)
            print(f"Bigram neural net model loss: {loss:.4f}")
            
            W = torch.load(f"{args.evaluate}/W_trigram.pt")
            loss = evaluate_model('trigram_nn', W, args.file, args.file)
            print(f"Trigram neural net model loss: {loss:.4f}")

if __name__ == "__main__":
    main()

# Exercises:
# E01: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. 
#      Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model?
# E02: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. 
#      Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?
#
# $ python3 MakeMore.py --all --evaluate ./all-names --file names_test.txt
# Evaluating all model(s) against names_test.txt..
# Bigram table model loss: 2.4534
# Trigram table model loss: 2.0934
# Bigram neural net model loss: 2.4822
# Trigram neural net model loss: 2.2648

# $ python3 MakeMore.py --all --evaluate ./all-names --file names_dev.txt
# Evaluating all model(s) against names_dev.txt..
# Bigram table model loss: 2.4534
# Trigram table model loss: 2.0934
# Bigram neural net model loss: 2.4602
# Trigram neural net model loss: 2.2458

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

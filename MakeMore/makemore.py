
# read names.txt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path



ctoi = {'.': 0}
itoc = {0: '.'}
for i in range(ord('z')-ord('a')+1):
    ctoi[chr(ord('a') + i)] = i +1
    itoc[i+1] = chr(ord('a') + i)

def save_bigram_tensors():
    with open('names.txt','r') as file:
        names = file.read().split()
    
    N = torch.ones((len(itoc),len(itoc)), dtype=torch.int32)
    xs = []
    ys = []
    for name in names:
        n = '.' + name + '.'
        for c1, c2 in zip(n,n[1:]):
            n1 = ctoi[c1]
            n2 = ctoi[c2]
            N[n1, n2] += 1
            xs.append(n1)
            ys.append(n2)
            
    P = N / N.sum(1,keepdim=True)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    torch.save(xs, "xs.pt")
    torch.save(ys, "ys.pt")
    torch.save(P, "P.pt")
    torch.save(N, "N.pt")

def load_bigram_tensors():
    if not Path("xs.pt").is_file():
        save_bigram_tensors()

    xs = torch.load("xs.pt")
    ys = torch.load("ys.pt")
    P = torch.load("P.pt")
    N = torch.load("N.pt")
    return xs, ys, P, N

def bigram(cant):
    xs, ys, P, _ = load_bigram_tensors()
 
    print(xs[0].item())
    loss = torch.tensor([-torch.log(P[x.item(),y.item()]) for x, y in zip(xs,ys)]).mean()
    print('loss: ', loss)
    print('table')
    print()
    g_cpu = torch.Generator().manual_seed(2147483647)
    for _ in range(cant):
        select = 0
        word = '' 
        while True:
            prob = P[select]
            select = torch.multinomial(prob, num_samples=1, replacement=True, generator=g_cpu).item()
            word += itoc[select]
            if select == 0:
                break
        print(word)

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

def NN(cant):
    W = load_W()
    print()
    print('NN')
    
    g_cpu = torch.Generator().manual_seed(2147483647)
    for i in range(cant):
        select = 0
        word = '' 
        while True:
            xenc = F.one_hot(torch.tensor([select]), num_classes = len(itoc)).float()
            logits = xenc @ W
            counts = logits.exp()
            prob = counts / counts.sum(1,keepdim=True)
            
            select = torch.multinomial(prob, num_samples=1, replacement=True, generator=g_cpu).item()
            word += itoc[select]
            if select == 0:
                break
        print(word)

def main():
    bigram(10)
    NN(10)

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
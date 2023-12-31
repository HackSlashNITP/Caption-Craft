import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import tqdm

class SinglePass(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_loader = None
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self._model_state = None
        self._train_state = None
        self.device = device
        self.clip_grad_norm = None
        self.metrics = {"train": {}, "test": {}}
        self.using_tpu = False
        self.fp16 = None

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128*7*7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def fetch_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)

    def fetch_scheduler(self):
        return CosineAnnealingLR(self.fetch_optimizer(), T_max=10)

    def train_one_step(self, data, target, optimizer, scheduler, device):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = self(data)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss

    def train_one_epoch(self, data_loader, optimizer, scheduler, device):
        self.train()
        epoch_loss = 0
        progress_bar = tqdm(data_loader, desc='Training', total=len(data_loader))
        for data, target in progress_bar:
            loss = self.train_one_step(data, target, optimizer, scheduler, device)
            epoch_loss += loss.item()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/data.size(0))})
        train_loss = epoch_loss / len(data_loader)
        print(f'Train Loss: {train_loss:.4f}')
        return train_loss

    def test(self, test_dataset, batch_size, device):
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        self.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = self(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

    def fit(self, train_dataset, test_dataset, batch_size, epochs, device):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = self.fetch_optimizer()
        scheduler = self.fetch_scheduler()

        if next(self.parameters()).device != device:
            self.to(device)

        for epoch in range(epochs):
            print(f'Epoch: {epoch+1}/{epochs}')
            train_loss = self.train_one_epoch(train_loader, optimizer, scheduler, device)
            self.metrics["train"][f"epoch_{epoch+1}"] = train_loss

            # Test after each epoch
            self.test(test_dataset, batch_size, device)

            scheduler.step()

from torchvision import datasets, transforms
import torch
from tqdm import tqdm

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

print("Downloading and loading the training data...")
train_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)

test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Initializing the model...")
model = SinglePass()

print("Starting training...")
model.fit(train_dataset, test_dataset , batch_size=32, epochs=10, device=device)
print("Training completed.")

# simple implementation

import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the model
def train_model(model, train_loader, test_loader, optimizer, scheduler, device, epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}/{epochs}')
        model.train()
        epoch_loss = 0

        for data, target in tqdm(train_loader, desc='Training'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        print(f'Train Loss: {train_loss:.4f}')

        test_model(model, test_loader, device)

        scheduler.step()

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model initialization and training
model = SimpleNet()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Starting training...")
train_model(model, train_loader, test_loader, optimizer, scheduler, device, epochs=10)
print("Training completed.")

"""# **Langauge Model Test**"""

!pip install nltk
import nltk

import torch
import torch.nn as nn
from torch.nn import functional as F
from nltk import word_tokenize
nltk.download('punkt')

# hyperparameters
batch_size = 32
block_size = 64
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_head = 32
dropout = 0.0

torch.manual_seed(1337)

with open('sentences.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# tokenize the text into words
words = word_tokenize(text)

# here are all the unique words that occur in this text
vocab = sorted(list(set(words)))
vocab_size = len(vocab)
# create a mapping from words to integers
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
encode = lambda s: [stoi[c] for c in word_tokenize(s)] # encoder: take a string, output a list of integers
decode = lambda l: ' '.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

dropout_rate=0.15
bidirectional=True
n_embd = 64
n_layer = 8
words = word_tokenize(text)
vocab = sorted(list(set(words)))
vocab_size = len(vocab)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, dropout_rate=0.5, bidirectional=False):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, n_embd, num_layers=n_layer, batch_first=True, bidirectional=bidirectional, dropout=dropout_rate if n_layer > 1 else 0)
        self.lstm1 = nn.LSTM(n_embd * (2 if bidirectional else 1), n_embd, num_layers=n_layer, batch_first=True, bidirectional=bidirectional, dropout=dropout_rate if n_layer > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.lm_head = nn.Linear(n_embd * (2 if bidirectional else 1), vocab_size)

        # Weight initialization
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        nn.init.xavier_normal_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, idx, targets=None, hidden=None):
      B, T = idx.shape
      tok_emb = self.token_embedding_table(idx) # (B,T,C)

      # Pass the input through each LSTM layer
      x, hidden = self.lstm(tok_emb, hidden) # (B,T,C)
      x, hidden = self.lstm1(x, hidden) # (B,T,C)
      x = self.dropout(x)
      logits = self.lm_head(x) # (B,T,vocab_size)

      if targets is not None:
          B, T, C = logits.shape
          logits_flatten = logits.view(B*T, C)
          targets_flatten = targets.view(B*T)
          loss = F.cross_entropy(logits_flatten, targets_flatten)
      else:
          loss = None

      return logits, loss, hidden

    def generate(self, idx, max_new_tokens, hidden=None):
        for _ in range(max_new_tokens):
            tok_emb = self.token_embedding_table(idx) # (B,T,C)
            x, hidden = self.lstm(tok_emb, hidden) # (B,T,C)
            x = self.dropout(x)
            logits = self.lm_head(x) # (B,T,vocab_size)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx, hidden

model = LSTMModel(vocab_size, n_embd, n_layer, dropout_rate, bidirectional)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
from torch.optim.lr_scheduler import StepLR
import numpy as np

# Initialize the learning rate scheduler
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

# Initialize the best validation loss
best_val_loss = np.inf

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Save the model if the validation loss improved
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), 'best_model.pth')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss, _ = model(xb, yb, hidden=None)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Clip the gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Step the learning rate scheduler
    scheduler.step()

import random

numbers = list(range(1, 6))
random.shuffle(numbers)
random_number = random.choice(numbers)
print(numbers)
for i in range(5):
  context = torch.zeros((1, 1), dtype=torch.long, device=device)
  generated = m.generate(context, max_new_tokens= (10 + random_number*5))[0].tolist()
  generated = [item for sublist in generated for item in sublist]
  print(f"{i+1}. {decode(generated)}")
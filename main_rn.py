import os
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from collections import Counter
import pandas as pd
from GRU import GRU
from LSTM import LSTM
from BERT import BERT
from torch.utils.data import DataLoader
import spacy
from torchtext.datasets import IMDB, SST
from torchtext.data import Field, LabelField, BucketIterator
import torchtext
pd.options.display.max_columns = 100
pd.options.display.width = 1000

# os.system('python -m spacy download en')
#SRC = Field(tokenize='spacy', lower=True)
SRC = Field()
TRG = LabelField(dtype=torch.int64)

sp = spacy.load('en')
BATCH_SIZE = 300


train_data, test_data = SST.splits(SRC, TRG, validation=None)  # download dataset
#train_data, test_data = IMDB.splits(SRC, TRG)  # download dataset
train_data, val_data = train_data.split(0.8, (torch.Generator().manual_seed(10), ))


print(f'Train data, n={len(train_data)}')
print(pd.Series([x.label for x in train_data.examples]).value_counts(normalize=True))
print(f'Val data, n={len(val_data)}')
print(pd.Series([x.label for x in val_data.examples]).value_counts(normalize=True))
print(f'Test data, n={len(test_data)}')
print(pd.Series([x.label for x in test_data.examples]).value_counts(normalize=True))

# display lenght of test and traing data
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

print(vars(train_data.examples[0]))

# Build vocabulary for source and target from training data

SRC.build_vocab(train_data, max_size=10000, min_freq=5, vectors="glove.6B.100d")  # using pretrained word embedding
TRG.build_vocab(train_data, min_freq=5)

print(vars(TRG.vocab))
print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in TRG vocabulary: {len(TRG.vocab)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# train and test iteartor
train_iterator, val_iterator, test_iterator = BucketIterator.splits(
      (train_data, val_data, test_data),
      batch_size=BATCH_SIZE,
      device=device
    )

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 100
DEC_EMB_DIM = 100
HID_DIM = 256
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# initializing our model
model = GRU(INPUT_DIM, OUTPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)

# loading pretrained word embedding
model.embedding.weight.data.copy_(SRC.vocab.vectors)

optimizer = optim.Adam(model.parameters(), lr=3e-3)

# defining learnig rate scheduler (optional)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

criterion = nn.CrossEntropyLoss()


# Model training function
def train(model, iterator, optimizer=optimizer, criterion=criterion, clip=1):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.text.to(device)
        trg = batch.label.to(device)
        optimizer.zero_grad()
        output = model(src)

        loss = criterion(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    mean_loss = epoch_loss / len(iterator)
    scheduler.step(mean_loss)
    return mean_loss  # mean loss


def check_accuracy(data_iterator, model):
    model.eval()
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for i, batch in enumerate(data_iterator):
            src = batch.text.to(device)
            trg = batch.label.to(device)
            output = model(src)

            total_correct += torch.sum(torch.eq(output.argmax(1), trg))
            total_count += len(trg)

    return f'{total_correct}/{total_count}', round(float(total_correct/total_count), 3)


df = pd.DataFrame()

# loop and train our model
total_epoch = 40
for epoch in range(total_epoch):
    result = train(model=model, iterator=train_iterator)
    print(f'Epoch {epoch} -->', result)
    train_check = check_accuracy(train_iterator, model)
    val_check = check_accuracy(val_iterator, model)
    test_check = check_accuracy(test_iterator, model)
    s = pd.Series({'train_accuracy_str': train_check[0],
                   'val_accuracy_str': val_check[0],
                   'test_accuracy_str': test_check[0],
                   'train_accuracy': train_check[1],
                   'val_accuracy': val_check[1],
                   'test_accuracy': test_check[1],
                   }, name=epoch)
    df = df.append(s)[s.index.tolist()]
    torch.save(model.state_dict(), os.path.join('model_snapshots', f'{epoch}.pth'))
    print(df)

best_epoch = df['val_accuracy'].idxmax()
print(df.loc[best_epoch])


def do_prediction(sentence):

    if type(sentence) == str:
        tokanised_sentence = [word.text for word in sp.tokenizer(sentence)]
    else:
        tokanised_sentence = sentence

    input_data = [SRC.vocab.stoi[word.lower()] for word in tokanised_sentence]
    input_data = torch.tensor(input_data, dtype=torch.int64).unsqueeze(1).to(device)

    model.eval()
    output = model(input_data)
    label_mapping = train_data.fields['label'].vocab.stoi
    r = {'text': sentence, **{k: v for k, v in zip(sorted(label_mapping, key=lambda x: label_mapping[x]), output[0].tolist())}}
    return r


print(f'Loading model on epoch={best_epoch}')
model.load_state_dict(torch.load(f'model_snapshots/{best_epoch}.pth'))

for e in range(total_epoch):
    os.remove(f'model_snapshots/{e}.pth')

to_predict = [
    'That was great!',
    'That was very bad!',
    'It is wonderful film. I like it!',
    "Terrible, stupid, tasteless! Worst thing I've ever seen",
    "Don't know, what to say.",
]

print(pd.DataFrame([do_prediction(p) for p in to_predict]))

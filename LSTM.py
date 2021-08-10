import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        # input_dim <--- vocabulary size
        # output_dim <--- len ([positive, negative]) == 2
        # emb_dim <--- embedding dimension of embedding matrix

        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # shape: [source_len, batch_size]
        embedded = self.dropout(self.embedding(src))  # sahpe: [src_len, batch_size, embed_dim]
        output, (hidden, cell) = self.rnn(embedded)
        # output[batch, hidden_dim]
        # hiddden[n_layers, batch, hidden_dim]
        output = self.fc1(output[-1])
        output = self.fc2(self.relu(output))
        return output

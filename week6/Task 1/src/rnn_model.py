"""
rnn_model.py

Defines recurrent architectures for univariate time series forecasting:
a vanilla (Elman) RNN, an LSTM, and a GRU — all sharing the same interface
so they can be trained and compared under identical conditions.
"""

import torch
import torch.nn as nn


class SequenceForecaster(nn.Module):
    """
    A single recurrent layer (RNN / LSTM / GRU, chosen via `cell_type`)
    followed by a dense output layer that maps the final hidden state to
    a single scalar forecast (next time step's value).

    Input shape:  (batch, seq_len, input_size)
    Output shape: (batch, 1)
    """

    def __init__(self, cell_type: str = "lstm", input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        cell_type = cell_type.lower()
        self.cell_type = cell_type

        rnn_kwargs = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if cell_type == "rnn":
            self.rnn = nn.RNN(**rnn_kwargs, nonlinearity="tanh")
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif cell_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}. Use 'rnn', 'lstm', or 'gru'.")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out: (batch, seq_len, hidden_size); we only need the last time step's hidden state
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]           # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)           # (batch, 1)


def build_model(cell_type: str = "lstm", hidden_size: int = 64, num_layers: int = 2) -> SequenceForecaster:
    """Convenience factory with sensible defaults."""
    return SequenceForecaster(
        cell_type=cell_type,
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.2,
    )

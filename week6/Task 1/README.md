# RNN/LSTM/GRU Time Series Forecaster — Daily Minimum Temperatures

A recurrent neural network project forecasting next-day minimum
temperature from a 30-day lookback window, comparing three recurrent
architectures — a vanilla (Elman) RNN, an LSTM, and a GRU — trained and
evaluated under identical conditions.

> **Note on dataset choice:** The task allowed either IMDB sentiment
> analysis or stock-price forecasting. IMDB's standard hosts weren't
> reachable from this project's network environment, so this project uses
> **time series forecasting** instead, on the classic Daily Minimum
> Temperatures (Melbourne, Australia, 1981–1990) dataset — 3,650 daily
> readings, sourced from a public GitHub-hosted CSV mirror. The
> preprocessing (windowing, scaling) and modeling approach (LSTM/GRU vs.
> vanilla RNN) apply identically to stock price data.

## Sequential Dataset

- **Source:** [Daily Minimum Temperatures, Melbourne](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv) — 3,650 daily minimum temperature readings (°C), 1981–1990.
- **Task:** univariate time series forecasting — predict the next day's minimum temperature from the preceding 30 days.
- **Split:** chronological (not shuffled) — earliest ~70% for training, next ~15% for validation, most recent ~15% for testing — so no future information leaks into training.

## Preprocessing

1. **Scaling:** `MinMaxScaler` fit only on the training split, applied to all splits (avoids test-set leakage).
2. **Windowing:** sliding windows of 30 consecutive days as input (`X`), with the 31st day as the target (`y`). Validation/test windows include the preceding 30 days of context so no data is wasted at split boundaries.
3. **Tensors:** each window reshaped to `(batch, 30, 1)` — sequence length 30, 1 feature (temperature) — the standard PyTorch RNN input format.

## Model Architecture

All three models share the same wrapper (`SequenceForecaster` in `src/rnn_model.py`) and hyperparameters, differing only in the recurrent cell:

| Component | Setting |
|---|---|
| Recurrent cell | `nn.RNN` / `nn.LSTM` / `nn.GRU` (swappable) |
| Hidden size | 64 |
| Recurrent layers | 2 (stacked) |
| Dropout | 0.2 (between recurrent layers + before output layer) |
| Output | `Linear(64 → 1)` on the final time step's hidden state |

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Max epochs | 60 (early stopping patience 10) |
| Learning rate | 0.001 (Adam), halved on validation plateau |
| Loss | MSE (on scaled values) |
| Gradient clipping | max norm 1.0 |
| Window size | 30 days |
| Random seed | 42 |

## Results (test set, original °C scale)

| Model | RMSE (°C) | MAE (°C) | Epochs run | Final avg. grad norm |
|---|---|---|---|---|
| Vanilla RNN | 2.235 | 1.743 | 33 | 0.128 |
| LSTM | 2.237 | 1.742 | 29 | 0.077 |
| **GRU** | **2.230** | **1.738** | 29 | 0.072 |

All three models converge to very similar accuracy on this task — see
[`docs/performance_evaluation.md`](docs/performance_evaluation.md) for a full
discussion of *why* (short 30-day window, gradient clipping, single-step
forecasting), and where the gap between vanilla RNNs and gated architectures
(LSTM/GRU) would be expected to widen (longer sequences, harder long-range
dependencies).

- [`docs/training_history.png`](docs/training_history.png) — training/validation loss curves and per-epoch gradient norms for all three models
- [`docs/prediction_comparison.png`](docs/prediction_comparison.png) — actual vs. predicted temperature on the test set

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train and evaluate all three models (downloads the dataset if needed, trains RNN/LSTM/GRU, evaluates on the test set, saves all artifacts below):

```bash
python src/train.py
```

Outputs:
- `models/saved_rnn_model.pth`, `models/saved_lstm_model.pth`, `models/saved_gru_model.pth`
- `docs/training_metrics.json` — full per-epoch history (loss + gradient norms) and final test metrics for all three models
- `docs/training_history.png` — loss curves + gradient norm plot
- `docs/prediction_comparison.png` — actual vs. predicted plot on the test set

## Project Structure

```
README.md
requirements.txt
src/
  preprocess.py   # data loading, scaling, windowing
  rnn_model.py    # shared RNN/LSTM/GRU architecture definition
  train.py        # training loop for all 3 models, evaluation, plots
docs/
  performance_evaluation.md   # written comparison of LSTM vs. RNN limitations
  training_metrics.json       # generated after running train.py
  training_history.png
  prediction_comparison.png
models/
  saved_rnn_model.pth   # generated after running train.py
  saved_lstm_model.pth
  saved_gru_model.pth
```

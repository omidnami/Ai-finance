# AI Finance

**AI Finance** is an open-source experiment in building a small financial neural network **from scratch**, designed to train and run on a CPU. The project's initial goal is educational: understand, implement, train, and evaluate a custom model rather than reuse pretrained model weights.

> **Status:** Planning / initial design (v0.1). Model training and prediction commands are **not implemented yet**. This README documents the intended first version; it does not claim that a working predictor is available.

## Project goals

- Build a custom multilayer perceptron (MLP) with **Python and NumPy**, including forward propagation, backpropagation, loss, and optimization.
- Train with real MetaTrader OHLCV data exported to Excel.
- Calculate technical indicators through the Python [`ta` library](https://technical-analysis-library-in-python.readthedocs.io/en/latest/).
- Use **the same feature-processing pipeline** for training, validation, testing, and inference.
- Keep the model lightweight, CPU-friendly, reproducible, and extensible.
- Evaluate predictions on strictly chronological holdout data; avoid look-ahead bias and data leakage.

## Proposed v0.1 model

| Setting | Initial design |
| --- | --- |
| Market data | Real 15-minute OHLCV candles |
| Lookback window | 20 completed candles |
| Input shape | 20 candles × 24 features = 480 inputs |
| Neural network | `480 → 128 → 64 → 32 → 1` |
| Hidden activation | ReLU |
| Output activation | Sigmoid |
| Target | Direction of the next candle's close relative to the latest observed close |
| Loss | Binary cross-entropy |
| Optimizer | AdamW (custom implementation) |
| Compute | CPU; no GPU required |

**The 24-feature schema is provisional** and will be finalized before implementation. Candidate features include OHLCV, MACD (12, 26, 9), RSI (14), moving averages (20, 50), and time/session information. Feature definitions, normalization, and indicator settings will be versioned.

## Input data

The expected MetaTrader export contains these columns:

| Column | Meaning |
| --- | --- |
| `time` | Unix timestamp (seconds) |
| `open`, `high`, `low`, `close` | OHLC prices |
| `volume` | Market volume / tick volume depending on source |
| `timeframe` | Candle interval in minutes (`15` for v0.1) |

Market closures, holidays, and session breaks can produce legitimate timestamp gaps. The data pipeline will **not fabricate missing candles**; it will preserve and validate real market chronology. Timestamps and session/time-zone handling will be checked against the data source.

## Getting started

### Prerequisites (planned)

- Python 3.12+
- NumPy
- Pandas and an Excel reader (e.g. openpyxl)
- `ta` for technical indicators
- Matplotlib for training diagnostics
- pytest for tests

### Clone the repository

```bash
git clone https://github.com/omidnami/Ai-finance.git
cd Ai-finance
```

### Prepare a local Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate using `.venv\Scripts\activate`.

**Next steps:** The dependency manifest, data-import command, model training entry point, and prediction CLI will be added as implementation begins. Do not treat these setup steps as a working training guide yet.

## Planned training workflow

1. Import historical 15-minute candles from Excel.
2. Validate columns, ordering, candle completeness, and market-session gaps.
3. Derive MACD, RSI, moving averages, and calendar/session features through one shared processing service.
4. Build rolling 20-candle samples and next-candle direction labels.
5. Split data **chronologically** into training, validation, and test periods.
6. Fit normalization parameters on the training split only, then reuse them without refitting.
7. Train the custom NumPy network, track loss and metrics, and save versioned model checkpoints.
8. Test on unseen periods before using the same feature pipeline for inference.

**Disclaimer:** This project is for research and education, not financial advice or a guarantee of profitable trading.

## Support websites

- **Omid Nami:** https://omidnami.com
- **TurnX:** https://turnx.ir

## License

Apache License 2.0. See `LICENSE` when included in the repository.

# Project Documentation

**Author**: SimicX AI Quant  
**Copyright**: (C) 2025-2026 SimicX. All rights reserved.  
**Generated**: 2026-01-04 20:29

## Overview

This project is a comprehensive quantitative trading framework designed for the systematic development, optimization, and backtesting of alpha-generating strategies. The system facilitates a rigorous alpha engineering pipeline, transitioning from high-dimensional hyperparameter tuning on historical data to out-of-sample execution. Built to handle large-scale financial datasets, the platform emphasizes institutional-grade compliance and realistic market simulation to ensure that trading signals are both statistically significant and executable in live market conditions.

The methodology centers on a multi-layered approach to bias mitigation and temporal integrity. The system employs vectorized signal generation across expansive date ranges to prevent data leakage, paired with strict lookahead bias prevention protocols that enforce execution at next-day open prices or intraday mid-price proxies. To ensure financial realism, the framework implements a no-leverage constraint and cash-basis position sizing, where trade execution is dynamically validated against available capital. Furthermore, the architecture is optimized for high-performance computing, utilizing multi-GPU hardware acceleration for intensive model training and iterative strategy refinement.

Upon execution, the framework produces a suite of standardized artifacts and performance analytics. Key outputs include optimized hyperparameter profiles, a detailed transaction registry featuring automated rejection logging for risk breaches, and a comprehensive trading sheet covering the entire out-of-sample period. These results provide a transparent audit trail of strategy performance, allowing for a post-hoc analysis of predictive accuracy and risk-adjusted returns without the influence of in-sample overfit or future-peeking fallacies.

## Implementation Plan

### Progress
- Total: 3 | Done: 3 | In Progress: 0 | Failed: 0

### Verification Order
The following files will be executed (in order) to verify the generated code works:
`tune.py -> main.py`

### Files
| Status | Verified | File | Description | Dependencies |
|--------|----------|------|-------------|--------------|
| [x] | - | `signal_gen.py` | Core model definition and signal generation logic | simicx.data_loader |
| [x] | ✓ | `tune.py` | Hyperparameter optimization script | signal_gen.py, simicx.data_loader |
| [x] | ✓ | `main.py` | Production entry point | signal_gen.py, simicx.data_loader, simicx.trading_sim |


### Progress Log

#### `signal_gen.py`
Completed successfully
Fixed review issue: Violation of Constraint #1 and Constraint #7. The signal generation logic fails to generate signals for the start of the trading period (2025-01-01). The model requires `seq_len` (default 100) days of lookback data to make a prediction. `signal_gen` relies solely on `ohlcv_df` (which starts 2025-01-01 via `get_trading_data`) for inference features. Consequently, the first signal is generated 100 days into the year, missing the required coverage of the full test period. The code must prepend the tail of the training data to the inference data to enable predictions from Day 1.
Fixed review issue: Violation of Constraint #6 and Master Plan. Position sizing is hardcoded to a fixed quantity of 100 shares per asset (`'quantity': 100`). The Master Plan explicitly requires `Q = (Capital / K) / Price_close` and Constraint #6 mandates sizing based on available cash. A fixed quantity of 100 will likely result in significantly incorrect leverage (either trivial exposure for a $1M portfolio or insufficient cash for high-priced stocks).
Fixed review issue: Violation of Constraint #5 (NaN Handling). While the code initializes pivot tables with zeros, it does not strictly follow the Master Plan's requirement to 'Forward fill missing prices' or explicitly drop NaNs before passing data to the model. Relying on implicit zero-filling for missing data points can introduce noise into the `train_std` calculation and Transformer inputs.

#### `tune.py`
Completed successfully

#### `main.py`
Audit failed: CRITICAL ERRORS:

1. **Instruction #1 violation - argparse not used**: Instructions explicitly specify `parser.add_argument('--phase', choices=['limited', 'full'], default='limited')` using argparse syntax. The code implements manual argument parsing with `sys.argv` instead of using the `argparse` module. While functionally equivalent, this deviates from the explicit instruction format.

2. **Instruction #3 violation - Missing imports**: Instructions explicitly require `from simicx.data_loader import get_trading_data, LIMITED_TICKERS, FULL_TICKERS`. The code only imports `get_trading_data` and loads tickers from `simicx/alpha_config.json` via a separate function. The code comment acknowledges this: `# Load config to get tickers (LIMITED_TICKERS/FULL_TICKERS not exported from data_loader)`. While this workaround is reasonable given the API reference doesn't list these constants, it deviates from the explicit import instruction.

3. **Instruction #4 partial deviation**: Instructions specify `sheet = signal_gen(trading_data, **best_params)`. The code maps parameters via `map_params_for_signal_gen()` before calling signal_gen with explicit kwargs. While this mapping is necessary (best_params uses 'learning_rate'/'lookback' but signal_gen expects 'lr'/'seq_len'), it doesn't match the simple `**best_params` unpacking specified.

POSITIVE ASPECTS:
- Data path validation passes: `simicx/alpha_config.json` matches DATA ASSETS
- All API calls use correct parameter names per the reference
- `trading_sim` and `signal_gen` are called with valid parameters
- Code is syntactically valid
- Import/usage consistency is maintained
Completed successfully



## Verification Log

| File | Result | Duration | Notes |
|------|--------|----------|-------|
| `tune.py` | ✓ Passed | 76.1s |  |
| `main.py` | ✓ Passed | 194.3s |  |



## API Reference

### `main.py`

> **Import**: `from main import ...`

> 
Main entry point for HRformer Alpha Strategy production pipeline.

This module orchestrates the complete trading workflow:
1. CLI argument parsing for phase selection
2. Loading optimized hyperparameters from tuning
3. Data loading with appropriate ticker universe
4. Signal generation using trained HRformer model
5. Trading simulation for backtesting

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT


**`parse_args`**
```python
def parse_args() -> Dict[str, str]
```
> Parse command line arguments using argparse.

Returns:
    Dictionary with parsed arguments including:
    - phase: 'limited' or 'full' trading universe

Example:
    >>> import sys
    >>> sys.argv = ['main.py', '--phase', 'limited']
    >>> args = parse_args()
    >>> args['phase']
    'limited'

**`load_best_params`**
```python
def load_best_params(filepath: Path = BEST_PARAMS_PATH) -> Dict[str, Any]
```
> Load optimized hyperparameters from JSON file.

Args:
    filepath: Path to best_params.json file

Returns:
    Dictionary containing hyperparameters with keys:
    learning_rate, hidden_dim, dropout, lookback, batch_size, epochs

Raises:
    SystemExit: If file is missing (Critical Error)
    ValueError: If required keys are missing

Example:
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as td:
    ...     p = Path(td) / "params.json"
    ...     _ = p.write_text('{"learning_rate":0.001,"hidden_dim":64,"dropout":0.1,"lookback":48,"batch_size":32,"epochs":10}')
    ...     params = load_best_params(p)
    ...     params['learning_rate']
    0.001

**`load_alpha_config`**
```python
def load_alpha_config(filepath: Path = ALPHA_CONFIG_PATH) -> Dict[str, Any]
```
> Load alpha configuration including ticker lists.

Args:
    filepath: Path to alpha_config.json file

Returns:
    Dictionary containing configuration with keys:
    LIMITED_TICKERS, FULL_TICKERS, TRAINING_END_DATE, etc.

Raises:
    FileNotFoundError: If config file is missing

Example:
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as td:
    ...     cfg_path = Path(td) / "config.json"
    ...     _ = cfg_path.write_text('{"LIMITED_TICKERS":["AAPL"],"FULL_TICKERS":["AAPL","MSFT"]}')
    ...     cfg = load_alpha_config(cfg_path)
    ...     'LIMITED_TICKERS' in cfg
    True

**`get_tickers_for_phase`**
```python
def get_tickers_for_phase(phase: str, config: Dict[str, Any] = None) -> List[str]
```
> Get appropriate ticker list based on phase.

Uses LIMITED_TICKERS/FULL_TICKERS from data_loader if available,
otherwise falls back to alpha_config.json.

Args:
    phase: 'limited' or 'full'
    config: Alpha config dictionary with ticker lists (optional fallback)

Returns:
    List of ticker symbols for the specified phase

Raises:
    KeyError: If required ticker list not in config

Example:
    >>> config = {'LIMITED_TICKERS': ['AAPL'], 'FULL_TICKERS': ['AAPL', 'MSFT']}
    >>> tickers = get_tickers_for_phase('limited', config)
    >>> 'AAPL' in tickers
    True

**`map_params_for_signal_gen`**
```python
def map_params_for_signal_gen(best_params: Dict[str, Any]) -> Dict[str, Any]
```
> Map best_params keys to signal_gen parameter names.

Transforms parameter names from tuning output format to signal_gen format:
- learning_rate -> lr
- lookback -> seq_len

Args:
    best_params: Dictionary from best_params.json with keys:
        learning_rate, hidden_dim, dropout, lookback, batch_size, epochs

Returns:
    Dictionary with signal_gen compatible parameter names

Example:
    >>> params = {'learning_rate': 0.001, 'hidden_dim': 64, 'dropout': 0.1,
    ...           'lookback': 48, 'batch_size': 32, 'epochs': 10}
    >>> mapped = map_params_for_signal_gen(params)
    >>> mapped['lr']
    0.001
    >>> mapped['seq_len']
    48

**`run_pipeline`**
```python
def run_pipeline(phase: str) -> float
```
> Execute the complete trading pipeline.

Orchestrates data loading, signal generation, and trading simulation
for the specified phase.

Args:
    phase: 'limited' or 'full' ticker universe

Returns:
    Final P&L value from trading simulation

Example:
    >>> # Run with limited phase (requires actual data)
    >>> # pnl = run_pipeline('limited')

**`main`**
```python
def main() -> None
```
> Main entry point for the trading pipeline.

Parses CLI arguments and runs the complete pipeline.

**`simicx_test_load_best_params`**
```python
def simicx_test_load_best_params()
```
> Test loading best parameters from JSON file.

**`simicx_test_get_tickers_for_phase`**
```python
def simicx_test_get_tickers_for_phase()
```
> Test ticker selection based on phase.

**`simicx_test_integration_with_deps`**
```python
def simicx_test_integration_with_deps()
```
> Integration test with signal_gen and trading_sim dependencies.

Tests the complete pipeline interface including:
- Import verification for all dependencies
- Parameter loading and mapping
- Column renaming logic for trading_sim compatibility

**`simicx_test_argparse`**
```python
def simicx_test_argparse()
```
> Test that argparse is used correctly.

---

### `signal_gen.py`

> **Import**: `from signal_gen import ...`

> 
Signal Generation Module for HRformer Alpha Strategy

This module implements a Hierarchical Reversible Transformer (HRformer) for 
multi-component time series forecasting with inter-stock correlation attention.

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT


**class `RevIN`**
```python
class RevIN(nn.Module):
```
> Reversible Instance Normalization.

Normalizes input along the time dimension and stores statistics
for reversing the normalization later. This is crucial for time series
where distribution shifts need to be handled gracefully.

Args:
    num_features: Number of input features
    eps: Small constant for numerical stability
    affine: If True, learn scale and shift parameters
    
Example:
    >>> revin = RevIN(num_features=5)
    >>> x = torch.randn(32, 100, 5)  # (batch, time, features)
    >>> x_norm = revin(x, mode='norm')
    >>> x_denorm = revin(x_norm, mode='denorm')
    >>> torch.allclose(x, x_denorm, atol=1e-5)
    True

**`RevIN.__init__`**
```python
def __init__(self, num_features: int, eps: float = 1e-05, affine: bool = True)
```
**`RevIN.forward`**
```python
def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor
```
> Apply or reverse instance normalization.

Args:
    x: Input tensor of shape (batch, time, features)
    mode: 'norm' to normalize, 'denorm' to denormalize
    
Returns:
    Normalized or denormalized tensor of same shape
    
Raises:
    RuntimeError: If denorm is called before norm
    ValueError: If mode is not 'norm' or 'denorm'

**class `MCDL`**
```python
class MCDL(nn.Module):
```
> Multi-Component Decomposition Layer.

Decomposes time series into three components:
- Trend: Extracted using AvgPool1d (low-frequency component)
- Cyclic: Extracted using FFT, keeping frequencies above threshold
- Volatility: Residual after removing Trend and Cyclic

Args:
    kernel_size: Kernel size for average pooling (trend extraction)
    freq_threshold: Frequency threshold for cyclic extraction (0-1, proportion of max freq)
    
Example:
    >>> mcdl = MCDL(kernel_size=5, freq_threshold=0.1)
    >>> x = torch.randn(32, 100, 5)  # (batch, time, features)
    >>> trend, cyclic, volatility = mcdl(x)
    >>> trend.shape
    torch.Size([32, 100, 5])

**`MCDL.__init__`**
```python
def __init__(self, kernel_size: int = 5, freq_threshold: float = 0.1)
```
**`MCDL.forward`**
```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```
> Decompose input into trend, cyclic, and volatility components.

Args:
    x: Input tensor of shape (batch, time, features)
    
Returns:
    Tuple of (trend, cyclic, volatility), each with same shape as input

**class `FourierAttention`**
```python
class FourierAttention(nn.Module):
```
> Fourier Attention Layer.

Applies learnable attention weights in the frequency domain.
This allows the model to learn which frequency components are
most important for prediction.

Args:
    seq_len: Length of input sequence
    hidden_dim: Hidden dimension
    
Example:
    >>> fa = FourierAttention(seq_len=100, hidden_dim=64)
    >>> x = torch.randn(32, 100, 64)
    >>> output = fa(x)
    >>> output.shape
    torch.Size([32, 100, 64])

**`FourierAttention.__init__`**
```python
def __init__(self, seq_len: int, hidden_dim: int)
```
**`FourierAttention.forward`**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```
> Apply Fourier attention.

Args:
    x: Input tensor of shape (batch, time, hidden)
    
Returns:
    Output tensor of shape (batch, time, hidden)

**class `CTE`**
```python
class CTE(nn.Module):
```
> Component-wise Temporal Encoder.

Encodes each component (trend, cyclic, volatility) with specialized architectures:
- Trend: TransformerEncoder (captures long-range dependencies)
- Cyclic: FourierAttention (learnable weights in freq domain)
- Volatility: RevIN -> MLP -> LSTM -> RevIN -> MLP (handles non-stationary noise)

Args:
    input_dim: Number of input features
    hidden_dim: Hidden dimension for all encoders
    seq_len: Sequence length
    n_heads: Number of attention heads for transformer
    n_layers: Number of transformer layers
    dropout: Dropout rate
    
Example:
    >>> cte = CTE(input_dim=5, hidden_dim=64, seq_len=100)
    >>> trend = torch.randn(32, 100, 5)
    >>> cyclic = torch.randn(32, 100, 5)
    >>> volatility = torch.randn(32, 100, 5)
    >>> enc_t, enc_c, enc_v = cte(trend, cyclic, volatility)
    >>> enc_t.shape
    torch.Size([32, 100, 64])

**`CTE.__init__`**
```python
def __init__(self, input_dim: int, hidden_dim: int, seq_len: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1)
```
**`CTE.forward`**
```python
def forward(self, trend: torch.Tensor, cyclic: torch.Tensor, volatility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```
> Encode trend, cyclic, and volatility components.

Args:
    trend: Trend component of shape (batch, time, features)
    cyclic: Cyclic component of shape (batch, time, features)
    volatility: Volatility component of shape (batch, time, features)
    
Returns:
    Tuple of encoded (trend, cyclic, volatility), each shape (batch, time, hidden)

**class `AMCI`**
```python
class AMCI(nn.Module):
```
> Adaptive Multi-Component Integration.

Uses learnable gates to adaptively combine encoded components.
The gates are computed based on the concatenated components,
allowing the model to dynamically weight each component.

Args:
    hidden_dim: Hidden dimension
    
Example:
    >>> amci = AMCI(hidden_dim=64)
    >>> trend = torch.randn(32, 100, 64)
    >>> cyclic = torch.randn(32, 100, 64)
    >>> vol = torch.randn(32, 100, 64)
    >>> output = amci(trend, cyclic, vol)
    >>> output.shape
    torch.Size([32, 100, 64])

**`AMCI.__init__`**
```python
def __init__(self, hidden_dim: int)
```
**`AMCI.forward`**
```python
def forward(self, trend: torch.Tensor, cyclic: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor
```
> Adaptively combine three components using learned gates.

Args:
    trend: Encoded trend of shape (batch, time, hidden)
    cyclic: Encoded cyclic of shape (batch, time, hidden)
    volatility: Encoded volatility of shape (batch, time, hidden)
    
Returns:
    Combined output of shape (batch, time, hidden)

**class `ISCA`**
```python
class ISCA(nn.Module):
```
> Inter-Stock Correlation Attention.

Applies cross-attention across the stock dimension to capture
inter-stock correlations. This allows the model to learn
relationships between different stocks.

Args:
    hidden_dim: Hidden dimension
    n_heads: Number of attention heads
    dropout: Dropout rate
    
Example:
    >>> isca = ISCA(hidden_dim=64, n_heads=4)
    >>> x = torch.randn(32, 10, 64)  # (batch, stocks, hidden)
    >>> output = isca(x)
    >>> output.shape
    torch.Size([32, 10, 64])

**`ISCA.__init__`**
```python
def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1)
```
**`ISCA.forward`**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```
> Apply cross-attention across stock dimension.

Args:
    x: Input tensor of shape (batch, stocks, hidden)
    
Returns:
    Output tensor of shape (batch, stocks, hidden)

**class `HRformer`**
```python
class HRformer(nn.Module):
```
> Hierarchical Reversible Transformer for Time Series Forecasting.

Combines all components for multi-stock time series prediction:
1. MCDL: Multi-Component Decomposition (Trend, Cyclic, Volatility)
2. CTE: Component-wise Temporal Encoding
3. AMCI: Adaptive Multi-Component Integration
4. ISCA: Inter-Stock Correlation Attention

Args:
    input_dim: Number of input features (OHLCV = 5)
    hidden_dim: Hidden dimension
    seq_len: Sequence length
    n_stocks: Number of stocks
    n_heads: Number of attention heads
    n_layers: Number of transformer layers
    dropout: Dropout rate
    kernel_size: Kernel size for trend extraction
    freq_threshold: Frequency threshold for cyclic extraction
    
Example:
    >>> model = HRformer(input_dim=5, hidden_dim=64, seq_len=100, n_stocks=20)
    >>> x = torch.randn(32, 20, 100, 5)  # (batch, stocks, time, features)
    >>> probs = model(x)  # (batch, stocks)
    >>> probs.shape
    torch.Size([32, 20])

**`HRformer.__init__`**
```python
def __init__(self, input_dim: int = 5, hidden_dim: int = 64, seq_len: int = 100, n_stocks: int = 20, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1, kernel_size: int = 5, freq_threshold: float = 0.1)
```
**`HRformer.forward`**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```
> Forward pass through HRformer.

Args:
    x: Input tensor of shape (batch, stocks, time, features)
    
Returns:
    Probability tensor of shape (batch, stocks) - probability of price rise

**`_pivot_ohlcv_data`**
```python
def _pivot_ohlcv_data(df: pd.DataFrame, ticker_list: List[str]) -> Tuple[np.ndarray, List[pd.Timestamp], List[str]]
```
> Pivot OHLCV data to 3D array (time, ticker, feature).

Applies forward-fill then backward-fill to handle missing data
(Constraint #5: NaN Handling per Master Plan).

Args:
    df: OHLCV DataFrame with columns: time, ticker, open, high, low, close, volume
    ticker_list: List of tickers to include

Returns:
    Tuple of (data_array, dates, common_tickers)
    - data_array: Shape (n_times, n_tickers, 5)
    - dates: List of timestamps
    - common_tickers: List of tickers actually present

**`_create_training_sequences`**
```python
def _create_training_sequences(data: np.ndarray, labels: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]
```
> Create training sequences from time series data.

Args:
    data: Normalized data of shape (time, stocks, features)
    labels: Labels of shape (time, stocks)
    seq_len: Sequence length
    
Returns:
    Tuple of (X, y) arrays
    - X: Shape (samples, time, stocks, features)
    - y: Shape (samples, stocks)

**`signal_gen`**
```python
def signal_gen(ohlcv_df: pd.DataFrame, phase: str = 'full', train_cutoff_date: Optional[str] = None, train_data: Optional[pd.DataFrame] = None, lr: float = 0.0001, epochs: int = 10, batch_size: int = 32, hidden_dim: int = 64, seq_len: int = 100, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1, kernel_size: int = 5, freq_threshold: float = 0.1, lookahead: int = 48, rebalance_period: int = 48, initial_capital: float = 1000000.0) -> pd.DataFrame
```
> Generate trading signals using HRformer model.

Trains on historical data and generates buy/sell signals for portfolio construction.
On each rebalance date (every 48 days), selects top K stocks by predicted probability
of price rise in the next 48 days.

Position sizing follows Master Plan formula: Q = (Capital / K) / Price_close

Args:
    ohlcv_df: OHLCV DataFrame for prediction with columns:
              time, ticker, open, high, low, close, volume
    phase: 'full' or 'limited' - determines K value (20 vs 3) and ticker universe
    train_cutoff_date: Optional date cutoff for training data (for tune.py leakage prevention)
    train_data: Optional override for training data (bypasses get_training_data)
    lr: Learning rate for optimizer
    epochs: Number of training epochs
    batch_size: Batch size for training
    hidden_dim: Hidden dimension for model
    seq_len: Sequence length for temporal modeling
    n_heads: Number of attention heads
    n_layers: Number of transformer layers
    dropout: Dropout rate
    kernel_size: Kernel size for trend extraction in MCDL
    freq_threshold: Frequency threshold for cyclic extraction in MCDL
    lookahead: Number of days to look ahead for labels (default 48)
    rebalance_period: Rebalancing period in days (default 48)
    initial_capital: Initial portfolio capital for position sizing (default 1,000,000)

Returns:
    DataFrame with columns ['date', 'ticker', 'action', 'quantity', 'price']
    - action: 'buy' for top K stocks, 'sell' for exiting positions

Example:
    >>> import pandas as pd
    >>> # Load prediction data
    >>> signals = signal_gen(ohlcv_df, phase='limited', epochs=5)
    >>> signals.head()
       date ticker action  quantity   price
    0  2025-01-02   AAPL    buy       100  185.50

**`simicx_test_revin`**
```python
def simicx_test_revin()
```
> Test RevIN normalization and denormalization.

**`simicx_test_mcdl`**
```python
def simicx_test_mcdl()
```
> Test Multi-Component Decomposition Layer.

**`simicx_test_hrformer`**
```python
def simicx_test_hrformer()
```
> Test full HRformer model forward pass.

**`simicx_test_integration_signal_gen`**
```python
def simicx_test_integration_signal_gen()
```
> Integration test for signal_gen with data_loader.

---

### `simicx/data_loader.py`

> **Import**: `from simicx.data_loader import ...`

> 
SimicX Data Loader Module (Database Version)

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT

Centralized OHLCV data loading from MongoDB with strict temporal controls
for alpha discovery and backtesting.

Key Features:
- Strict train/test split: Training ≤ 2024-12-31, Trading ≥ 2025-01-01
- Date alignment across tickers (consistent coverage)
- 2-phase testing support: LIMITED → FULL tickers
- Multi-asset extensibility (not equity-specific)

Usage:
    from data_loader import get_training_data, get_trading_data
    
    # For tune.py (hyperparameter optimization)
    train_df = get_training_data(LIMITED_TICKERS, years_back=3)
    
    # For main.py (backtesting)
    trade_df = get_trading_data(FULL_TICKERS)


**`get_mongo_client`**
```python
def get_mongo_client() -> MongoClient
```
> Get or create MongoDB client connection (thread-safe singleton).

Returns:
    MongoClient instance with connection pooling.

Raises:
    RuntimeError: If connection to MongoDB fails (Fail Fast).

Example:
    >>> client = get_mongo_client()
    >>> db = client[MONGODB_DATABASE]

**`get_collection`**
```python
def get_collection()
```
> Get OHLCV collection instance.

Returns:
    MongoDB collection for OHLCV data.

**`get_tickers`**
```python
def get_tickers() -> List[str]
```
> Get list of unique ticker symbols available in the database.

Returns:
    List[str]: Sorted list of available ticker symbols.

Example:
    >>> tickers = get_tickers()
    >>> print(f"Found {len(tickers)} tickers")
    >>> 'SPY' in tickers
    True

**`get_date_range`**
```python
def get_date_range(ticker: str) -> Tuple[datetime, datetime]
```
> Get the date range (start and end dates) for a specific ticker.

Args:
    ticker: Stock ticker symbol (e.g., 'SPY', 'AAPL').

Returns:
    Tuple[datetime, datetime]: (start_date, end_date) for the ticker.

Raises:
    ValueError: If the ticker is not found in the database.

Example:
    >>> start, end = get_date_range('SPY')
    >>> print(f"SPY data: {start.date()} to {end.date()}")

**`get_data`**
```python
def get_data(ticker: Optional[str] = None, tickers: Optional[List[str]] = None, phase: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None, align_dates: bool = True) -> pd.DataFrame
```
> Get OHLCV data with optional filtering by ticker(s), phase, and date range.

Args:
    ticker: Single ticker symbol (e.g., 'SPY'). Mutually exclusive with tickers/phase.
    tickers: List of ticker symbols. Mutually exclusive with ticker/phase.
    phase: 'limited' (LIMITED_TICKERS) or 'full' (FULL_TICKERS). Overrides ticker/tickers.
    start_date: Start date (inclusive) in 'YYYY-MM-DD' format.
    end_date: End date (inclusive) in 'YYYY-MM-DD' format.
    align_dates: If True, only return dates where ALL tickers have data.

Returns:
    pd.DataFrame: OHLCV data with columns: time, ticker, open, high, low, close, volume

Raises:
    ValueError: If neither phase, ticker nor tickers is provided.

**`get_training_data`**
```python
def get_training_data(tickers: Optional[List[str]] = None, phase: Optional[str] = None, years_back: Optional[int] = None, align_dates: bool = True) -> pd.DataFrame
```
> Get training/tuning data (all data up to and including 2024-12-31).

CRITICAL: This function ensures NO data after 2024-12-31 is included.

Args:
    tickers: List of ticker symbols. Defaults to FULL_TICKERS if phase not set.
    phase: 'limited' or 'full'. Sets tickers and default years_back from config.
    years_back: Override default years_back.
    align_dates: If True, only return dates where ALL tickers have data.

Returns:
    pd.DataFrame: Training OHLCV data.

**`get_trading_data`**
```python
def get_trading_data(tickers: Optional[List[str]] = None, align_dates: bool = True) -> pd.DataFrame
```
> Get trading simulation data (all data from start of 2025 onwards).

CRITICAL: This function ensures ONLY data from 2025-Jan-01 onwards is returned,
which should be used for backtesting and performance reporting.

Args:
    tickers: List of ticker symbols. Defaults to FULL_TICKERS.
    align_dates: If True, only return dates where ALL tickers have data.

Returns:
    pd.DataFrame: Trading OHLCV data (2025 onwards).

Example:
    >>> # Trading data for backtesting
    >>> trade_df = get_trading_data(FULL_TICKERS)
    >>> trade_df['time'].min()  # Should be >= 2025-Jan-01

**`get_ohlcv`**
```python
def get_ohlcv(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame
```
> Convenience alias for get_data() - get OHLCV data for a single ticker.

Args:
    ticker: Stock ticker symbol.
    start_date: Optional start date.
    end_date: Optional end date.

Returns:
    pd.DataFrame: OHLCV data for the specified ticker.

**`simicx_test_data_loader`**
```python
def simicx_test_data_loader()
```
> Test function for data_loader module.

Verifies:
1. Database connectivity
2. Ticker availability
3. Date range queries
4. Temporal split integrity (training ≤ 2024, trading ≥ 2025)
5. Date alignment across tickers

---

### `simicx/trading_sim.py`

> **Import**: `from simicx.trading_sim import ...`

> 
SimicX Trading Simulation Module

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT

Comprehensive backtesting engine for trading strategies with realistic fee modeling,
position tracking, constraint validation, and performance metrics.

Usage:
    from alpha_stream.tools.trading_sim import trading_sim
    
    pnl, pnl_details = trading_sim(
        trading_sheet=trading_df,
        initial_capital=1_000_000.0
    )


**class `Position`**
```python
class Position:
```
> Represents a position in a single asset with FIFO (First-In-First-Out) cost basis tracking.

This class manages both long and short positions for a single security ticker,
tracking individual lots for accurate cost basis and P&L calculations.

Position Types:
    - **Long positions**: positive quantity (bought stock, expecting price increase)
    - **Short positions**: negative quantity (borrowed and sold stock, expecting price decrease)

FIFO Accounting:
    When selling/covering positions, the oldest lots are consumed first. This provides
    accurate cost basis tracking for tax purposes and precise realized P&L calculations.

Attributes:
    ticker: The asset ticker (e.g., 'AAPL', 'DIA')
    lots: List of (quantity, price_per_share) tuples. Each tuple represents a lot:
          - quantity > 0 for long lots (shares owned)
          - quantity < 0 for short lots (shares owed)
          - price is the cost basis per share (including commission for buys)

Example:
    >>> # Create a new position and add lots
    >>> pos = Position(ticker='AAPL')
    >>> pos.add(100, 150.0)  # Buy 100 shares at $150
    >>> pos.add(50, 155.0)   # Buy 50 more at $155
    >>> 
    >>> # Check position state
    >>> pos.quantity  # Total shares: 150
    150.0
    >>> pos.avg_cost  # Weighted average: (100*150 + 50*155) / 150 = 151.67
    151.66666666666666
    >>> pos.market_value  # Total cost basis: 100*150 + 50*155 = 22750
    22750.0
    >>> 
    >>> # Sell using FIFO - removes from oldest lot first
    >>> removed_qty, cost_basis = pos.remove(75)  # Sell 75 shares
    >>> removed_qty  # Actually removed
    75.0
    >>> cost_basis  # Cost basis of sold shares (75 * $150 from first lot)
    11250.0
    >>> pos.lots  # Remaining: 25 shares at $150, 50 shares at $155
    [(25, 150.0), (50, 155.0)]
    
    >>> # Short position example
    >>> short_pos = Position(ticker='TSLA')
    >>> short_pos.add(-100, 200.0)  # Short 100 shares at $200 (net proceeds per share)
    >>> short_pos.quantity  # Negative = owe shares
    -100

**`Position.quantity`**
```python
def quantity(self) -> float
```
> Total quantity held across all lots.

Returns:
    float: Sum of all lot quantities.
           Positive for net long position (own shares).
           Negative for net short position (owe shares).
           Zero if no position or balanced.

Example:
    >>> pos = Position(ticker='AAPL')
    >>> pos.add(100, 150.0)  # Long 100
    >>> pos.add(-30, 155.0)  # Short 30 (partially close)
    >>> pos.quantity
    70.0

**`Position.avg_cost`**
```python
def avg_cost(self) -> float
```
> Weighted average cost basis per share across all lots.

Calculates the volume-weighted average price of all open lots.
For long positions, this represents the average purchase price.
For short positions, this represents the average sale price (proceeds received).

Returns:
    float: Average cost per share. Returns 0.0 if no position held.

Formula:
    avg_cost = Σ(quantity_i × price_i) / Σ(quantity_i)

Example:
    >>> pos = Position(ticker='AAPL')
    >>> pos.add(100, 150.0)  # 100 × $150 = $15,000
    >>> pos.add(50, 160.0)   # 50 × $160 = $8,000
    >>> pos.avg_cost  # ($15,000 + $8,000) / 150 = $153.33
    153.33333333333334
    
    >>> # With no position
    >>> empty_pos = Position(ticker='MSFT')
    >>> empty_pos.avg_cost
    0.0

**`Position.market_value`**
```python
def market_value(self) -> float
```
> Total cost basis of the position (quantity × cost per share for each lot).

This represents the total capital invested in the position. For unrealized
P&L calculation, compare this with `quantity × current_market_price`.

Returns:
    float: Sum of (quantity × price) for all lots.
           Positive for long positions (capital deployed).
           Negative for short positions (proceeds received).

Note:
    This is NOT the current market value. Use with current prices:
    `unrealized_pnl = (quantity × current_price) - market_value`

Example:
    >>> pos = Position(ticker='AAPL')
    >>> pos.add(100, 150.0)  # Cost: $15,000
    >>> pos.add(50, 160.0)   # Cost: $8,000
    >>> pos.market_value     # Total cost basis: $23,000
    23000.0
    
    >>> # Calculate unrealized P&L if current price is $170
    >>> current_price = 170.0
    >>> unrealized_pnl = (pos.quantity * current_price) - pos.market_value
    >>> unrealized_pnl  # 150 × $170 - $23,000 = $2,500
    2500.0

**`Position.add`**
```python
def add(self, quantity: float, price_per_share: float) -> None
```
> Add shares to this position as a new lot.

Creates a new lot entry in the position's lot list. Does NOT merge with
existing lots to maintain accurate FIFO tracking.

Position Direction:
    - quantity > 0: Adding long position (buying shares)
    - quantity < 0: Adding short position (shorting shares)

Args:
    quantity: Number of shares to add. Can be positive (long) or negative (short).
             Zero quantity is ignored (no lot created).
    price_per_share: Cost basis per share, should INCLUDE commission for accuracy.
                    For buys: (execution_price × quantity + commission) / quantity
                    For shorts: net_proceeds / quantity (after commission)

Returns:
    None. Modifies the position in-place.

Example:
    >>> pos = Position(ticker='AAPL')
    >>> 
    >>> # Add long position (100 shares at $150.15 including commission)
    >>> pos.add(100, 150.15)
    >>> pos.lots
    [(100, 150.15)]
    >>> 
    >>> # Add another lot at different price
    >>> pos.add(50, 155.0)
    >>> pos.lots  # Two separate lots maintained
    [(100, 150.15), (50, 155.0)]
    >>> 
    >>> # Add short position
    >>> short_pos = Position(ticker='TSLA')
    >>> short_pos.add(-100, 200.0)  # Short 100 at $200/share net proceeds
    >>> short_pos.quantity
    -100
    
Note:
    - Each call creates a new lot, even if price matches existing lots
    - Zero quantity is silently ignored (useful for conditional adds)
    - For accurate P&L, include commission in price_per_share

**`Position.remove`**
```python
def remove(self, quantity: float, is_short_covering: bool = False) -> Tuple[float, float]
```
> Remove shares from position using FIFO (First-In-First-Out) accounting.

Consumes lots in chronological order (oldest first) until the requested
quantity is removed. Handles partial lot consumption correctly.

FIFO Logic:
    1. Start with the oldest lot (first in the list)
    2. If lot quantity ≤ remaining to remove: consume entire lot
    3. If lot quantity > remaining: consume partial lot, update lot size
    4. Repeat until requested quantity is removed or no lots remain

Args:
    quantity: Number of shares to remove (always pass a POSITIVE value).
             The method handles both long and short position removal internally.
    is_short_covering: Optional flag indicating if this remove is for covering
                      a short position. Currently used for documentation purposes
                      but may affect future logic extensions.

Returns:
    Tuple[float, float]: A tuple containing:
        - actual_quantity_removed: How many shares were actually removed.
          May be less than requested if position is smaller.
        - cost_basis: Total cost basis of removed shares (quantity × price
          summed across consumed lots). For short positions, this is negative
          (representing proceeds received).

Example:
    >>> pos = Position(ticker='AAPL')
    >>> pos.add(100, 150.0)  # Lot 1: 100 @ $150
    >>> pos.add(50, 160.0)   # Lot 2: 50 @ $160
    >>> 
    >>> # Remove 75 shares (FIFO takes from oldest lot first)
    >>> removed_qty, cost_basis = pos.remove(75)
    >>> removed_qty
    75.0
    >>> cost_basis  # 75 × $150 from first lot
    11250.0
    >>> pos.lots  # First lot reduced, second untouched
    [(25, 150.0), (50, 160.0)]
    >>> 
    >>> # Remove remaining 25 from first lot + 30 from second
    >>> removed_qty, cost_basis = pos.remove(55)
    >>> cost_basis  # (25 × $150) + (30 × $160) = 3750 + 4800 = 8550
    8550.0
    >>> pos.lots  # 20 shares remaining from second lot
    [(20, 160.0)]
    >>> 
    >>> # Try to remove more than available
    >>> removed_qty, cost_basis = pos.remove(50)
    >>> removed_qty  # Only 20 available
    20.0
    >>> pos.lots  # Position fully closed
    []
    
    >>> # Edge case: remove from empty position
    >>> empty_pos = Position(ticker='MSFT')
    >>> empty_pos.remove(100)
    (0.0, 0.0)

Note:
    - For short positions (negative lot quantities), cost_basis represents
      the proceeds received when the short was opened (negative value).
    - The function gracefully handles partial fills when position is smaller
      than requested quantity.
    - Zero or negative quantity returns (0.0, 0.0) immediately.

**class `Portfolio`**
```python
class Portfolio:
```
> Portfolio state tracking with cash and positions.

Manages the overall portfolio state including cash balance and all open positions
(both long and short). Provides methods to calculate total portfolio value and
track holdings across multiple assets.

Attributes:
    cash: Current cash balance in the portfolio
    positions: Dictionary mapping ticker -> Position for all active positions

Example:
    >>> portfolio = Portfolio(cash=100_000.0)
    >>> # Buy 100 shares of AAPL at $150
    >>> pos = portfolio.get_position('AAPL')
    >>> pos.add(100, 150.0)
    >>> portfolio.cash -= 100 * 150.0
    >>> 
    >>> # Check portfolio value
    >>> prices = {'AAPL': 155.0}
    >>> total_value = portfolio.get_total_value(prices)
    >>> print(f"Portfolio value: ${total_value:,.2f}")  # Cash + holdings

**`Portfolio.get_position`**
```python
def get_position(self, ticker: str) -> Position
```
> Get or create position for a given ticker.

If the ticker doesn't exist in the portfolio, a new empty Position
is created and added to the portfolio.

Args:
    ticker: Asset ticker (e.g., 'AAPL', 'DIA')
    
Returns:
    Position object for the specified ticker

**`Portfolio.get_holdings_value`**
```python
def get_holdings_value(self, prices: Dict[str, float]) -> float
```
> Calculate total market value of all holdings at current prices.

For long positions (quantity > 0): positive value (asset)
For short positions (quantity < 0): negative value (liability)

Args:
    prices: Dictionary mapping ticker -> current market price

Returns:
    Total market value of all holdings. Positive for net long positions,
    negative for net short positions.
    
Example:
    >>> portfolio = Portfolio(cash=50_000)
    >>> # Long 100 AAPL, Short 50 TSLA
    >>> portfolio.get_position('AAPL').add(100, 150.0)
    >>> portfolio.get_position('TSLA').add(-50, 200.0)  # Short
    >>> prices = {'AAPL': 155.0, 'TSLA': 195.0}
    >>> holdings = portfolio.get_holdings_value(prices)
    >>> # = (100 * 155) + (-50 * 195) = 15,500 - 9,750 = 5,750

**`Portfolio.get_total_value`**
```python
def get_total_value(self, prices: Dict[str, float]) -> float
```
> Calculate total portfolio value (cash + holdings).

This represents the liquidation value of the portfolio if all positions
were closed at the given prices.

Args:
    prices: Dictionary mapping ticker -> current market price
    
Returns:
    Total portfolio value = cash + holdings_value
    
Note:
    For portfolios with short positions, holdings_value may be negative,
    representing the liability from short positions.

**`validate_trading_sheet`**
```python
def validate_trading_sheet(trading_sheet: pd.DataFrame) -> pd.DataFrame
```
> Validate and normalize trading sheet input.

Performs comprehensive validation and normalization of trading signals:
- Normalizes column names to lowercase
- Validates required columns are present
- Ensures action values are 'buy' or 'sell'
- Converts time to datetime format
- Validates numeric columns (quantity, price)
- Removes invalid rows with warnings
- Sorts by time chronologically

Args:
    trading_sheet: DataFrame with trade instructions containing columns:
        - time: Trading timestamp (str or datetime)
        - ticker: Asset ticker symbol (str)
        - action: 'buy' or 'sell' (case-insensitive)
        - quantity: Trade quantity (numeric)
        - price: Target execution price (numeric)
    
Returns:
    Validated DataFrame with normalized column names, sorted by time,
    with only valid rows retained
    
Raises:
    ValueError: If required columns are missing or action values are invalid
    
Warnings:
    Issues warnings when removing rows with invalid quantity/price values
    
Example:
    >>> import pandas as pd
    >>> # Valid input
    >>> trades = pd.DataFrame({
    ...     'TIME': ['2024-01-02 09:30:00', '2024-01-02 10:00:00'],
    ...     'TICKER': ['AAPL', 'MSFT'],
    ...     'Action': ['BUY', 'SELL'],  # Case-insensitive
    ...     'Quantity': [100, 50],
    ...     'Price': [150.50, 380.25]
    ... })
    >>> validated = validate_trading_sheet(trades)
    >>> validated.columns.tolist()
    ['time', 'ticker', 'action', 'quantity', 'price']
    
    >>> # Invalid input - missing column
    >>> bad_trades = pd.DataFrame({'time': ['2024-01-02'], 'ticker': ['AAPL']})
    >>> validate_trading_sheet(bad_trades)  # Raises ValueError
    
    >>> # Invalid input - bad action value
    >>> bad_trades = pd.DataFrame({
    ...     'time': ['2024-01-02'], 'ticker': ['AAPL'],
    ...     'action': ['HOLD'], 'quantity': [100], 'price': [150]
    ... })
    >>> validate_trading_sheet(bad_trades)  # Raises ValueError
    
Note:
    - Column names are case-insensitive and trimmed of whitespace
    - Empty DataFrames return an empty DataFrame with correct columns
    - Rows with NaN quantity or price are removed with a warning
    - All trades are sorted chronologically for proper execution order

**`calculate_execution_price`**
```python
def calculate_execution_price(target_price: float, action: str, slippage_rate: float = 0.0005, spread_rate: float = 0.0001) -> float
```
> Calculate realistic execution price with slippage and bid-ask spread.

Models the market impact of executing a trade by adjusting the target price
for realistic market conditions. Both slippage and spread work against the trader.

Cost Components:
    **Slippage**: The difference between expected and actual execution price due to
    market movement, order size impact, or latency. Always works against you.
    
    **Spread**: The difference between bid and ask prices. Buyers pay the ask (higher),
    sellers receive the bid (lower). The `spread_rate` represents half the spread.

Formulas:
    - Buy:  execution_price = target_price × (1 + slippage_rate + spread_rate)
    - Sell: execution_price = target_price × (1 - slippage_rate - spread_rate)

Args:
    target_price: The target/signal price from your trading strategy.
                 This is typically the close price or a calculated entry price.
    action: Trade direction - 'buy' or 'sell' (case-sensitive).
    slippage_rate: Slippage as a fraction of price. Default 0.0005 (0.05% or 5 bps).
                  Larger orders or less liquid assets typically have higher slippage.
    spread_rate: Half-spread as a fraction of price. Default 0.0001 (0.01% or 1 bp).
                Represents half of the bid-ask spread. Full spread = 2 × spread_rate.

Returns:
    float: Adjusted execution price after accounting for slippage and spread.

Example:
    >>> # Buying at $100 with default rates
    >>> calculate_execution_price(100.0, 'buy')
    100.06  # = 100 × (1 + 0.0005 + 0.0001)
    
    >>> # Selling at $100 with default rates  
    >>> calculate_execution_price(100.0, 'sell')
    99.94  # = 100 × (1 - 0.0005 - 0.0001)
    
    >>> # High-impact trade with larger slippage
    >>> calculate_execution_price(50.0, 'buy', slippage_rate=0.002, spread_rate=0.001)
    50.15  # = 50 × (1 + 0.002 + 0.001) = 50 × 1.003
    
    >>> # Calculate round-trip cost (buy then sell same price)
    >>> buy_price = calculate_execution_price(100.0, 'buy')
    >>> sell_price = calculate_execution_price(100.0, 'sell')
    >>> round_trip_cost = buy_price - sell_price  # $0.12 per share
    >>> round_trip_cost_pct = (round_trip_cost / 100.0) * 100  # 0.12%

Note:
    - Default values represent typical costs for liquid US equities
    - Crypto, forex, and less liquid assets often have higher rates
    - These costs are IN ADDITION to commissions
    - For limit orders with guaranteed fills, you may set both rates to 0

**`calculate_performance_metrics`**
```python
def calculate_performance_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]
```
> Calculate comprehensive performance metrics for trading strategy evaluation.

Computes a wide range of industry-standard metrics to assess strategy performance,
risk characteristics, and return quality. All calculations assume 252 trading days
per year for annualization.

Metric Categories:
    **Return Metrics**:
    - `total_return`: Cumulative return over the entire period
    - `annualized_return`: Geometric mean annual return
    - `avg_daily_return`: Arithmetic mean of daily returns
    - `volatility`: Annualized standard deviation of returns
    
    **Risk-Adjusted Metrics**:
    - `sharpe_ratio`: Excess return per unit of total risk
    - `sortino_ratio`: Excess return per unit of downside risk
    - `calmar_ratio`: Annualized return / max drawdown
    
    **Drawdown Analysis**:
    - `max_drawdown`: Largest peak-to-trough decline (negative value)
    - `avg_drawdown`: Average drawdown when in drawdown
    - `max_drawdown_duration`: Longest drawdown period in days
    
    **Win/Loss Statistics**:
    - `win_rate`: Percentage of positive return days
    - `profit_factor`: Gross profit / gross loss
    - `payoff_ratio`: Average win / average loss (absolute)
    
    **Distribution Characteristics**:
    - `skewness`: Return distribution asymmetry (positive = right tail)
    - `kurtosis`: Return distribution tail thickness (>3 = fat tails)
    - `var_95`: 5th percentile daily return (Value at Risk)
    - `cvar_95`: Mean of returns below VaR (Conditional VaR / Expected Shortfall)

Args:
    returns: pandas Series of daily returns in decimal format.
            Example: 0.01 represents a +1% daily return, -0.02 represents a -2% return.
            Must be simple returns, not log returns.
    risk_free_rate: Annual risk-free rate in decimal format.
                   Default 0.02 represents 2% annual risk-free rate.
                   Used for Sharpe and Sortino ratio calculations.

Returns:
    Dict[str, float]: Dictionary containing all computed metrics.
    Returns empty dict if returns series is empty or has fewer than 2 values.

Example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Generate sample returns (252 trading days)
    >>> np.random.seed(42)
    >>> daily_returns = pd.Series(np.random.normal(0.0005, 0.02, 252))
    >>> 
    >>> metrics = calculate_performance_metrics(daily_returns)
    >>> 
    >>> # Access individual metrics
    >>> print(f"Total Return: {metrics['total_return']*100:.2f}%")
    >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    >>> print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    >>> print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    
    >>> # Using with trading_sim results
    >>> pnl, details = trading_sim(trading_sheet=trades, ohlcv_path='data.csv')
    >>> if 'metrics' in details.attrs:
    ...     metrics = details.attrs['metrics']
    ...     print(f"Strategy Sharpe: {metrics['sharpe_ratio']:.2f}")

Formulas:
    - Sharpe Ratio: (E[R] - Rf) × √252 / (σ × √252)
    - Sortino Ratio: (E[R] - Rf) × 252 / (σ_downside × √252)
    - Max Drawdown: min((cumulative - running_max) / running_max)
    - VaR 95%: 5th percentile of return distribution
    - CVaR 95%: E[R | R ≤ VaR_95]

Note:
    - Returns 999.0 for infinite ratios (e.g., Sortino with no downside days)
    - Metrics are computed on non-NaN values only
    - Assumes continuous daily data (gaps may affect drawdown duration)

**`convert_signals_to_trades`**
```python
def convert_signals_to_trades(signals: pd.DataFrame, signal_type: str, ohlcv_df: pd.DataFrame, initial_capital: float = 1000000.0, max_position_pct: float = 0.25) -> pd.DataFrame
```
> Convert raw alpha signals into a trading sheet.

Args:
    signals: DataFrame with [time, ticker, signal]
    signal_type: 'TARGET_WEIGHT', 'ALPHA_SCORE', or 'BINARY'
    ohlcv_df: Market data for price lookup
    initial_capital: For calculating quantities
    max_position_pct: Max allocation per asset
    
Returns:
    DataFrame trading_sheet [time, ticker, action, quantity, price]

**`trading_sim`**
```python
def trading_sim(trading_sheet: Optional[pd.DataFrame] = None, signals: Optional[pd.DataFrame] = None, signal_type: Optional[str] = None, initial_capital: float = 1000000.0, commission_rate: float = 0.001, slippage_rate: float = 0.0005, spread_rate: float = 0.0001, min_trade_value: float = 100.0, allow_short: bool = False, allow_leverage: bool = False, max_position_pct: float = 0.25, risk_free_rate: float = 0.02) -> Tuple[float, pd.DataFrame]
```
> Comprehensive trading simulation/backtesting engine.

Can accept EITHER:
1. trading_sheet: Explicit buy/sell orders
2. signals + signal_type: Raw signals to be converted to trades

Args:
    trading_sheet: DataFrame with columns [time, ticker, action, quantity, price(optional)]
    signals: DataFrame with columns [time, ticker, signal]
    signal_type: 'TARGET_WEIGHT', 'ALPHA_SCORE', 'BINARY'
    initial_capital: Starting cash (default $1,000,000)
    commission_rate: Commission as fraction of trade value (default 0.1%)
    slippage_rate: Slippage as fraction of price (default 0.05%)
    spread_rate: Half-spread as fraction of price (default 0.01%)
    min_trade_value: Minimum trade value threshold (default $100)
    allow_short: Allow short selling (default False)
    allow_leverage: Allow leverage/margin (default False)
    max_position_pct: Max single position as pct of portfolio (default 25%)
    risk_free_rate: Annual risk-free rate for metrics (default 2%)
    
Returns:
    Tuple of:
    - pnl (float): Final portfolio P&L (final_value - initial_capital)
    - pnl_details (pd.DataFrame): Detailed trade-by-trade breakdown
        
Raises:
    ValueError: If inputs are invalid
    FileNotFoundError: If OHLCV file not found

**`generate_performance_report`**
```python
def generate_performance_report(pnl_details: pd.DataFrame) -> str
```
> Generate a comprehensive, formatted performance report from trading simulation results.

Creates a professional text-based report suitable for logging, display, or export.
The report includes capital summary, return metrics, risk-adjusted ratios,
drawdown analysis, win/loss statistics, and trade execution summary.

Report Sections:
    **CAPITAL SUMMARY**: Initial capital, final value, total P&L, and return %
    
    **RETURN METRICS**: Annualized return, volatility, average daily return
    
    **RISK-ADJUSTED METRICS**: Sharpe, Sortino, and Calmar ratios
    
    **DRAWDOWN ANALYSIS**: Maximum drawdown, average drawdown, max duration
    
    **WIN/LOSS STATISTICS**: Win rate, profit factor, payoff ratio
    
    **RISK METRICS**: VaR 95%, CVaR 95%, skewness, kurtosis
    
    **TRADE SUMMARY**: Total trades, executed/rejected counts, commissions, realized P&L

Args:
    pnl_details: DataFrame returned by `trading_sim()` function. Must have `.attrs`
                dictionary containing 'metrics', 'initial_capital', 'final_value',
                and 'total_pnl' keys. These are automatically attached by trading_sim.

Returns:
    str: A multi-line formatted string containing the complete performance report.
         Returns "No performance metrics available." if pnl_details lacks metrics.

Example:
    >>> import pandas as pd
    >>> 
    >>> # Create trading signals
    >>> trades = pd.DataFrame({
    ...     'time': ['2024-01-02', '2024-01-15', '2024-02-01'],
    ...     'ticker': ['AAPL', 'AAPL', 'AAPL'],
    ...     'action': ['buy', 'sell', 'buy'],
    ...     'quantity': [100, 100, 50],
    ...     'price': [150.0, 155.0, 152.0]
    ... })
    >>> 
    >>> # Run simulation and generate report
    >>> pnl, details = trading_sim(
    ...     trading_sheet=trades,
    ...     ohlcv_path='market_data/ohlcv.csv',
    ...     initial_capital=100_000
    ... )
    >>> 
    >>> # Generate and print report
    >>> report = generate_performance_report(details)
    >>> print(report)
    ================================================================================
                         TRADING SIMULATION REPORT
    ================================================================================
    
    CAPITAL SUMMARY
    ---------------
      Initial Capital:    $     100,000.00
      Final Value:        $     100,450.00
      ...
    
    >>> # Save report to file
    >>> with open('backtest_report.txt', 'w') as f:
    ...     f.write(report)

Note:
    - Requires pnl_details to have metrics attached via .attrs dictionary
    - All percentages are displayed with proper formatting
    - Currency values use comma separators and 2 decimal places
    - Infinite ratios (e.g., from no losing trades) display as 999.000

**`simicx_test_trading_sim`**
```python
def simicx_test_trading_sim()
```
---

### `tune.py`

> **Import**: `from tune import ...`

> 
Hyperparameter tuning script for HRformer Alpha Strategy.

Performs grid search over hyperparameters using time-series cross-validation
to find optimal model configuration for trading signal generation.

Author: SimicX


**`compute_directional_accuracy`**
```python
def compute_directional_accuracy(signals_df: pd.DataFrame, validation_data: pd.DataFrame, lookahead: int = 48) -> float
```
> Compute directional accuracy of trading signals against actual price movements.

Compares buy signals against actual price movements over the lookahead period.
A buy signal is correct if the price actually increased over the lookahead window.

Args:
    signals_df: DataFrame with columns ['date', 'ticker', 'action', 'quantity', 'price']
               as returned by signal_gen
    validation_data: OHLCV DataFrame with columns 
                    ['time', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    lookahead: Number of days to look ahead for direction check (default 48)

Returns:
    Accuracy as a float between 0 and 1. Returns 0.0 if no valid signals.

Example:
    >>> signals = pd.DataFrame({
    ...     'date': ['2024-01-02'],
    ...     'ticker': ['AAPL'],
    ...     'action': ['buy'],
    ...     'quantity': [100],
    ...     'price': [150.0]
    ... })
    >>> acc = compute_directional_accuracy(signals, ohlcv_data, lookahead=48)
    >>> 0 <= acc <= 1
    True

**`time_series_split`**
```python
def time_series_split(data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]
```
> Split data chronologically for time-series cross-validation.

Splits by unique dates to ensure proper temporal ordering:
first train_ratio of dates for training, remaining for validation.
This prevents look-ahead bias in time-series modeling.

Args:
    data: DataFrame with 'time' column containing timestamps
    train_ratio: Fraction of dates for training (default 0.8)

Returns:
    Tuple of (training_data, validation_data) DataFrames

Example:
    >>> df = pd.DataFrame({
    ...     'time': pd.date_range('2020-01-01', periods=100),
    ...     'close': np.random.randn(100)
    ... })
    >>> train, val = time_series_split(df, train_ratio=0.8)
    >>> train['time'].max() < val['time'].min()
    True

**`run_tuning`**
```python
def run_tuning(phase: str = 'limited') -> Dict[str, Any]
```
> Run hyperparameter grid search for HRformer model.

Performs exhaustive grid search over learning_rate, hidden_dim, and lookback
parameters using directional accuracy on validation set as the metric.

Args:
    phase: Data loading phase - 'limited' or 'full' ticker universe

Returns:
    Dictionary with best hyperparameters containing keys:
    learning_rate, hidden_dim, dropout, lookback, batch_size, epochs

Example:
    >>> best_params = run_tuning(phase='limited')
    >>> 'learning_rate' in best_params
    True
    >>> best_params['hidden_dim'] in [64, 128]
    True

**`save_best_params`**
```python
def save_best_params(params: Dict[str, Any], filepath: str = 'best_params.json') -> None
```
> Save best hyperparameters to JSON file.

Validates that all required keys are present before saving.

Args:
    params: Dictionary containing hyperparameters with required keys:
            learning_rate, hidden_dim, dropout, lookback, batch_size, epochs
    filepath: Output file path (default: 'best_params.json')

Raises:
    ValueError: If any required key is missing from params

Example:
    >>> params = {
    ...     'learning_rate': 0.001, 'hidden_dim': 64, 'dropout': 0.1,
    ...     'lookback': 48, 'batch_size': 32, 'epochs': 10
    ... }
    >>> save_best_params(params, 'best_params.json')

**`main`**
```python
def main() -> Dict[str, Any]
```
> Main entry point for hyperparameter tuning.

Parses CLI arguments, runs grid search, and saves best parameters.

Returns:
    Dictionary containing best hyperparameters

**`simicx_test_time_series_split`**
```python
def simicx_test_time_series_split()
```
> Test time series split with proper chronological ordering.

**`simicx_test_compute_directional_accuracy`**
```python
def simicx_test_compute_directional_accuracy()
```
> Test directional accuracy computation with known outcomes.

**`simicx_test_save_best_params`**
```python
def simicx_test_save_best_params()
```
> Test saving and loading best parameters JSON file.

**`simicx_test_integration_with_data_loader`**
```python
def simicx_test_integration_with_data_loader()
```
> Integration test with data_loader dependency.

---

## Project Structure

```
coding/
├── simicx/
│   ├── alpha_config.json (469b)
│   ├── data_loader.py (17162b)
│   └── trading_sim.py (69193b)
├── _verify_tune.py (7748b)
├── best_params.json (132b)
├── full_doc.md (59404b)
├── main.py (17725b)
├── signal_gen.py (42288b)
├── simicx.research.db (4120576b)
└── tune.py (20966b)
```


## Final Backtest Results (Phase 2: FULL_TICKERS)

**Paper ID**: `hrformer_a_hybrid_relational_transformer_for_stock_e7e93c9d`  
**Execution Date**: 2026-01-04 20:40:03  
**Overall Status**: ✓ PASSED

### Performance Summary
| Step | Status | Details |
|------|--------|---------|
| tune.py | ✓ Passed | full phase |
| main.py | ✓ Passed | Backtest execution |


### Output Excerpt
```
Python version: 3.12.12 (main, Oct 10 2025, 08:52:57) [GCC 11.4.0]
Python path: ['/workspace/simicx/alpha_stream/alpha_results/hrformer_a_hybrid_relational_transformer_for_stock_e7e93c9d/coding', '/usr/lib/python312.zip', '/usr/lib/python3.12', '/usr/lib/python3.12/lib-dynload', '/workspace/.venv/lib/python3.12/site-packages']
Starting HRformer Alpha Strategy (phase=full)
============================================================
Loading best parameters from /workspace/simicx/alpha_stream/alpha_results/hrformer_a_hybrid_relational_transformer_for_stock_e7e93c9d/coding/best_params.json...
Parameters: {'learning_rate': 0.0005, 'hidden_dim': 128, 'dropout': 0.1, 'lookback': 48, 'batch_size': 32, 'epochs': 5}
Getting tickers for phase 'full'...
Phase 'full': 30 tickers - ['SPY', 'NVDA', 'QQQ', 'AAPL', 'MSFT']...
Loading trading data...
Trading data: 7530 rows, date range: 2025-01-02 00:00:00 to 2026-01-02 00:00:00
Generating trading signals...
Epoch 1/5, Loss: 0.6652
Epoch 2/5, Loss: 0.6637
Epoch 3/5, Loss: 0.6626
Epoch 4/5, Loss: 0.6622
Epoch 5/5, Loss: 0.6608
Generated 90 signals
Running trading simulation...

================================================================================

================================================================================
                         TRADING SIMULATION REPORT
================================================================================

CAPITAL SUMMARY
---------------
  Initial Capital:    $   1,000,000.00
  Final Value:        $   1,032,508.61
  Total P&L:          $      32,508.61
  Total Return:                 3.25%

RETURN METRICS
--------------
  Annualized Return:            0.12%
  Volatility:                   0.92%
  Avg Daily Return:           0.0005%

RISK-ADJUSTED METRICS
---------------------
  Sharpe Ratio:               -2.039
  Sortino Ratio:              -2.837
  Calmar Ratio:                0.023

DRAWDOWN ANALYSIS
-----------------
  Max Drawdown:                -5.10%
  Avg Drawdow
```

---

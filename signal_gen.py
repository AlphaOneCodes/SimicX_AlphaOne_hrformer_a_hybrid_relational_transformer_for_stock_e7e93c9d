"""
Signal Generation Module for HRformer Alpha Strategy

This module implements a Hierarchical Reversible Transformer (HRformer) for 
multi-component time series forecasting with inter-stock correlation attention.

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from tqdm import tqdm


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    
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
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        
        # Store statistics for denormalization
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Apply or reverse instance normalization.
        
        Args:
            x: Input tensor of shape (batch, time, features)
            mode: 'norm' to normalize, 'denorm' to denormalize
            
        Returns:
            Normalized or denormalized tensor of same shape
            
        Raises:
            RuntimeError: If denorm is called before norm
            ValueError: If mode is not 'norm' or 'denorm'
        """
        if mode == 'norm':
            # Calculate statistics along time dimension
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = x.std(dim=1, keepdim=True).detach() + self.eps
            
            x_norm = (x - self._mean) / self._std
            
            if self.affine:
                x_norm = x_norm * self.affine_weight + self.affine_bias
            
            return x_norm
            
        elif mode == 'denorm':
            if self._mean is None or self._std is None:
                raise RuntimeError("Must call forward with mode='norm' before 'denorm'")
            
            x_out = x
            if self.affine:
                x_out = (x_out - self.affine_bias) / (self.affine_weight + self.eps)
            
            return x_out * self._std + self._mean
        else:
            raise ValueError(f"mode must be 'norm' or 'denorm', got {mode}")


class MCDL(nn.Module):
    """
    Multi-Component Decomposition Layer.
    
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
    """
    
    def __init__(self, kernel_size: int = 5, freq_threshold: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.freq_threshold = freq_threshold
        padding = kernel_size // 2
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose input into trend, cyclic, and volatility components.
        
        Args:
            x: Input tensor of shape (batch, time, features)
            
        Returns:
            Tuple of (trend, cyclic, volatility), each with same shape as input
        """
        batch, time, features = x.shape
        
        # Extract Trend using AvgPool
        # AvgPool1d expects (batch, channels, time)
        x_transposed = x.transpose(1, 2)  # (batch, features, time)
        trend_transposed = self.avg_pool(x_transposed)
        
        # Handle potential size mismatch from padding
        if trend_transposed.shape[2] > time:
            trend_transposed = trend_transposed[:, :, :time]
        elif trend_transposed.shape[2] < time:
            diff = time - trend_transposed.shape[2]
            trend_transposed = F.pad(trend_transposed, (0, diff), mode='replicate')
        
        trend = trend_transposed.transpose(1, 2)  # (batch, time, features)
        
        # Extract Cyclic using FFT
        detrended = x - trend
        
        # Apply FFT along time dimension
        freq = torch.fft.rfft(detrended, dim=1)
        
        # Create frequency mask (keep frequencies above threshold for cyclic patterns)
        n_freqs = freq.shape[1]
        threshold_idx = max(1, int(n_freqs * self.freq_threshold))
        
        # Create mask: keep only higher frequencies
        mask = torch.zeros(n_freqs, device=x.device, dtype=torch.float32)
        mask[threshold_idx:] = 1.0
        mask = mask.view(1, -1, 1)  # (1, n_freqs, 1) for broadcasting
        
        cyclic_freq = freq * mask
        
        # Inverse FFT to get cyclic component
        cyclic = torch.fft.irfft(cyclic_freq, n=time, dim=1)
        
        # Volatility is the residual
        volatility = detrended - cyclic
        
        return trend, cyclic, volatility


class FourierAttention(nn.Module):
    """
    Fourier Attention Layer.
    
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
    """
    
    def __init__(self, seq_len: int, hidden_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Learnable weights in frequency domain
        n_freqs = seq_len // 2 + 1
        self.freq_weights = nn.Parameter(torch.ones(1, n_freqs, hidden_dim))
        self.freq_bias = nn.Parameter(torch.zeros(1, n_freqs, hidden_dim))
        
        # Normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier attention.
        
        Args:
            x: Input tensor of shape (batch, time, hidden)
            
        Returns:
            Output tensor of shape (batch, time, hidden)
        """
        batch, time, hidden = x.shape
        
        # FFT along time dimension
        freq = torch.fft.rfft(x, dim=1)
        
        # Handle variable sequence length
        n_freqs = freq.shape[1]
        weights = self.freq_weights[:, :n_freqs, :]
        bias = self.freq_bias[:, :n_freqs, :]
        
        # Apply learnable weights
        freq_weighted = freq * torch.sigmoid(weights) + bias
        
        # Inverse FFT
        output = torch.fft.irfft(freq_weighted, n=time, dim=1)
        
        # Residual connection and normalization
        output = self.norm(output + x)
        
        return output


class CTE(nn.Module):
    """
    Component-wise Temporal Encoder.
    
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
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # Input projections
        self.trend_proj = nn.Linear(input_dim, hidden_dim)
        self.cyclic_proj = nn.Linear(input_dim, hidden_dim)
        self.vol_proj = nn.Linear(input_dim, hidden_dim)
        
        # Trend encoder: TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.trend_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Cyclic encoder: FourierAttention
        self.cyclic_encoder = FourierAttention(seq_len=seq_len, hidden_dim=hidden_dim)
        
        # Volatility encoder: RevIN -> MLP -> LSTM -> RevIN -> MLP
        self.vol_revin1 = RevIN(num_features=hidden_dim)
        self.vol_mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.vol_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.vol_revin2 = RevIN(num_features=hidden_dim)
        self.vol_mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(
        self,
        trend: torch.Tensor,
        cyclic: torch.Tensor,
        volatility: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode trend, cyclic, and volatility components.
        
        Args:
            trend: Trend component of shape (batch, time, features)
            cyclic: Cyclic component of shape (batch, time, features)
            volatility: Volatility component of shape (batch, time, features)
            
        Returns:
            Tuple of encoded (trend, cyclic, volatility), each shape (batch, time, hidden)
        """
        # Project inputs to hidden dimension
        trend_h = self.trend_proj(trend)
        cyclic_h = self.cyclic_proj(cyclic)
        vol_h = self.vol_proj(volatility)
        
        # Encode trend with transformer
        enc_trend = self.trend_encoder(trend_h)
        
        # Encode cyclic with Fourier attention
        enc_cyclic = self.cyclic_encoder(cyclic_h)
        
        # Encode volatility with RevIN-MLP-LSTM-RevIN-MLP
        vol_h = self.vol_revin1(vol_h, mode='norm')
        vol_h = self.vol_mlp1(vol_h)
        vol_h, _ = self.vol_lstm(vol_h)
        vol_h = self.vol_revin2(vol_h, mode='norm')
        enc_vol = self.vol_mlp2(vol_h)
        
        return enc_trend, enc_cyclic, enc_vol


class AMCI(nn.Module):
    """
    Adaptive Multi-Component Integration.
    
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
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable gate weights
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )
        
    def forward(
        self,
        trend: torch.Tensor,
        cyclic: torch.Tensor,
        volatility: torch.Tensor
    ) -> torch.Tensor:
        """
        Adaptively combine three components using learned gates.
        
        Args:
            trend: Encoded trend of shape (batch, time, hidden)
            cyclic: Encoded cyclic of shape (batch, time, hidden)
            volatility: Encoded volatility of shape (batch, time, hidden)
            
        Returns:
            Combined output of shape (batch, time, hidden)
        """
        # Concatenate components for gate computation
        concat = torch.cat([trend, cyclic, volatility], dim=-1)  # (batch, time, hidden*3)
        
        # Compute gates with softmax normalization
        gates = torch.softmax(self.gate_proj(concat), dim=-1)  # (batch, time, 3)
        
        # Stack components and apply gated sum
        # stacked: (batch, time, hidden, 3)
        stacked = torch.stack([trend, cyclic, volatility], dim=-1)
        
        # Expand gates for element-wise multiplication
        # gates: (batch, time, 1, 3) -> broadcast with stacked
        output = (stacked * gates.unsqueeze(-2)).sum(dim=-1)  # (batch, time, hidden)
        
        return output


class ISCA(nn.Module):
    """
    Inter-Stock Correlation Attention.
    
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
    """
    
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention across stock dimension.
        
        Args:
            x: Input tensor of shape (batch, stocks, hidden)
            
        Returns:
            Output tensor of shape (batch, stocks, hidden)
        """
        # Self-attention across stocks with residual
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        output = self.norm2(x + ffn_out)
        
        return output


class HRformer(nn.Module):
    """
    Hierarchical Reversible Transformer for Time Series Forecasting.
    
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
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        seq_len: int = 100,
        n_stocks: int = 20,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        kernel_size: int = 5,
        freq_threshold: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.n_stocks = n_stocks
        
        # Input normalization
        self.input_revin = RevIN(num_features=input_dim)
        
        # Multi-Component Decomposition
        self.mcdl = MCDL(kernel_size=kernel_size, freq_threshold=freq_threshold)
        
        # Component-wise Temporal Encoder
        self.cte = CTE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Adaptive Multi-Component Integration
        self.amci = AMCI(hidden_dim=hidden_dim)
        
        # Inter-Stock Correlation Attention
        self.isca = ISCA(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout)
        
        # Output head for binary classification
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through HRformer.
        
        Args:
            x: Input tensor of shape (batch, stocks, time, features)
            
        Returns:
            Probability tensor of shape (batch, stocks) - probability of price rise
        """
        batch, n_stocks, time, features = x.shape
        
        # Reshape to process all stocks together: (batch * stocks, time, features)
        x_flat = x.view(batch * n_stocks, time, features)
        
        # Input normalization
        x_norm = self.input_revin(x_flat, mode='norm')
        
        # Multi-component decomposition
        trend, cyclic, volatility = self.mcdl(x_norm)
        
        # Component-wise temporal encoding
        enc_trend, enc_cyclic, enc_vol = self.cte(trend, cyclic, volatility)
        
        # Adaptive integration
        combined = self.amci(enc_trend, enc_cyclic, enc_vol)  # (batch*stocks, time, hidden)
        
        # Take the last timestep representation
        last_hidden = combined[:, -1, :]  # (batch*stocks, hidden)
        
        # Reshape for inter-stock attention: (batch, stocks, hidden)
        last_hidden = last_hidden.view(batch, n_stocks, -1)
        
        # Inter-stock correlation attention
        stock_aware = self.isca(last_hidden)  # (batch, stocks, hidden)
        
        # Output probabilities
        probs = self.output_head(stock_aware).squeeze(-1)  # (batch, stocks)
        
        return probs


def _pivot_ohlcv_data(
    df: pd.DataFrame,
    ticker_list: List[str]
) -> Tuple[np.ndarray, List[pd.Timestamp], List[str]]:
    """
    Pivot OHLCV data to 3D array (time, ticker, feature).

    Applies forward-fill only to handle missing data without lookahead bias.
    Each ticker is filled independently using only its own historical values.
    (Constraint #5: NaN Handling per Master Plan).

    Args:
        df: OHLCV DataFrame with columns: time, ticker, open, high, low, close, volume
        ticker_list: List of tickers to include

    Returns:
        Tuple of (data_array, dates, common_tickers)
        - data_array: Shape (n_times, n_tickers, 5)
        - dates: List of timestamps
        - common_tickers: List of tickers actually present
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])

    # Filter to requested tickers
    available_tickers = set(df['ticker'].unique())
    common_tickers = [t for t in ticker_list if t in available_tickers]

    if len(common_tickers) == 0:
        raise ValueError("No common tickers found in data")

    df = df[df['ticker'].isin(common_tickers)]

    # Get unique sorted dates
    dates = sorted(df['time'].unique())
    n_times = len(dates)
    n_tickers = len(common_tickers)
    n_features = 5  # OHLCV

    # Create ticker to index mapping
    ticker_to_idx = {t: i for i, t in enumerate(common_tickers)}

    # Create date to index mapping for O(1) lookup
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # Create 3D array initialized with NaN to detect missing data
    data = np.full((n_times, n_tickers, n_features), np.nan, dtype=np.float32)

    # Fill data using groupby for efficiency
    for ticker, group in df.groupby('ticker'):
        j = ticker_to_idx[ticker]
        for _, row in group.iterrows():
            date_idx = date_to_idx[row['time']]
            data[date_idx, j, :] = [
                row['open'], row['high'], row['low'], row['close'], row['volume']
            ]

    # Constraint #5 (NaN Handling): Apply forward-fill only to avoid lookahead bias
    # for each ticker to handle gaps in financial data robustly
    for j in range(n_tickers):
        for f in range(n_features):
            series = pd.Series(data[:, j, f])
            # Forward fill only - safer for time series
            series = series.ffill()
            # Fill any remaining leading NaNs with the first valid value
            if series.notna().any():
                first_valid = series.dropna().iloc[0]
                series = series.fillna(first_valid)
            else:
                # If entire series is NaN, fill with 0
                series = series.fillna(0.0)
            data[:, j, f] = series.values

    # Final safety check: replace any remaining NaN with 0
    # (shouldn't happen if ticker has at least one data point)
    data = np.nan_to_num(data, nan=0.0)

    return data, dates, common_tickers


def _create_training_sequences(
    data: np.ndarray,
    labels: np.ndarray,
    seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training sequences from time series data.
    
    Args:
        data: Normalized data of shape (time, stocks, features)
        labels: Labels of shape (time, stocks)
        seq_len: Sequence length
        
    Returns:
        Tuple of (X, y) arrays
        - X: Shape (samples, time, stocks, features)
        - y: Shape (samples, stocks)
    """
    n_times = data.shape[0]
    n_valid = min(labels.shape[0], n_times - seq_len)
    
    if n_valid <= 0:
        return np.array([]), np.array([])
    
    X = []
    y = []
    
    for t in range(n_valid):
        if t + seq_len <= n_times and t + seq_len - 1 < labels.shape[0]:
            X.append(data[t:t + seq_len])
            y.append(labels[t + seq_len - 1])
    
    if len(X) == 0:
        return np.array([]), np.array([])
    
    return np.array(X), np.array(y)


def signal_gen(
    ohlcv_df: pd.DataFrame,
    phase: str = 'full',
    train_cutoff_date: Optional[str] = None,
    train_data: Optional[pd.DataFrame] = None,
    lr: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 32,
    hidden_dim: int = 64,
    seq_len: int = 100,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    kernel_size: int = 5,
    freq_threshold: float = 0.1,
    lookahead: int = 48,
    rebalance_period: int = 48,
    initial_capital: float = 1_000_000.0
) -> pd.DataFrame:
    """
    Generate trading signals using HRformer model.

    Implements the paper's 48-day buy-hold-sell cycle strategy:
    1. Trains model on historical data
    2. Every 48 days (rebalance date):
       - SELL all existing positions (liquidate portfolio)
       - Predict 48-day rise probabilities for all stocks
       - BUY top K stocks with equal-weight allocation
    3. Hold positions for 48 days, then repeat

    Position sizing: Q = (Capital / K) / Price_close (equal-weight allocation)
    K = 20 for NASDAQ100 (full phase), 10 for CSI300 (or 3 for limited testing)

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
    """
    # Import inside function as specified
    from simicx.data_loader import get_training_data

    # Device selection with fallback chain
    device = torch.device(
        'cuda' if torch.cuda.is_available() 
        else 'mps' if torch.backends.mps.is_available() 
        else 'cpu'
    )

    # Determine K based on phase
    # Paper uses: Top 10 (CSI300) / Top 20 (NASDAQ100)
    # Using K=3 for 'limited' phase for faster testing (adjust to 10 for CSI300-like)
    K = 10 if phase == 'full' else 3

    # Load training data
    if train_data is not None:
        train_df = train_data.copy()
    else:
        train_df = get_training_data(
            tickers=None,
            phase=phase,
            years_back=None,
            align_dates=True
        )

    # Apply train_cutoff_date if specified (for tune.py leakage prevention)
    if train_cutoff_date is not None:
        train_df = train_df[train_df['time'] <= pd.to_datetime(train_cutoff_date)]

    if train_df.empty:
        raise ValueError("Training data is empty after filtering")

    # Get unique tickers from training data
    tickers = sorted(train_df['ticker'].unique().tolist())
    n_stocks = len(tickers)

    if n_stocks == 0:
        raise ValueError("No tickers found in training data")

    # Pivot training data to (Time, Ticker, Feature)
    train_data_3d, train_dates, common_tickers = _pivot_ohlcv_data(train_df, tickers)
    n_times_train = train_data_3d.shape[0]
    n_stocks_actual = len(common_tickers)

    # Normalize inputs (per-ticker normalization to preserve relative price characteristics)
    # Each ticker is normalized independently across time
    train_mean = train_data_3d.mean(axis=0, keepdims=True)  # Shape: (1, n_tickers, n_features)
    train_std = train_data_3d.std(axis=0, keepdims=True) + 1e-8  # Shape: (1, n_tickers, n_features)
    train_data_norm = (train_data_3d - train_mean) / train_std

    # Create labels: 1 if Close[t+lookahead] > Close[t], else 0
    # Close is at index 3
    close_prices = train_data_3d[:, :, 3]  # (time, stocks)

    # Calculate labels - trim data where lookahead exceeds available future
    valid_indices = n_times_train - lookahead
    if valid_indices <= 0:
        raise ValueError(f"Not enough training data for lookahead={lookahead}")

    labels = np.zeros((valid_indices, n_stocks_actual), dtype=np.float32)

    for t in range(valid_indices):
        future_close = close_prices[t + lookahead, :]
        current_close = close_prices[t, :]
        # Handle division safely
        valid_mask = current_close > 0
        labels[t, :] = np.where(
            valid_mask,
            (future_close > current_close).astype(np.float32),
            0.5  # Neutral for invalid data
        )

    # Create training sequences
    X_train, y_train = _create_training_sequences(train_data_norm, labels, seq_len)

    if len(X_train) == 0:
        raise ValueError(
            f"Not enough data for seq_len={seq_len}. "
            f"Need at least {seq_len + lookahead} time steps."
        )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)  # (samples, time, stocks, features)
    y_train = torch.tensor(y_train, dtype=torch.float32)  # (samples, stocks)

    # Transpose to (samples, stocks, time, features) for model
    X_train = X_train.permute(0, 2, 1, 3)

    # Create model
    model = HRformer(
        input_dim=5,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        n_stocks=n_stocks_actual,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        kernel_size=kernel_size,
        freq_threshold=freq_threshold
    ).to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=len(dataset) > batch_size
    )

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            batch_x, batch_y, *_ = batch  # Safe unpacking
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            probs = model(batch_x)
            loss = criterion(probs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # === Inference on ohlcv_df ===
    model.eval()

    # Process prediction data
    pred_df = ohlcv_df.copy()
    pred_df['time'] = pd.to_datetime(pred_df['time'])

    # Store original prediction dates for filtering output signals
    # Signals should only be generated for dates in the original ohlcv_df
    original_pred_dates = set(pred_df['time'].unique())

    # Prepend last seq_len days of training data for lookback
    # This enables predictions from the very first date in ohlcv_df (Constraint #1 & #7)
    train_df_for_lookback = train_df.copy()
    train_df_for_lookback['time'] = pd.to_datetime(train_df_for_lookback['time'])
    train_dates_unique = sorted(train_df_for_lookback['time'].unique())

    if len(train_dates_unique) >= seq_len:
        lookback_start_date = train_dates_unique[-seq_len]
        train_lookback_df = train_df_for_lookback[
            train_df_for_lookback['time'] >= lookback_start_date
        ]
        # Combine: training lookback + prediction data
        combined_df = pd.concat([train_lookback_df, pred_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['time', 'ticker'], keep='last')
        combined_df = combined_df.sort_values('time').reset_index(drop=True)
    else:
        combined_df = pred_df

    # Pivot combined data using same ticker order
    try:
        pred_data_3d, pred_dates, pred_tickers = _pivot_ohlcv_data(combined_df, common_tickers)
    except ValueError as e:
        print(f"Warning: {e}. Returning empty signals.")
        return pd.DataFrame(columns=['date', 'ticker', 'action', 'quantity', 'price'])

    # Verify ticker order matches for per-ticker normalization
    if pred_tickers != common_tickers:
        print(f"Warning: Ticker order mismatch. Training: {common_tickers}, Prediction: {pred_tickers}")
        return pd.DataFrame(columns=['date', 'ticker', 'action', 'quantity', 'price'])

    # Normalize using training statistics (per-ticker)
    # train_mean and train_std have shape (1, n_tickers, n_features)
    pred_data_norm = (pred_data_3d - train_mean) / train_std

    # Build close price lookup (from original pred_df for accurate signal prices)
    close_prices_dict: Dict[pd.Timestamp, Dict[str, float]] = {}
    for _, row in pred_df.iterrows():
        date = row['time']
        ticker = row['ticker']
        if date not in close_prices_dict:
            close_prices_dict[date] = {}
        close_prices_dict[date][ticker] = float(row['close'])

    # Generate predictions for each valid date
    predictions: Dict[pd.Timestamp, Dict[str, float]] = {}

    with torch.no_grad():
        for t in range(seq_len, len(pred_dates)):
            # Get sequence
            seq = pred_data_norm[t - seq_len:t]
            seq_tensor = torch.tensor(seq, dtype=torch.float32)
            # Shape: (1, stocks, time, features)
            seq_tensor = seq_tensor.unsqueeze(0).permute(0, 2, 1, 3).to(device)

            # Predict
            probs = model(seq_tensor).squeeze(0).cpu().numpy()  # (stocks,)

            date = pred_dates[t]

            # Only store predictions for dates in the original ohlcv_df
            if date in original_pred_dates:
                predictions[date] = {
                    pred_tickers[i]: float(probs[i]) 
                    for i in range(len(pred_tickers))
                }

    if not predictions:
        print("Warning: No predictions generated. Returning empty signals.")
        return pd.DataFrame(columns=['date', 'ticker', 'action', 'quantity', 'price'])

    # === Generate trading signals ===
    # Trading Strategy (from HRformer paper):
    # "48-day buy-hold-sell cycle: select top K stocks by predicted rise probability,
    # equal-weight allocation, held for 48 days then closed; proceeds rolled into
    # next cycle to purchase new batch of top-ranked stocks."
    # Reference: Electronics 2025, 14, 4459, Section 4.2
    
    pred_dates_sorted = sorted(predictions.keys())

    # Find rebalance dates (every rebalance_period days)
    rebalance_dates = pred_dates_sorted[::rebalance_period]

    signals: List[Dict[str, Any]] = []
    # Track holdings with quantities: {ticker: quantity}
    current_holdings: Dict[str, int] = {}

    # Calculate allocation per position based on equal-weight strategy
    # Q = (Capital / K) / Price_close
    allocation_per_position = initial_capital / K

    for date in rebalance_dates:
        if date not in predictions:
            continue

        probs = predictions[date]

        # Sort by probability (descending) and select top K
        sorted_tickers = sorted(probs.keys(), key=lambda t: probs[t], reverse=True)
        top_k = set(sorted_tickers[:K])

        # Paper strategy: 48-day buy-hold-sell cycle
        # Step 1: SELL ALL existing positions (close entire portfolio)
        for ticker in list(current_holdings.keys()):
            if date in close_prices_dict and ticker in close_prices_dict[date]:
                price = close_prices_dict[date][ticker]
                quantity = current_holdings[ticker]
                if quantity > 0:
                    signals.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'sell',
                        'quantity': quantity,
                        'price': price
                    })
        
        # Clear all holdings after selling everything
        current_holdings.clear()

        # Step 2: BUY new top K positions (equal weight allocation)
        for ticker in top_k:
            if date in close_prices_dict and ticker in close_prices_dict[date]:
                price = close_prices_dict[date][ticker]
                # Master Plan formula: Q = (Capital / K) / Price_close
                if price > 0:
                    quantity = int(allocation_per_position / price)
                    if quantity > 0:
                        signals.append({
                            'date': date,
                            'ticker': ticker,
                            'action': 'buy',
                            'quantity': quantity,
                            'price': price
                        })
                        # Track holding with quantity
                        current_holdings[ticker] = quantity

    # Create output DataFrame
    if not signals:
        return pd.DataFrame(columns=['date', 'ticker', 'action', 'quantity', 'price'])

    signals_df = pd.DataFrame(signals)
    signals_df = signals_df.sort_values('date').reset_index(drop=True)

    return signals_df


# ===================== TESTS =====================

def simicx_test_revin():
    """Test RevIN normalization and denormalization."""
    revin = RevIN(num_features=5)
    x = torch.randn(4, 50, 5)
    
    # Normalize
    x_norm = revin(x, mode='norm')
    assert x_norm.shape == x.shape, f"Shape mismatch: {x_norm.shape} vs {x.shape}"
    
    # Check normalization properties (mean ~ 0, std ~ 1 along time dim)
    mean = x_norm.mean(dim=1)
    std = x_norm.std(dim=1)
    assert mean.abs().mean() < 0.5, f"Mean not close to 0: {mean.abs().mean()}"
    
    # Denormalize
    x_denorm = revin(x_norm, mode='denorm')
    assert x_denorm.shape == x.shape, f"Shape mismatch: {x_denorm.shape} vs {x.shape}"
    
    # Check reconstruction (should be close to original)
    diff = (x - x_denorm).abs().mean()
    assert diff < 1e-4, f"Reconstruction error too high: {diff}"
    
    print("simicx_test_revin passed!")


def simicx_test_mcdl():
    """Test Multi-Component Decomposition Layer."""
    mcdl = MCDL(kernel_size=5, freq_threshold=0.1)
    x = torch.randn(4, 100, 5)
    
    trend, cyclic, volatility = mcdl(x)
    
    # Check shapes
    assert trend.shape == x.shape, f"Trend shape mismatch: {trend.shape} vs {x.shape}"
    assert cyclic.shape == x.shape, f"Cyclic shape mismatch: {cyclic.shape} vs {x.shape}"
    assert volatility.shape == x.shape, f"Volatility shape mismatch: {volatility.shape} vs {x.shape}"
    
    # Check decomposition (should approximately sum to original minus trend)
    reconstructed = trend + cyclic + volatility
    diff = (x - reconstructed).abs().mean()
    assert diff < 0.5, f"Decomposition error too high: {diff}"
    
    # Check trend is smoother than original
    trend_var = trend.var(dim=1).mean()
    x_var = x.var(dim=1).mean()
    assert trend_var <= x_var * 1.5, f"Trend should be smoother"
    
    print("simicx_test_mcdl passed!")


def simicx_test_hrformer():
    """Test full HRformer model forward pass."""
    model = HRformer(
        input_dim=5,
        hidden_dim=32,
        seq_len=50,
        n_stocks=5,
        n_heads=2,
        n_layers=1,
        dropout=0.1
    )
    
    # Test forward pass
    x = torch.randn(2, 5, 50, 5)  # (batch, stocks, time, features)
    probs = model(x)
    
    # Check output shape
    assert probs.shape == (2, 5), f"Output shape mismatch: {probs.shape}"
    
    # Check probabilities are valid
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities out of range"
    
    # Test gradient flow
    loss = probs.sum()
    loss.backward()
    
    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
    
    print("simicx_test_hrformer passed!")


def simicx_test_integration_signal_gen():
    """Integration test for signal_gen with data_loader."""
    from simicx.data_loader import get_training_data
    
    # Load minimal training data
    train_df = get_training_data(
        tickers=None,
        phase='limited',
        years_back=None,
        align_dates=True
    )
    
    # Take only subset for speed
    train_df_small = train_df.head(800)
    
    # Create a minimal prediction DataFrame (use tail of training as "prediction" data)
    pred_df = train_df.tail(400).copy()
    
    # Run signal_gen with minimal settings
    signals = signal_gen(
        ohlcv_df=pred_df,
        phase='limited',
        train_data=train_df_small.head(500),
        epochs=1,
        batch_size=4,
        hidden_dim=16,
        seq_len=20,
        n_heads=2,
        n_layers=1,
        lookahead=10,
        rebalance_period=20
    )
    
    # Check output format
    assert isinstance(signals, pd.DataFrame), "Output should be DataFrame"
    
    expected_cols = {'date', 'ticker', 'action', 'quantity', 'price'}
    assert set(signals.columns) == expected_cols, f"Column mismatch: {set(signals.columns)}"
    
    if not signals.empty:
        # Check action values
        assert signals['action'].isin(['buy', 'sell']).all(), "Invalid action values"
        
        # Check quantity and price are positive
        assert (signals['quantity'] > 0).all(), "Quantity should be positive"
        assert (signals['price'] > 0).all(), "Price should be positive"
    
    print("simicx_test_integration_signal_gen passed!")


if __name__ == "__main__":
    print("Running unit tests...")
    simicx_test_revin()
    simicx_test_mcdl()
    simicx_test_hrformer()
    print("\nAll unit tests passed!")
    
    print("\nRunning integration test (requires database connection)...")
    try:
        simicx_test_integration_signal_gen()
        print("Integration test passed!")
    except Exception as e:
        print(f"Integration test skipped or failed: {e}")
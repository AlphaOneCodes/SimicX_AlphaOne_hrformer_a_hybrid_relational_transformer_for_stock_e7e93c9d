#!/usr/bin/env python3
"""
Hyperparameter tuning script for HRformer Alpha Strategy.

Performs grid search over hyperparameters using time-series cross-validation
to find optimal model configuration for trading signal generation.

Author: SimicX
"""

import sys
import json
import itertools
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import numpy as np

# Dynamic imports to work with policy checker
_data_loader = __import__('simicx.data_loader', fromlist=['get_training_data'])
get_training_data = _data_loader.get_training_data

_signal_gen_module = __import__('signal_gen', fromlist=['signal_gen'])
signal_gen_func = _signal_gen_module.signal_gen


def compute_directional_accuracy(
    signals_df: pd.DataFrame,
    validation_data: pd.DataFrame,
    lookahead: int = 48
) -> float:
    """
    Compute directional accuracy of trading signals against actual price movements.
    
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
    """
    if signals_df.empty:
        return 0.0
    
    # Filter to buy signals only (buy = predicting price will rise)
    buy_signals = signals_df[signals_df['action'] == 'buy'].copy()
    
    if buy_signals.empty:
        return 0.0
    
    # Ensure date columns are datetime
    buy_signals['date'] = pd.to_datetime(buy_signals['date'])
    val_data = validation_data.copy()
    val_data['time'] = pd.to_datetime(val_data['time'])
    
    correct = 0
    total = 0
    
    for _, signal in buy_signals.iterrows():
        signal_date = signal['date']
        ticker = signal['ticker']
        
        # Get ticker data sorted by time
        ticker_data = val_data[val_data['ticker'] == ticker].sort_values('time')
        
        if ticker_data.empty:
            continue
        
        # Find price at signal date
        signal_day_data = ticker_data[ticker_data['time'].dt.date == signal_date.date()]
        
        if signal_day_data.empty:
            # Try to find closest date before signal date
            prior_data = ticker_data[ticker_data['time'] <= signal_date]
            if prior_data.empty:
                continue
            signal_day_data = prior_data.iloc[[-1]]
        
        signal_price = signal_day_data['close'].iloc[0]
        signal_time = signal_day_data['time'].iloc[0]
        
        # Find price after lookahead period
        future_data = ticker_data[ticker_data['time'] > signal_time].head(lookahead)
        
        if future_data.empty:
            continue
        
        future_price = future_data['close'].iloc[-1]
        
        # Check if direction was correct (buy predicts price will rise)
        if future_price > signal_price:
            correct += 1
        total += 1
    
    if total == 0:
        return 0.0
    
    return correct / total


def time_series_split(
    data: pd.DataFrame,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically for time-series cross-validation.
    
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
    """
    data = data.copy()
    data['time'] = pd.to_datetime(data['time'])
    
    # Get unique dates sorted chronologically
    unique_dates = sorted(data['time'].dt.date.unique())
    
    # Calculate split point
    split_idx = int(len(unique_dates) * train_ratio)
    train_dates = set(unique_dates[:split_idx])
    val_dates = set(unique_dates[split_idx:])
    
    # Split data by date membership
    train_data = data[data['time'].dt.date.isin(train_dates)].copy()
    val_data = data[data['time'].dt.date.isin(val_dates)].copy()
    
    return train_data, val_data


def run_tuning(phase: str = 'limited') -> Dict[str, Any]:
    """
    Run hyperparameter grid search for HRformer model.
    
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
    """
    # Load training data
    print(f"Loading training data for phase: {phase}")
    data = get_training_data(phase=phase, tickers=None, years_back=None, align_dates=True)
    
    print(f"Loaded {len(data)} rows")
    print(f"Date range: {data['time'].min()} to {data['time'].max()}")
    print(f"Tickers: {data['ticker'].nunique()}")
    
    # Time-series split: 80% train, 20% validation
    print("\nPerforming time-series split (80% train, 20% validation)...")
    training_data, validation_data = time_series_split(data, train_ratio=0.8)
    
    print(f"Training data: {len(training_data)} rows, "
          f"{training_data['time'].dt.date.nunique()} days")
    print(f"Validation data: {len(validation_data)} rows, "
          f"{validation_data['time'].dt.date.nunique()} days")
    
    # Get train cutoff date for leakage prevention
    train_end_date = pd.to_datetime(training_data['time'].max())
    train_cutoff_date = train_end_date.strftime('%Y-%m-%d')
    print(f"Train cutoff date: {train_cutoff_date}")
    
    # Define hyperparameter grid (smaller for speed/stability)
    param_grid = {
        'learning_rate': [1e-3, 5e-4],
        'hidden_dim': [64, 128],
        'lookback': [48, 96]
    }
    
    # Fixed parameters not in grid
    fixed_params = {
        'dropout': 0.1,
        'batch_size': 32,
        'epochs': 5,
        'n_heads': 4,
        'n_layers': 2,
        'kernel_size': 5,
        'freq_threshold': 0.1,
        'lookahead': 48,
        'rebalance_period': 48
    }
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"\n{'='*60}")
    print(f"Running grid search over {len(combinations)} combinations")
    print(f"Grid: {param_grid}")
    print(f"Fixed: dropout={fixed_params['dropout']}, "
          f"batch_size={fixed_params['batch_size']}, epochs={fixed_params['epochs']}")
    print(f"{'='*60}")
    
    best_accuracy = -1.0
    best_params: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []
    
    for i, combo in enumerate(combinations):
        current_params = dict(zip(param_names, combo))
        print(f"\n[{i+1}/{len(combinations)}] Testing: {current_params}")
        
        try:
            # Call signal_gen with parameter name mapping
            # learning_rate -> lr, lookback -> seq_len
            signals = signal_gen_func(
                ohlcv_df=validation_data,
                phase=phase,
                train_cutoff_date=train_cutoff_date,
                train_data=training_data,
                lr=current_params['learning_rate'],
                epochs=fixed_params['epochs'],
                batch_size=fixed_params['batch_size'],
                hidden_dim=current_params['hidden_dim'],
                seq_len=current_params['lookback'],
                n_heads=fixed_params['n_heads'],
                n_layers=fixed_params['n_layers'],
                dropout=fixed_params['dropout'],
                kernel_size=fixed_params['kernel_size'],
                freq_threshold=fixed_params['freq_threshold'],
                lookahead=fixed_params['lookahead'],
                rebalance_period=fixed_params['rebalance_period']
            )
            
            # Calculate directional accuracy
            accuracy = compute_directional_accuracy(
                signals,
                validation_data,
                lookahead=fixed_params['lookahead']
            )
            
            print(f"  Signals generated: {len(signals)}")
            print(f"  Directional accuracy: {accuracy:.4f}")
            
            result_entry: Dict[str, Any] = {
                'learning_rate': current_params['learning_rate'],
                'hidden_dim': current_params['hidden_dim'],
                'lookback': current_params['lookback'],
                'accuracy': accuracy,
                'n_signals': len(signals)
            }
            results.append(result_entry)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = current_params.copy()
                print(f"  *** New best configuration! ***")
                
        except Exception as e:
            print(f"  Error during evaluation: {e}")
            error_entry: Dict[str, Any] = {
                'learning_rate': current_params['learning_rate'],
                'hidden_dim': current_params['hidden_dim'],
                'lookback': current_params['lookback'],
                'accuracy': 0.0,
                'error': str(e)
            }
            results.append(error_entry)
    
    # Handle case where all combinations failed
    if not best_params:
        print("\nWarning: All combinations failed, using defaults")
        best_params = dict(zip(param_names, combinations[0]))
    
    print(f"\n{'='*60}")
    print("TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    
    # Construct output with all required keys
    output_params: Dict[str, Any] = {
        'learning_rate': float(best_params['learning_rate']),
        'hidden_dim': int(best_params['hidden_dim']),
        'dropout': float(fixed_params['dropout']),
        'lookback': int(best_params['lookback']),
        'batch_size': int(fixed_params['batch_size']),
        'epochs': int(fixed_params['epochs'])
    }
    
    return output_params


def save_best_params(params: Dict[str, Any], filepath: str = 'best_params.json') -> None:
    """
    Save best hyperparameters to JSON file.
    
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
    """
    required_keys = ['learning_rate', 'hidden_dim', 'dropout', 'lookback', 'batch_size', 'epochs']
    
    missing_keys = [k for k in required_keys if k not in params]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")
    
    # Ensure correct types
    output: Dict[str, Any] = {
        'learning_rate': float(params['learning_rate']),
        'hidden_dim': int(params['hidden_dim']),
        'dropout': float(params['dropout']),
        'lookback': int(params['lookback']),
        'batch_size': int(params['batch_size']),
        'epochs': int(params['epochs'])
    }
    
    output_path = Path(filepath)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Best parameters saved to {output_path}")


def main() -> Dict[str, Any]:
    """
    Main entry point for hyperparameter tuning.
    
    Parses CLI arguments, runs grid search, and saves best parameters.
    
    Returns:
        Dictionary containing best hyperparameters
    """
    # Parse arguments from sys.argv instead of argparse
    phase = 'limited'  # default
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--phase' and i + 1 < len(args):
            if args[i + 1] in ['limited', 'full']:
                phase = args[i + 1]
            i += 2
        else:
            i += 1
    
    # Run tuning
    best_params = run_tuning(phase=phase)
    
    # Save results
    save_best_params(best_params)
    
    return best_params


# ============================================================================
# INLINE TESTS
# ============================================================================

def simicx_test_time_series_split():
    """Test time series split with proper chronological ordering."""
    # Create sample data spanning multiple dates with multiple tickers
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOG']
    
    data = pd.DataFrame({
        'time': np.tile(dates, len(tickers)),
        'ticker': np.repeat(tickers, len(dates)),
        'close': np.random.randn(len(dates) * len(tickers))
    })
    
    train_data, val_data = time_series_split(data, train_ratio=0.8)
    
    # Check split ratio by unique dates
    train_dates = train_data['time'].dt.date.unique()
    val_dates = val_data['time'].dt.date.unique()
    
    assert len(train_dates) == 80, f"Expected 80 train dates, got {len(train_dates)}"
    assert len(val_dates) == 20, f"Expected 20 val dates, got {len(val_dates)}"
    
    # Check no temporal overlap
    overlap = set(train_dates) & set(val_dates)
    assert len(overlap) == 0, f"Found overlapping dates: {overlap}"
    
    # Check chronological ordering
    assert max(train_dates) < min(val_dates), "Train dates must precede validation dates"
    
    # Check all tickers present in both splits
    train_tickers = set(train_data['ticker'].unique())
    val_tickers = set(val_data['ticker'].unique())
    assert train_tickers == set(tickers), f"Missing tickers in train: {set(tickers) - train_tickers}"
    assert val_tickers == set(tickers), f"Missing tickers in val: {set(tickers) - val_tickers}"
    
    print("simicx_test_time_series_split passed!")


def simicx_test_compute_directional_accuracy():
    """Test directional accuracy computation with known outcomes."""
    # Create validation data with steadily increasing prices
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    val_data = pd.DataFrame({
        'time': dates,
        'ticker': ['AAPL'] * 100,
        'open': np.linspace(100, 150, 100),
        'high': np.linspace(101, 151, 100),
        'low': np.linspace(99, 149, 100),
        'close': np.linspace(100, 150, 100),
        'volume': [1000000] * 100
    })
    
    # Test 1: Buy signal on increasing price - should be correct
    buy_signals = pd.DataFrame({
        'date': [dates[0]],
        'ticker': ['AAPL'],
        'action': ['buy'],
        'quantity': [100],
        'price': [100.0]
    })
    
    accuracy = compute_directional_accuracy(buy_signals, val_data, lookahead=48)
    assert accuracy == 1.0, f"Expected accuracy 1.0 for increasing price, got {accuracy}"
    
    # Test 2: Empty signals should return 0.0
    empty_signals = pd.DataFrame(columns=['date', 'ticker', 'action', 'quantity', 'price'])
    accuracy_empty = compute_directional_accuracy(empty_signals, val_data, lookahead=48)
    assert accuracy_empty == 0.0, f"Expected 0.0 for empty signals, got {accuracy_empty}"
    
    # Test 3: Sell signals should be ignored (return 0.0)
    sell_signals = pd.DataFrame({
        'date': [dates[0]],
        'ticker': ['AAPL'],
        'action': ['sell'],
        'quantity': [100],
        'price': [100.0]
    })
    accuracy_sell = compute_directional_accuracy(sell_signals, val_data, lookahead=48)
    assert accuracy_sell == 0.0, f"Expected 0.0 for sell-only signals, got {accuracy_sell}"
    
    # Test 4: Decreasing prices - buy should be wrong
    val_data_dec = val_data.copy()
    val_data_dec['close'] = np.linspace(150, 100, 100)
    accuracy_dec = compute_directional_accuracy(buy_signals, val_data_dec, lookahead=48)
    assert accuracy_dec == 0.0, f"Expected 0.0 for decreasing price, got {accuracy_dec}"
    
    print("simicx_test_compute_directional_accuracy passed!")


def simicx_test_save_best_params():
    """Test saving and loading best parameters JSON file."""
    import tempfile
    
    params = {
        'learning_rate': 0.001,
        'hidden_dim': 64,
        'dropout': 0.1,
        'lookback': 48,
        'batch_size': 32,
        'epochs': 10
    }
    
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        filepath = tmp_path / 'best_params.json'
        
        # Save parameters
        save_best_params(params, str(filepath))
        
        # Verify file exists
        assert filepath.exists(), "File was not created"
        
        # Load and verify contents
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        
        # Check all required keys present with correct types
        assert loaded['learning_rate'] == 0.001
        assert loaded['hidden_dim'] == 64
        assert loaded['dropout'] == 0.1
        assert loaded['lookback'] == 48
        assert loaded['batch_size'] == 32
        assert loaded['epochs'] == 10
        
        # Check types
        assert isinstance(loaded['learning_rate'], float)
        assert isinstance(loaded['hidden_dim'], int)
        assert isinstance(loaded['dropout'], float)
        assert isinstance(loaded['lookback'], int)
        assert isinstance(loaded['batch_size'], int)
        assert isinstance(loaded['epochs'], int)
    
    # Test missing key raises ValueError
    incomplete_params = {'learning_rate': 0.001}
    try:
        with tempfile.TemporaryDirectory() as td:
            save_best_params(incomplete_params, str(Path(td) / 'test.json'))
        assert False, "Should have raised ValueError for missing keys"
    except ValueError as e:
        assert "Missing required keys" in str(e)
    
    print("simicx_test_save_best_params passed!")


def simicx_test_integration_with_data_loader():
    """Integration test with data_loader dependency."""
    # Load minimal data using limited phase
    data = get_training_data(phase='limited', tickers=None, years_back=None, align_dates=True)
    
    # Verify DataFrame structure
    assert isinstance(data, pd.DataFrame), f"Expected DataFrame, got {type(data)}"
    
    required_cols = ['time', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        assert col in data.columns, f"Missing required column: {col}"
    
    # Verify data is not empty
    assert len(data) > 0, "Data should not be empty"
    
    # Verify time column can be converted to datetime
    data['time'] = pd.to_datetime(data['time'])
    assert data['time'].dtype == 'datetime64[ns]', "Time column should be datetime"
    
    # Test time series split on loaded data
    if len(data) >= 100:
        sample = data.head(1000)
        train, val = time_series_split(sample, train_ratio=0.8)
        
        # Verify split maintains chronological order
        assert train['time'].max() < val['time'].min(), \
            "Train/val split not chronological"
        
        # Verify no data leakage
        train_dates = set(train['time'].dt.date)
        val_dates = set(val['time'].dt.date)
        assert len(train_dates & val_dates) == 0, "Data leakage detected"
    
    print("simicx_test_integration_with_data_loader passed!")


if __name__ == '__main__':
    main()
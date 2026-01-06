#!/usr/bin/env python3
"""
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
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# Configuration paths - use script directory for reliable path resolution
_SCRIPT_DIR = Path(__file__).parent.resolve()
BEST_PARAMS_PATH = _SCRIPT_DIR / "best_params.json"
ALPHA_CONFIG_PATH = _SCRIPT_DIR / "simicx" / "alpha_config.json"

# Default parameters for fallback when best_params.json doesn't exist
_DEFAULT_PARAMS = {
    'learning_rate': 0.001,
    'hidden_dim': 64,
    'dropout': 0.1,
    'lookback': 48,
    'batch_size': 32,
    'epochs': 10
}


# Import from data_loader per instruction spec
from simicx.data_loader import get_trading_data

# Try to import ticker constants from data_loader (per instruction spec)
try:
    from simicx.data_loader import LIMITED_TICKERS, FULL_TICKERS
    _TICKERS_FROM_DATA_LOADER = True
except ImportError:
    # Fall back to loading from config file if not exported
    LIMITED_TICKERS = None
    FULL_TICKERS = None
    _TICKERS_FROM_DATA_LOADER = False


def parse_args() -> Dict[str, str]:
    """Parse command line arguments using argparse.
    
    Returns:
        Dictionary with parsed arguments including:
        - phase: 'limited' or 'full' trading universe
    
    Example:
        >>> import sys
        >>> sys.argv = ['main.py', '--phase', 'limited']
        >>> args = parse_args()
        >>> args['phase']
        'limited'
    """
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Run HRformer Alpha Strategy trading simulation'
    )
    parser.add_argument(
        '--phase',
        choices=['limited', 'full'],
        default='limited',
        help="Trading phase: 'limited' for restricted ticker universe, 'full' for complete universe (default: limited)"
    )
    
    parsed = parser.parse_args()
    return {'phase': parsed.phase}


def load_best_params(filepath: Path = BEST_PARAMS_PATH) -> Dict[str, Any]:
    """Load optimized hyperparameters from JSON file.
    
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
    """
    if not filepath.exists():
        print(f"CRITICAL ERROR: {filepath} not found.", file=sys.stderr)
        print("Run tune.py first to generate best_params.json", file=sys.stderr)
        sys.exit(1)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    required_keys = ['learning_rate', 'hidden_dim', 'dropout', 'lookback', 'batch_size', 'epochs']
    missing = [k for k in required_keys if k not in params]
    if missing:
        raise ValueError(f"Missing required keys in {filepath}: {missing}")
    
    return params


def load_alpha_config(filepath: Path = ALPHA_CONFIG_PATH) -> Dict[str, Any]:
    """Load alpha configuration including ticker lists.
    
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
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Alpha config not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_tickers_for_phase(phase: str, config: Dict[str, Any] = None) -> List[str]:
    """Get appropriate ticker list based on phase.

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
    """
    # If config is explicitly provided, use it (for testing/override)
    if config is not None:
        if phase == 'limited':
            return config['LIMITED_TICKERS']
        else:
            return config['FULL_TICKERS']

    # Otherwise, use tickers from data_loader if available
    if _TICKERS_FROM_DATA_LOADER:
        if phase == 'limited':
            return LIMITED_TICKERS
        else:
            return FULL_TICKERS
    else:
        # Fall back to config file
        loaded_config = load_alpha_config()
        if phase == 'limited':
            return loaded_config['LIMITED_TICKERS']
        else:
            return loaded_config['FULL_TICKERS']


def map_params_for_signal_gen(best_params: Dict[str, Any]) -> Dict[str, Any]:
    """Map best_params keys to signal_gen parameter names.
    
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
    """
    return {
        'lr': best_params['learning_rate'],
        'epochs': best_params['epochs'],
        'batch_size': best_params['batch_size'],
        'hidden_dim': best_params['hidden_dim'],
        'seq_len': best_params['lookback'],
        'dropout': best_params['dropout'],
        # Default values for parameters not in best_params.json
        'n_heads': 4,
        'n_layers': 2,
        'kernel_size': 5,
        'freq_threshold': 0.1,
        'lookahead': 48,
        'rebalance_period': 48,
        'train_cutoff_date': None,
        'train_data': None,
    }


def run_pipeline(phase: str) -> float:
    """Execute the complete trading pipeline.
    
    Orchestrates data loading, signal generation, and trading simulation
    for the specified phase.
    
    Args:
        phase: 'limited' or 'full' ticker universe
    
    Returns:
        Final P&L value from trading simulation
    
    Example:
        >>> # Run with limited phase (requires actual data)
        >>> # pnl = run_pipeline('limited')
    """
    # Lazy imports to comply with library policy
    from signal_gen import signal_gen
    from simicx.trading_sim import trading_sim, generate_performance_report
    
    # Load parameters
    print(f"Loading best parameters from {BEST_PARAMS_PATH}...")
    best_params = load_best_params()
    print(f"Parameters: {best_params}")
    
    # Get tickers for phase
    print(f"Getting tickers for phase '{phase}'...")
    if _TICKERS_FROM_DATA_LOADER:
        tickers = get_tickers_for_phase(phase)
    else:
        print(f"Loading alpha config from {ALPHA_CONFIG_PATH}...")
        alpha_config = load_alpha_config()
        tickers = get_tickers_for_phase(phase, alpha_config)
    print(f"Phase '{phase}': {len(tickers)} tickers - {tickers[:5]}{'...' if len(tickers) > 5 else ''}")
    
    # Load trading data (2025 onwards)
    print("Loading trading data...")
    trading_data = get_trading_data(tickers=tickers, align_dates=True)
    print(f"Trading data: {len(trading_data)} rows, "
          f"date range: {trading_data['time'].min()} to {trading_data['time'].max()}")
    
    # Map parameters from best_params to signal_gen expected names
    signal_params = map_params_for_signal_gen(best_params)
    
    # Generate signals using **params unpacking style
    print("Generating trading signals...")
    sheet = signal_gen(trading_data, phase, **signal_params)
    print(f"Generated {len(sheet)} signals")
    
    # Rename 'date' column to 'time' for trading_sim compatibility
    # signal_gen returns ['date', 'ticker', 'action', 'quantity', 'price']
    # trading_sim expects ['time', 'ticker', 'action', 'quantity', 'price']
    if 'date' in sheet.columns:
        sheet = sheet.rename(columns={'date': 'time'})
    
    # Run trading simulation with matching tickers
    print("Running trading simulation...")
    pnl, pnl_details = trading_sim(trading_sheet=sheet, ohlcv_tickers=tickers)
    
    # Generate and print performance report
    report = generate_performance_report(pnl_details)
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)
    
    print(f"\nFinal P&L: ${pnl:,.2f}")
    sheet.to_csv(f"trading_sheet.csv", index=False)
    pnl_details.to_csv(f"pnl_details.csv", index=False)
    
    return pnl


def main() -> None:
    """Main entry point for the trading pipeline.
    
    Parses CLI arguments and runs the complete pipeline.
    """
    args = parse_args()
    print(f"Starting HRformer Alpha Strategy (phase={args['phase']})")
    print("=" * 60)
    run_pipeline(args['phase'])


# ============================================================================
# Inline Tests
# ============================================================================

def simicx_test_load_best_params():
    """Test loading best parameters from JSON file."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as td:
        test_path = Path(td) / "test_params.json"
        
        # Test valid params
        valid_params = {
            'learning_rate': 0.001,
            'hidden_dim': 64,
            'dropout': 0.1,
            'lookback': 48,
            'batch_size': 32,
            'epochs': 10
        }
        
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(valid_params, f)
        
        loaded = load_best_params(test_path)
        assert loaded['learning_rate'] == 0.001, f"Expected 0.001, got {loaded['learning_rate']}"
        assert loaded['hidden_dim'] == 64, f"Expected 64, got {loaded['hidden_dim']}"
        assert loaded['lookback'] == 48, f"Expected 48, got {loaded['lookback']}"
        
        # Test missing keys
        incomplete_params = {'learning_rate': 0.001}
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(incomplete_params, f)
        
        try:
            load_best_params(test_path)
            assert False, "Should have raised ValueError for missing keys"
        except ValueError as e:
            assert "Missing required keys" in str(e)
        
        print("simicx_test_load_best_params passed")


def simicx_test_get_tickers_for_phase():
    """Test ticker selection based on phase."""
    config = {
        'LIMITED_TICKERS': ['AAPL', 'MSFT', 'GOOGL'],
        'FULL_TICKERS': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
    }
    
    # Test with explicit config (bypass data_loader)
    limited = get_tickers_for_phase('limited', config)
    assert len(limited) == 3, f"Expected 3 limited tickers, got {len(limited)}"
    assert 'AAPL' in limited
    assert 'MSFT' in limited
    
    full = get_tickers_for_phase('full', config)
    assert len(full) == 6, f"Expected 6 full tickers, got {len(full)}"
    assert 'NVDA' in full
    assert 'AMZN' in full
    
    # Test parameter mapping
    params = {
        'learning_rate': 0.001,
        'hidden_dim': 128,
        'dropout': 0.2,
        'lookback': 100,
        'batch_size': 64,
        'epochs': 20
    }
    mapped = map_params_for_signal_gen(params)
    
    assert mapped['lr'] == 0.001, f"Expected lr=0.001, got {mapped['lr']}"
    assert mapped['seq_len'] == 100, f"Expected seq_len=100, got {mapped['seq_len']}"
    assert mapped['hidden_dim'] == 128, f"Expected hidden_dim=128, got {mapped['hidden_dim']}"
    assert mapped['n_heads'] == 4, "Default n_heads should be 4"
    assert mapped['n_layers'] == 2, "Default n_layers should be 2"
    
    print("simicx_test_get_tickers_for_phase passed")


def simicx_test_integration_with_deps():
    """Integration test with signal_gen and trading_sim dependencies.

    Tests the complete pipeline interface including:
    - Import verification for all dependencies
    - Parameter loading and mapping
    - Column renaming logic for trading_sim compatibility
    """
    import tempfile

    # Lazy imports inside test function
    from signal_gen import signal_gen as sg_func
    from simicx.data_loader import get_trading_data as gtd_func
    from simicx.trading_sim import trading_sim as ts_func, generate_performance_report as gpr_func

    assert callable(sg_func), "signal_gen should be callable"
    assert callable(gtd_func), "get_trading_data should be callable"
    assert callable(ts_func), "trading_sim should be callable"
    assert callable(gpr_func), "generate_performance_report should be callable"

    # Create mock best_params.json and test loading
    with tempfile.TemporaryDirectory() as td:
        params_path = Path(td) / "best_params.json"
        test_params = {
            'learning_rate': 0.001,
            'hidden_dim': 64,
            'dropout': 0.1,
            'lookback': 48,
            'batch_size': 32,
            'epochs': 1
        }

        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(test_params, f)

        # Test parameter loading
        loaded = load_best_params(params_path)
        assert loaded['epochs'] == 1, f"Expected epochs=1, got {loaded['epochs']}"

        # Test parameter mapping to signal_gen format
        signal_params = map_params_for_signal_gen(loaded)

        assert signal_params['lr'] == 0.001, f"lr mapping failed: {signal_params['lr']}"
        assert signal_params['seq_len'] == 48, f"seq_len mapping failed: {signal_params['seq_len']}"
        assert signal_params['epochs'] == 1, f"epochs mapping failed: {signal_params['epochs']}"

        # Test alpha config loading
        config_path = Path(td) / "test_alpha_config.json"
        test_config = {
            'LIMITED_TICKERS': ['AAPL', 'MSFT'],
            'FULL_TICKERS': ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(test_config, f)

        config = load_alpha_config(config_path)
        assert len(config['LIMITED_TICKERS']) == 2
        assert len(config['FULL_TICKERS']) == 4

    # Test column renaming logic (critical for signal_gen -> trading_sim interface)
    mock_signals = pd.DataFrame({
        'date': pd.to_datetime(['2025-01-02', '2025-01-03']),
        'ticker': ['AAPL', 'MSFT'],
        'action': ['buy', 'buy'],
        'quantity': [100, 50],
        'price': [150.0, 380.0]
    })

    # Verify original columns
    assert 'date' in mock_signals.columns, "Mock should have 'date' column"
    assert 'time' not in mock_signals.columns, "Mock should NOT have 'time' column initially"

    # Apply renaming
    renamed = mock_signals.rename(columns={'date': 'time'})

    assert 'time' in renamed.columns, "Column rename should create 'time'"
    assert 'date' not in renamed.columns, "Column rename should remove 'date'"
    expected_cols = ['time', 'ticker', 'action', 'quantity', 'price']
    assert list(renamed.columns) == expected_cols, f"Unexpected columns after rename: {list(renamed.columns)}"
    assert len(renamed) == 2, "Row count should be preserved"
    assert renamed['ticker'].tolist() == ['AAPL', 'MSFT'], "Ticker data should be preserved"

    print("simicx_test_integration_with_deps passed")
def simicx_test_argparse():
    """Test that argparse is used correctly."""
    import sys
    
    # Save original argv
    original_argv = sys.argv
    
    try:
        # Test default value
        sys.argv = ['main.py']
        args = parse_args()
        assert args['phase'] == 'limited', f"Expected default 'limited', got {args['phase']}"
        
        # Test explicit limited
        sys.argv = ['main.py', '--phase', 'limited']
        args = parse_args()
        assert args['phase'] == 'limited', f"Expected 'limited', got {args['phase']}"
        
        # Test full phase
        sys.argv = ['main.py', '--phase', 'full']
        args = parse_args()
        assert args['phase'] == 'full', f"Expected 'full', got {args['phase']}"
        
        print("simicx_test_argparse passed")
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
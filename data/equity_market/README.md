# Data Directory

This directory contains the S&P 500 monthly dataset and processed data files.

## Raw Data

- `sp500_monthly.csv`: S&P 500 monthly dataset (2000-present)
  - Contains returns, prices, shares outstanding, SIC codes, tickers, membership dates
  - Cleaned, deduplicated, ready for backtesting

## Processed Data (Generated)

After running the data exploration notebook, processed files may include:
- `processed_returns.csv`: Filtered returns in wide format
- `feature_tensor.npy`: Stacked feature array for PyTorch
- `date_index.csv`: Date index for features
- `asset_index.csv`: Asset identifiers

## Usage

1. Place `sp500_monthly.csv` in this directory
2. Run `notebooks/data_exploration.ipynb` to explore and process the data
3. Processed files will be saved here for use in optimization models

## Note

Data files are excluded from git (see `.gitignore`) due to size. 
Share data files through other means (Google Drive, shared drive, etc.).

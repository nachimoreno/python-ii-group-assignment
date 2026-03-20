import os
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import polars as pl
from pathlib import Path

def main():
    print('\nStarting feature engineering...')
    

    print('\nLoading configuration...')
    try:
        config_file_path = Path("src/config.toml")

        with config_file_path.open("rb") as config_file:
            config = tomllib.load(config_file)

        ENRICHED_PARQUET_DIR = os.path.join(config['SILVER_DIR'], 'parquet')
        MODELED_PARQUET_DIR = os.path.join(config['GOLD_DIR'], 'parquet')
        COMPANIES = config['companies']
    
    except Exception as e:
        print('Failed to load configuration. Raising exception...')
        raise e
    print('Successfully loaded configuration.')
    

    print('\nCreating or reading gold directory...')
    try:
        os.makedirs(MODELED_PARQUET_DIR, exist_ok=True)
    except Exception as e:
        print('Failed to create gold directory. Raising exception...')
        raise e
    print('Successfully created or read gold directory.')


    for company in COMPANIES:
        print(f'\nProcessing {company}...')
        try:
            file_path = os.path.join(ENRICHED_PARQUET_DIR, f'{company}_share_prices_cleaned.parquet')
            lf = pl.scan_parquet(file_path)

            lf = lf.with_columns(
                # Log Return
                (pl.col('Adj. Close') / pl.col('Adj. Close').shift(1)).log().alias(f'Log Return')
            )

            for window in [5, 10, 20]:
                lf = lf.with_columns(
                    # Rolling Log Returns
                    (pl.col('Adj. Close') / pl.col('Adj. Close').shift(window)).log().alias(f'Log Return {window}d'),

                    # Volatility
                    pl.col('Log Return').rolling_std(window).log1p().alias(f'Volatility {window}d'),

                    # Moving Averages
                    pl.col("Adj. Close").rolling_mean(5).alias(f"Moving Average {window}d"),

                    # Momentum Pct. Change
                    ((pl.col("Adj. Close") / pl.col("Adj. Close").shift(window)) - 1).alias(f"Momentum Pct. {window}d"),

                    # Log Volume Ratio
                    ((pl.col("Volume") / pl.col("Volume").rolling_mean(window))).log().alias(f"Log Volume Ratio {window}d"),
                )

                lf = lf.with_columns(
                    # Log MA Ratio
                    (pl.col("Adj. Close") / pl.col(f"Moving Average {window}d")).log().alias(f"Log MA Ratio {window}d")
                )

            lf = lf.with_columns(
                # Intraday Returns
                ((pl.col('Close') / pl.col('Open')) - 1).alias('Intraday Pct. Return'),

                # Ranges
                (pl.col('High') - pl.col('Low')).alias('Range'),
                ((pl.col('High') - pl.col('Low')) / pl.col('Close')).alias('Range Pct.'),

                # Close Position
                (((pl.col('Close') - pl.col('Low')) / (pl.col('High') - pl.col('Low'))) - 0.5).alias('Close Position'),

                # Log Volume Change
                (pl.col("Volume") / pl.col("Volume").shift(1)).log().tanh().alias("Log Volume Change"),

                # Log Market Cap
                (pl.col("Adj. Close") * pl.col("Shares Outstanding")).log().alias("Log Market Cap"),

                # Dilution / Issuance
                (pl.col("Shares Outstanding") / pl.col("Shares Outstanding").shift(1) - 1).alias('Delta Pct. Dilution / Issuance'),

                # Volume Return Interaction
                (pl.col("Log Return") * pl.col("Log Volume Ratio 5d")).tanh().alias("Interaction Return Volume 5d"),
                (pl.col("Log Return") * pl.col("Log Volume Ratio 10d")).tanh().alias("Interaction Return Volume 10d"),
                (pl.col("Log Return") * pl.col("Log Volume Ratio 20d")).tanh().alias("Interaction Return Volume 20d"),

                # Volume Volatility Interaction
                (pl.col("Volatility 5d") * pl.col("Log Volume Ratio 5d")).tanh().alias("Interaction Volatility Volume 5d"),
                (pl.col("Volatility 10d") * pl.col("Log Volume Ratio 10d")).tanh().alias("Interaction Volatility Volume 10d"),
                (pl.col("Volatility 20d") * pl.col("Log Volume Ratio 20d")).tanh().alias("Interaction Volatility Volume 20d"),

                # Momentum Volatility Interaction
                (pl.col("Momentum Pct. 5d") * pl.col("Volatility 5d")).tanh().alias("Interaction Momentum Volatility 5d"),
                (pl.col("Momentum Pct. 10d") * pl.col("Volatility 10d")).tanh().alias("Interaction Momentum Volume 10d"),
                (pl.col("Momentum Pct. 20d") * pl.col("Volatility 20d")).tanh().alias("Interaction Momentum Volume 20d"),

                # Target Engineering
                (pl.col("Adj. Close").shift(-1) / pl.col("Adj. Close") - 1).alias("Target Return Metric"),
                ((pl.col("Adj. Close").shift(-1) / pl.col("Adj. Close") - 1) > 0).alias("Target Return Class")
            )

            lf = lf.drop([
                "Open", 
                "High",
                "Low",
                "Close",
                "Adj. Close",
                "Moving Average 5d",
                "Moving Average 10d",
                "Moving Average 20d",
                "Volume",
                "Range",
                "Shares Outstanding"
            ])

            lf = lf.drop_nulls()

            output_filepath = os.path.join(MODELED_PARQUET_DIR, f'{company}_share_prices_modeled.parquet')
            lf.collect().write_parquet(output_filepath)
            
        except Exception as e:
            print(f'Failed to load, process, and save {company} data. Raising exception...')
            raise e
        print(f'Successfully loaded, processed, and saved {company} data.')


    print('\nFeature engineering process finished.')


if __name__ == '__main__':
    main()
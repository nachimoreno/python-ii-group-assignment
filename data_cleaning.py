import tomllib
from pathlib import Path
import polars as pl

RAW_PARQUET_DIR = os.path.join('ETL', 'data', 'raw', 'parquet')
ENRICHED_PARQUET_DIR = os.path.join('ETL', 'data', 'enriched', 'parquet')

def main():
    print('\nStarting data cleaning...')

    print('\nLoading configuration...')
    try:
        config_file_path = Path("config.toml")

        with config_file_path.open("rb") as config_file:
            config = tomllib.load(config_file)

        COMPANIES = config['companies']
    except Exception as e:
        print('Failed to load configuration. Raising exception...')
        raise e
    print('Successfully loaded configuration.')

    print('\nReading parquet files...')
    try:
        companies_file_path = os.path.join(RAW_PARQUET_DIR, 'companies.parquet')
        companies = pl.read_parquet(companies_file_path)

        share_prices_file_path = os.path.join(RAW_PARQUET_DIR, 'share_prices.parquet')
        share_prices = pl.read_parquet(share_prices_file_path)
    except Exception as e:
        print('Failed to load parquet files. Raising exception...')
        raise e
    print('Successfully loaded parquet files.')

    print('\nFiltering down to selected companies...')
    try:
        print(f'Selected companies: {COMPANIES}')
        selected_companies = companies.filter(pl.col('Company Name').is_in(COMPANIES))
        SIMFIN_IDS = [id for id in selected_companies['SimFinId']]
        share_prices = share_prices.filter(pl.col('SimFinId').is_in(SIMFIN_IDS))
        print(f'Fetched {len(selected)} rows for {len(COMPANIES)} companies.')
    except Exception as e:
        print('Failed to filter dataframes. Raising exception...')
        raise e
    print('Successfully filtered dataframes.')

    print('\nDropping unnecessary columns...')
    try:
        share_prices = share_prices.drop(pl.col('Dividend'), pl.col('SimFinId'))
    except Exception as e:
        print('Failed to drop columns. Raising exception...')
        raise e
    print('Successfully dropped columns.')

    print(f'\nSaving to parquet at {ENRICHED_PARQUET_DIR}...')
    try:
        file_path = os.path.join(ENRICHED_PARQUET_DIR, 'share_prices_enriched.parquet')
        share_prices.write_parquet(file_path)
    except Exception as e:
        print('Failed to save to parquet. Raising exception...')
        raise e
    print('Successfully saved to parquet.')
    
    print('\nData cleaning process finished.')

if __name__ == '__main__':
    main()
from dotenv import load_dotenv
import os
import simfin as sf
import polars as pl
from pathlib import Path
try:
    import tomllib
except ImportError:
    import tomli as tomllib

def main():
    print('\nStarting data ingestion...')


    print('\nLoading configuration...')
    try:
        config_file_path = Path("src/config.toml")

        with config_file_path.open("rb") as config_file:
            config = tomllib.load(config_file)

        ETL_DIR = config['ETL_DIR']
        RAW_DATA_DIR = config['BRONZE_DIR']
        RAW_PARQUET_DIR = os.path.join(config['BRONZE_DIR'], 'parquet')

    except Exception as e:
        print('Failed to load configuration. Raising exception...')
        raise e
    print('Successfully loaded configuration.')


    print('Creating or reading ETL directories...')
    try:
        os.makedirs(ETL_DIR, exist_ok=True)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(RAW_PARQUET_DIR, exist_ok=True)
    except Exception as e:
        print('Failed to create ETL directories. Raising exception.')
        raise e
    print('Successfully created or read ETL directories.')


    print('\nLoading environment variables...')
    try:
        load_dotenv()
        sf.set_api_key(os.getenv('API_KEY'))
        sf.set_data_dir(RAW_DATA_DIR)
    except Exception as e:
        print('Failed to load environment variables. Raising exception.')
        raise e
    print('Environment variables loaded successfully.')


    print(f'\nLoading company data into {RAW_DATA_DIR}...')
    try:
        pd_companies = sf.load_companies(market='us')
        pl_companies = pl.DataFrame(pd_companies)
    except Exception as e:
        print('Failed to load company data. Raising exception.')
        raise e
    print('Successfully loaded company data.')


    print('\nLoading share prices...')
    try:
        pd_share_prices = sf.load_shareprices(variant='daily')
        pl_share_prices = pl.DataFrame(pd_share_prices)
    except Exception as e:
        print('Failed to load share prices. Raising exception.')
        raise e
    print('Successfully loaded share prices.')


    print(f'\nSaving to parquet at {RAW_PARQUET_DIR}...')
    try:
        share_prices_file_path = os.path.join(RAW_PARQUET_DIR, 'share_prices.parquet')
        pl_share_prices.write_parquet(share_prices_file_path)

        companies_file_path = os.path.join(RAW_PARQUET_DIR, 'companies.parquet')
        pl_companies.write_parquet(companies_file_path)
    except Exception as e:
        print('Failed to save to parquet. Raising exception.')
        raise e
    print('\nSuccessfully saved data to parquet.')


    print('\nData ingestion process finished.')

if __name__ == '__main__':
    main()
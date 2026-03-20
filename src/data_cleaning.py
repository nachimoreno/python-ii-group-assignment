try:
    import tomllib
except ImportError:
    import tomli as tomllib
from pathlib import Path
import os
import polars as pl

def main():
    print('\nStarting data cleaning...')


    print('\nLoading configuration...')
    try:
        config_file_path = Path("src/config.toml")

        with config_file_path.open("rb") as config_file:
            config = tomllib.load(config_file)
        
        RAW_PARQUET_DIR = os.path.join(config['BRONZE_DIR'], 'parquet')
        ENRICHED_PARQUET_DIR = os.path.join(config['SILVER_DIR'], 'parquet')
        COMPANIES = config['companies']
    except Exception as e:
        print('Failed to load configuration. Raising exception...')
        raise e
    print('Successfully loaded configuration.')


    print('\nCreating or reading silver directory...')
    try:
        os.makedirs(ENRICHED_PARQUET_DIR, exist_ok=True)
    except Exception as e:
        print('Failed to create silver directory. Raising exception...')
        raise e
    print('Successfully created or read silver directory.')


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
        company_dfs = {}

        for company in COMPANIES:
            selected_company = companies.filter(pl.col('Company Name') == company)
            SIMFIN_ID = selected_company['SimFinId'].item()
            company_dfs[company] = share_prices.filter(pl.col('SimFinId') == SIMFIN_ID)
            print(f'Fetched {len(selected_company)} rows for {company}.')

    except Exception as e:
        print('Failed to filter dataframes. Raising exception...')
        raise e
    print('Successfully filtered dataframes.')


    print('\nDropping unnecessary columns...')
    try:
        for company in COMPANIES:
            company_dfs[company] = company_dfs[company].drop(pl.col('Dividend'), pl.col('SimFinId'))
    except Exception as e:
        print('Failed to drop columns. Raising exception...')
        raise e
    print('Successfully dropped columns.')


    print(f'\nSaving to parquet at {ENRICHED_PARQUET_DIR}...')
    try:
        for company in COMPANIES:
            file_path = os.path.join(ENRICHED_PARQUET_DIR, f'{company}_share_prices_cleaned.parquet')
            company_dfs[company].write_parquet(file_path)
    except Exception as e:
        print('Failed to save to parquet. Raising exception...')
        raise e
    print('Successfully saved to parquet.')
    
    
    print('\nData cleaning process finished.')

if __name__ == '__main__':
    main()
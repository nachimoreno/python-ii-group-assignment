import os
import json
import pickle
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import polars as pl
import pandas as pd
from pathlib import Path


TARGET_CLASSIFICATION = 'Target Return Class'
TARGET_REGRESSION = 'Target Return Metric'


def load_saved_artifacts(model_output_dir, company):
    company_simple_name = company.split()[0].title()
    company_model_dir = os.path.join(model_output_dir, company_simple_name)

    with open(os.path.join(company_model_dir, 'classification_model.pkl'), 'rb') as model_file:
        classification_model = pickle.load(model_file)

    with open(os.path.join(company_model_dir, 'regression_model.pkl'), 'rb') as model_file:
        regression_model = pickle.load(model_file)

    with open(os.path.join(company_model_dir, 'metadata.json'), 'r', encoding='utf-8') as metadata_file:
        metadata = json.load(metadata_file)

    return classification_model, regression_model, metadata


def load_modeled_company_dataframe(modeled_parquet_dir, company):
    file_path = os.path.join(modeled_parquet_dir, f'{company}_share_prices_modeled.parquet')
    lazyframe = pl.scan_parquet(file_path)

    company_dataframe = lazyframe.collect().to_pandas()
    return company_dataframe


def load_raw_company_dataframe(enriched_parquet_dir, company):
    file_path = os.path.join(enriched_parquet_dir, f'{company}_share_prices_cleaned.parquet')
    lazyframe = pl.scan_parquet(file_path)

    company_dataframe = lazyframe.collect().to_pandas()
    return company_dataframe


def build_holdout_test_dataframes(modeled_company_dataframe, raw_company_dataframe, feature_columns, test_fraction=0.20):
    aligned_raw_company_dataframe = raw_company_dataframe.iloc[20:-1].copy()

    modeled_test_size = max(1, int(len(modeled_company_dataframe) * test_fraction))

    modeled_test_dataframe = modeled_company_dataframe.iloc[-modeled_test_size:].copy()
    raw_test_dataframe = aligned_raw_company_dataframe.iloc[-modeled_test_size:].copy()

    X_test = modeled_test_dataframe[feature_columns].copy()

    return modeled_test_dataframe, raw_test_dataframe, X_test


def add_model_predictions(modeled_test_dataframe, X_test, classification_model, regression_model):
    prediction_dataframe = modeled_test_dataframe.copy()

    prediction_dataframe['predicted_class'] = classification_model.predict(X_test)
    prediction_dataframe['predicted_probability'] = classification_model.predict_proba(X_test)[:, 1]
    prediction_dataframe['predicted_return'] = regression_model.predict(X_test)

    return prediction_dataframe


def attach_raw_price_data(prediction_dataframe, raw_test_dataframe, price_column):
    combined_dataframe = prediction_dataframe.copy()
    combined_dataframe[price_column] = raw_test_dataframe[price_column].to_numpy()

    return combined_dataframe


def determine_trade_size(predicted_probability, predicted_class, base_trade_size, max_trade_size):
    if predicted_class == 1:
        confidence = predicted_probability
    else:
        confidence = 1 - predicted_probability

    if confidence < 0.55:
        trade_size = 0
    elif confidence < 0.60:
        trade_size = base_trade_size
    elif confidence < 0.70:
        trade_size = base_trade_size * 2
    else:
        trade_size = max_trade_size

    return min(trade_size, max_trade_size)


def generate_actions(
    prediction_dataframe,
    base_trade_size=1,
    max_trade_size=5,
    buy_probability_threshold=0.55,
    sell_probability_threshold=0.45,
):
    action_dataframe = prediction_dataframe.copy()

    action_list = []
    trade_size_list = []

    predicted_return_is_usable = action_dataframe['predicted_return'].nunique() > 1

    for _, row in action_dataframe.iterrows():
        predicted_class = int(row['predicted_class'])
        predicted_probability = float(row['predicted_probability'])
        predicted_return = float(row['predicted_return'])

        if predicted_return_is_usable:
            absolute_predicted_return = abs(predicted_return)

            if absolute_predicted_return < 0.005:
                trade_size = 0
            elif absolute_predicted_return < 0.01:
                trade_size = base_trade_size
            elif absolute_predicted_return < 0.02:
                trade_size = base_trade_size * 2
            else:
                trade_size = max_trade_size
        else:
            trade_size = determine_trade_size(
                predicted_probability=predicted_probability,
                predicted_class=predicted_class,
                base_trade_size=base_trade_size,
                max_trade_size=max_trade_size,
            )

        action = 'HOLD'

        if predicted_return_is_usable:
            if predicted_class == 1 and predicted_probability >= buy_probability_threshold and predicted_return > 0:
                action = 'BUY'
            elif predicted_class == 0 and predicted_probability <= sell_probability_threshold and predicted_return < 0:
                action = 'SELL'
        else:
            if predicted_class == 1 and predicted_probability >= buy_probability_threshold:
                action = 'BUY'
            elif predicted_class == 0 and predicted_probability <= sell_probability_threshold:
                action = 'SELL'

        if trade_size == 0:
            action = 'HOLD'

        action_list.append(action)
        trade_size_list.append(min(trade_size, max_trade_size))

    action_dataframe['action'] = action_list
    action_dataframe['trade_size'] = trade_size_list

    return action_dataframe


def backtest_holdout_strategy(action_dataframe, price_column, initial_cash=10000.0):
    backtest_dataframe = action_dataframe.copy()

    cash = initial_cash
    shares_held = 0

    executed_action_list = []
    executed_trade_size_list = []
    cash_history = []
    shares_held_history = []
    portfolio_value_history = []
    trade_value_history = []

    for _, row in backtest_dataframe.iterrows():
        current_price = float(row[price_column])
        requested_action = row['action']
        requested_trade_size = int(row['trade_size'])

        executed_action = 'HOLD'
        executed_trade_size = 0
        trade_value = 0.0

        if requested_action == 'BUY' and requested_trade_size > 0:
            affordable_trade_size = int(cash // current_price)
            shares_to_buy = min(requested_trade_size, affordable_trade_size)

            if shares_to_buy > 0:
                trade_value = shares_to_buy * current_price
                cash -= trade_value
                shares_held += shares_to_buy
                executed_action = 'BUY'
                executed_trade_size = shares_to_buy

        elif requested_action == 'SELL' and requested_trade_size > 0:
            shares_to_sell = min(requested_trade_size, shares_held)

            if shares_to_sell > 0:
                trade_value = shares_to_sell * current_price
                cash += trade_value
                shares_held -= shares_to_sell
                executed_action = 'SELL'
                executed_trade_size = shares_to_sell

        portfolio_value = cash + shares_held * current_price

        executed_action_list.append(executed_action)
        executed_trade_size_list.append(executed_trade_size)
        trade_value_history.append(trade_value)
        cash_history.append(cash)
        shares_held_history.append(shares_held)
        portfolio_value_history.append(portfolio_value)

    backtest_dataframe['executed_action'] = executed_action_list
    backtest_dataframe['executed_trade_size'] = executed_trade_size_list
    backtest_dataframe['trade_value'] = trade_value_history
    backtest_dataframe['cash'] = cash_history
    backtest_dataframe['shares_held'] = shares_held_history
    backtest_dataframe['portfolio_value'] = portfolio_value_history

    return backtest_dataframe


def summarize_backtest(backtest_dataframe):
    initial_portfolio_value = float(backtest_dataframe['portfolio_value'].iloc[0])
    final_portfolio_value = float(backtest_dataframe['portfolio_value'].iloc[-1])
    total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value

    executed_buy_count = int((backtest_dataframe['executed_action'] == 'BUY').sum())
    executed_sell_count = int((backtest_dataframe['executed_action'] == 'SELL').sum())
    hold_count = int((backtest_dataframe['executed_action'] == 'HOLD').sum())

    total_shares_bought = int(
        backtest_dataframe.loc[
            backtest_dataframe['executed_action'] == 'BUY',
            'executed_trade_size',
        ].sum()
    )

    total_shares_sold = int(
        backtest_dataframe.loc[
            backtest_dataframe['executed_action'] == 'SELL',
            'executed_trade_size',
        ].sum()
    )

    return {
        'initial_portfolio_value': initial_portfolio_value,
        'final_portfolio_value': final_portfolio_value,
        'total_return': total_return,
        'executed_buy_count': executed_buy_count,
        'executed_sell_count': executed_sell_count,
        'hold_count': hold_count,
        'total_shares_bought': total_shares_bought,
        'total_shares_sold': total_shares_sold,
    }


def save_backtest_outputs(backtest_output_dir, company, backtest_dataframe, backtest_summary):
    company_simple_name = company.split()[0].title()
    company_backtest_dir = os.path.join(backtest_output_dir, company_simple_name)
    os.makedirs(company_backtest_dir, exist_ok=True)

    backtest_dataframe_path = os.path.join(company_backtest_dir, 'holdout_backtest.csv')
    backtest_summary_path = os.path.join(company_backtest_dir, 'holdout_backtest_summary.json')

    backtest_dataframe.to_csv(backtest_dataframe_path, index=False)

    with open(backtest_summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(backtest_summary, summary_file, indent=2)


def main():
    print('\nStarting backtesting process...')

    price_column = 'Close'
    test_fraction = 0.20
    initial_cash = 10000.0
    base_trade_size = 1
    max_trade_size = 5
    buy_probability_threshold = 0.55
    sell_probability_threshold = 0.45


    print('\nLoading configuration...')
    try:
        config_file_path = Path('src/config.toml')

        with config_file_path.open('rb') as config_file:
            config = tomllib.load(config_file)

        ENRICHED_PARQUET_DIR = os.path.join(config['SILVER_DIR'], 'parquet')
        MODELED_PARQUET_DIR = os.path.join(config['GOLD_DIR'], 'parquet')
        MODEL_OUTPUT_DIR = os.path.join(config['GOLD_DIR'], 'trained_models')
        BACKTEST_OUTPUT_DIR = os.path.join(config['GOLD_DIR'], 'backtests')
        COMPANIES = config['companies']

    except Exception as e:
        print('Failed to load configuration. Raising exception...')
        raise e
    print('Successfully loaded configuration.')


    print('\nCreating or reading backtests directory...')
    try:
        os.makedirs(BACKTEST_OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        print('Failed to create backtests directory. Raising exception...')
        raise e
    print('Successfully created or read backtests directory.')


    all_backtest_summaries = []

    for company in COMPANIES:
        print(f'\nBacktesting holdout strategy for {company}...')
        try:
            classification_model, regression_model, metadata = load_saved_artifacts(MODEL_OUTPUT_DIR, company)
            feature_columns = metadata['feature_columns']

            modeled_company_dataframe = load_modeled_company_dataframe(MODELED_PARQUET_DIR, company)
            raw_company_dataframe = load_raw_company_dataframe(ENRICHED_PARQUET_DIR, company)

            if price_column not in raw_company_dataframe.columns:
                raise ValueError(
                    f'{price_column} column not found for {company}. '
                    'Set price_column to the correct market price column name.'
                )

            modeled_test_dataframe, raw_test_dataframe, X_test = build_holdout_test_dataframes(
                modeled_company_dataframe=modeled_company_dataframe,
                raw_company_dataframe=raw_company_dataframe,
                feature_columns=feature_columns,
                test_fraction=test_fraction,
            )

            prediction_dataframe = add_model_predictions(
                modeled_test_dataframe=modeled_test_dataframe,
                X_test=X_test,
                classification_model=classification_model,
                regression_model=regression_model,
            )

            prediction_dataframe = attach_raw_price_data(
                prediction_dataframe=prediction_dataframe,
                raw_test_dataframe=raw_test_dataframe,
                price_column=price_column,
            )

            action_dataframe = generate_actions(
                prediction_dataframe=prediction_dataframe,
                base_trade_size=base_trade_size,
                max_trade_size=max_trade_size,
                buy_probability_threshold=buy_probability_threshold,
                sell_probability_threshold=sell_probability_threshold,
            )

            backtest_dataframe = backtest_holdout_strategy(
                action_dataframe=action_dataframe,
                price_column=price_column,
                initial_cash=initial_cash,
            )

            backtest_summary = summarize_backtest(backtest_dataframe)

            save_backtest_outputs(
                backtest_output_dir=BACKTEST_OUTPUT_DIR,
                company=company,
                backtest_dataframe=backtest_dataframe,
                backtest_summary=backtest_summary,
            )

            all_backtest_summaries.append(
                {
                    'company': company,
                    **backtest_summary,
                }
            )

        except Exception as e:
            print(f'Failed to load, backtest, summarize, and save {company} results. Raising exception...')
            raise e
        print(f'Successfully loaded, backtested, summarized, and saved {company} results.')

        print(f'  Initial portfolio value: {backtest_summary["initial_portfolio_value"]:.2f}')
        print(f'  Final portfolio value:   {backtest_summary["final_portfolio_value"]:.2f}')
        print(f'  Total return:            {backtest_summary["total_return"]:.4%}')
        print(f'  Buy trades:              {backtest_summary["executed_buy_count"]}')
        print(f'  Sell trades:             {backtest_summary["executed_sell_count"]}')
        print(f'  Holds:                   {backtest_summary["hold_count"]}')


    summary_dataframe = pd.DataFrame(all_backtest_summaries)
    summary_output_path = os.path.join(BACKTEST_OUTPUT_DIR, 'all_company_backtest_summary.csv')
    summary_dataframe.to_csv(summary_output_path, index=False)

    print('\nBacktesting process finished.')
    print(summary_dataframe)


if __name__ == '__main__':
    main()
import os
import json
import pickle
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import polars as pl
from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


TARGET_CLASSIFICATION = 'Target Return Class'
TARGET_REGRESSION = 'Target Return Metric'


def load_company_dataframe(modeled_parquet_dir, company):
    file_path = os.path.join(modeled_parquet_dir, f'{company}_share_prices_modeled.parquet')
    lazyframe = pl.scan_parquet(file_path)

    feature_columns = [
        column_name
        for column_name in lazyframe.collect_schema().names()
        if column_name not in {TARGET_CLASSIFICATION, TARGET_REGRESSION}
    ]

    company_dataframe = (
        lazyframe
        .select(feature_columns + [TARGET_CLASSIFICATION, TARGET_REGRESSION])
        .with_columns(
            pl.col(TARGET_CLASSIFICATION).cast(pl.Int64),
            pl.col(TARGET_REGRESSION).cast(pl.Float64),
        )
        .collect()
        .to_pandas()
    )

    return company_dataframe, feature_columns


def make_train_test_split(X, y, test_fraction=0.20):
    test_size = max(1, int(len(X) * test_fraction))

    X_train = X.iloc[:-test_size].copy()
    y_train = y.iloc[:-test_size].copy()

    X_test = X.iloc[-test_size:].copy()
    y_test = y.iloc[-test_size:].copy()

    return X_train, X_test, y_train, y_test


def make_time_series_split(X_train):
    return TimeSeriesSplit(
        n_splits=5,
        test_size=max(1, len(X_train) // 10),
        gap=0,
    )


def train_classification_model(X, y):
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    time_series_split = make_time_series_split(X_train)

    classifier = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=42,
    )

    classification_parameter_distributions = {
        'n_estimators': [200, 400, 800],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.3, 1.0],
        'reg_alpha': [0.0, 0.01, 0.1, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0],
    }

    hyperparameter_search = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=classification_parameter_distributions,
        n_iter=40,
        scoring='roc_auc',
        cv=time_series_split,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        refit=True,
    )

    hyperparameter_search.fit(X_train, y_train)

    best_model = hyperparameter_search.best_estimator_

    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = best_model.predict(X_test)

    metrics = {
        'best_cv_score': float(hyperparameter_search.best_score_),
        'best_params': hyperparameter_search.best_params_,
        'test_roc_auc': float(roc_auc_score(y_test, test_probabilities)),
        'test_accuracy': float(accuracy_score(y_test, test_predictions)),
        'classification_report': classification_report(y_test, test_predictions, output_dict=True),
    }

    return best_model, metrics


def train_regression_model(X, y):
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    time_series_split = make_time_series_split(X_train)

    regressor = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        random_state=42,
    )

    regression_parameter_distributions = {
        'n_estimators': [200, 400, 800],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.3, 1.0],
        'reg_alpha': [0.0, 0.01, 0.1, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0],
    }

    hyperparameter_search = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=regression_parameter_distributions,
        n_iter=40,
        scoring='neg_mean_squared_error',
        cv=time_series_split,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        refit=True,
    )

    hyperparameter_search.fit(X_train, y_train)

    best_model = hyperparameter_search.best_estimator_

    test_predictions = best_model.predict(X_test)

    metrics = {
        'best_cv_score': float(hyperparameter_search.best_score_),
        'best_params': hyperparameter_search.best_params_,
        'test_mae': float(mean_absolute_error(y_test, test_predictions)),
        'test_mse': float(mean_squared_error(y_test, test_predictions)),
        'test_rmse': float(mean_squared_error(y_test, test_predictions) ** 0.5),
        'test_r2': float(r2_score(y_test, test_predictions)),
    }

    return best_model, metrics


def save_model_artifacts(
    model_output_dir,
    company,
    feature_columns,
    classification_model,
    classification_metrics,
    regression_model,
    regression_metrics,
):
    company_simple_name = company.split()[0].title()
    company_output_dir = os.path.join(model_output_dir, company_simple_name)
    os.makedirs(company_output_dir, exist_ok=True)

    classification_model_path = os.path.join(company_output_dir, 'classification_model.pkl')
    regression_model_path = os.path.join(company_output_dir, 'regression_model.pkl')
    metadata_path = os.path.join(company_output_dir, 'metadata.json')

    with open(classification_model_path, 'wb') as classification_model_file:
        pickle.dump(classification_model, classification_model_file)

    with open(regression_model_path, 'wb') as regression_model_file:
        pickle.dump(regression_model, regression_model_file)

    metadata = {
        'company': company,
        'feature_columns': feature_columns,
        'target_classification': TARGET_CLASSIFICATION,
        'target_regression': TARGET_REGRESSION,
        'classification_metrics': classification_metrics,
        'regression_metrics': regression_metrics,
    }

    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def main():
    print('\nStarting model training process...')


    print('\nLoading configuration...')
    try:
        config_file_path = Path('src/config.toml')

        with config_file_path.open('rb') as config_file:
            config = tomllib.load(config_file)

        MODELED_PARQUET_DIR = os.path.join(config['GOLD_DIR'], 'parquet')
        MODEL_OUTPUT_DIR = os.path.join(config['GOLD_DIR'], 'trained_models')
        COMPANIES = config['companies']

    except Exception as e:
        print('Failed to load configuration. Raising exception...')
        raise e
    print('Successfully loaded configuration.')


    print('\nCreating or reading trained models directory...')
    try:
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        print('Failed to create trained models directory. Raising exception...')
        raise e
    print('Successfully created or read trained models directory.')


    all_company_results = []

    for company in COMPANIES:
        print(f'\nTraining models for {company}...')
        try:
            company_dataframe, feature_columns = load_company_dataframe(MODELED_PARQUET_DIR, company)

            X = company_dataframe[feature_columns]
            y_classification = company_dataframe[TARGET_CLASSIFICATION]
            y_regression = company_dataframe[TARGET_REGRESSION]

            classification_model, classification_metrics = train_classification_model(
                X,
                y_classification,
            )

            regression_model, regression_metrics = train_regression_model(
                X,
                y_regression,
            )

            save_model_artifacts(
                model_output_dir=MODEL_OUTPUT_DIR,
                company=company,
                feature_columns=feature_columns,
                classification_model=classification_model,
                classification_metrics=classification_metrics,
                regression_model=regression_model,
                regression_metrics=regression_metrics,
            )

            all_company_results.append(
                {
                    'company': company,
                    'classification_metrics': classification_metrics,
                    'regression_metrics': regression_metrics,
                }
            )

        except Exception as e:
            print(f'Failed to load, train, evaluate, and save {company} models. Raising exception...')
            raise e
        print(f'Successfully loaded, trained, evaluated, and saved {company} models.')

        print(f'  Classification ROC AUC - {classification_metrics["test_roc_auc"]:.4f}')
        print(f'  Classification Accuracy - {classification_metrics["test_accuracy"]:.4f}')
        print(f'  Regression RMSE - {regression_metrics["test_rmse"]:.6f}')
        print(f'  Regression R2 - {regression_metrics["test_r2"]:.4f}')


    print('\nModel training process finished.')


if __name__ == '__main__':
    main()
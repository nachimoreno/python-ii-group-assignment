import argparse
import src.data_ingestion as data_ingestion
import src.data_cleaning as data_cleaning
import src.feature_engineering as feature_engineering
import src.model_training as model_training
import src.trading_logic as trading_logic

def main():
    data_ingestion.main()
    data_cleaning.main()
    feature_engineering.main()
    model_training.main()
    trading_logic.main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ETL Pipeline')
    parser.add_argument('--stage', type=str, default='all', choices=['ingestion', 'cleaning', 'feature_engineering', 'model_training', 'trading_logic', 'all'], help='Stage to run')
    args = parser.parse_args()
    if args.stage == 'ingestion':
        data_ingestion.main()
    elif args.stage == 'cleaning':
        data_cleaning.main()
    elif args.stage == 'feature_engineering':
        feature_engineering.main()
    elif args.stage == 'model_training':
        model_training.main()
    elif args.stage == 'trading_logic':
        trading_logic.main()
    elif args.stage == 'all':
        main()
    else:
        print('Invalid stage. Please choose from ingestion, cleaning, feature_engineering, model_training, trading_logic, or all.')
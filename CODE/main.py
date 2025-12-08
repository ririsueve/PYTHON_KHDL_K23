import logging
from config import *
from modules.utils import SystemLogger
from modules.preprocessing import DataCleaner, FeatureEngineer, DataTransformer, DataPreparationPipeline
from modules.pipeline import AutoMLPipeline

def main():
    # 1. Logger
    SystemLogger("automl_run.log")
    logger = logging.getLogger("Main Pipeline")
    logger.info("=== BẮT ĐẦU PIPELINE (MODULAR VERSION) ===")

    # 2. Chuẩn bị dữ liệu (Cleaning & Transform)
    cleaner = DataCleaner(
        golden_specs=my_golden_specs, 
        range_rules=my_range_rules, 
        cols_to_drop=cols_to_drop
    )
    engineering = FeatureEngineer()
    transformer = DataTransformer(
        scaling_strategy='minmax',
        outlier_strategies= my_outlier_strategies,
        ordinal_mappings= my_ordinal_mappings,
        nominal_cols= nominal_columns,
        ignore_cols= ['OUTCOME']
    )

    data_pipeline = DataPreparationPipeline(
        file_path='DATA_RISK_CLASSIFY.csv',  # Đảm bảo file train.csv nằm cùng thư mục
        cleaner=cleaner,
        featuring=engineering,
        transformer=transformer,
        target_col='OUTCOME'
    )

    # Chạy tiền xử lý và lưu file
    train_file = "final_train_data.csv"
    test_file = "final_test_data.csv"
    train_data, test_data = data_pipeline.run(test_size=0.2)
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    # 3. Chạy AutoML Pipeline
    automl = AutoMLPipeline(train_file, test_file, tuning_method="default", feature_method= "rfe", n_features=15)
    automl.run()

if __name__ == "__main__":
    main()
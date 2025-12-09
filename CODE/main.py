import logging
import sys
import os
import argparse

from config import *
from modules.utils import SystemLogger
from modules.preprocessing import DataCleaner, FeatureEngineer, DataTransformer, DataPreparationPipeline
from modules.pipeline import AutoMLPipeline

sys.path.append(os.path.join(os.path.dirname(__file__), 'CODE'))

def parse_arguments():
    """
    Thiết lập các tham số dòng lệnh (Command Line Arguments)
    """
    parser = argparse.ArgumentParser(description="AutoML Pipeline cho bài toán Risk Prediction")

    # 1. Tham số về Dữ liệu
    parser.add_argument("--file", type=str, default="DATA/data.csv",
                        help="Đường dẫn đến file dữ liệu gốc (CSV)")
    
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Tỷ lệ chia tập Test (0.2 = 20%)")

    # 2. Tham số về Tuning & Feature Selection
    parser.add_argument("--tuning", type=str, default="random_search",
                        choices=["default", "grid_search", "random_search"],
                        help="Phương pháp tinh chỉnh tham số (Hyperparameter Tuning)")

    parser.add_argument("--feature_method", type=str, default="rfe",
                        choices=["rfe", "forward", "backward"],
                        help="Phương pháp lựa chọn đặc trưng")

    parser.add_argument("--n_features", type=int, default=15,
                        help="Số lượng đặc trưng muốn giữ lại")

    return parser.parse_args()


def main():
    # 1. Lấy tham số từ dòng lệnh
    args = parse_arguments()

    # 2. Logger
    SystemLogger("automl_run.log")
    logger = logging.getLogger("Main Pipeline")
    logger.info("=== BẮT ĐẦU PIPELINE (MODULAR VERSION) ===")

    # 3. Chuẩn bị dữ liệu (Cleaning & Transform)
    cleaner = DataCleaner(
        golden_specs=my_golden_specs, 
        range_rules=my_range_rules, 
        cols_to_drop=cols_to_drop,
        max_drop_ratio=max_drop_ratio,
        imputation_strategy=imputation_strategy,
        fuzzy_threshold=fuzzy_threshold
    )
    engineering = FeatureEngineer()
    transformer = DataTransformer(
        scaling_strategy=scaling_strategy,
        outlier_strategies= my_outlier_strategies,
        ordinal_mappings= my_ordinal_mappings,
        nominal_cols= nominal_columns,
        ignore_cols= ignore_cols
    )

    data_pipeline = DataPreparationPipeline(
        file_path= args.file,  # Đảm bảo file data.csv nằm cùng thư mục
        cleaner=cleaner,
        featuring=engineering,
        transformer=transformer,
        target_col='OUTCOME'
    )

    # Chạy tiền xử lý và lưu file
    train_file = "DATA/final_train_data.csv"
    test_file = "DATA/final_test_data.csv"
    train_data, test_data = data_pipeline.run(test_size=0.2)
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    # 4. Chạy AutoML Pipeline
    automl = AutoMLPipeline(
        train_file = train_file, 
        test_file = test_file, 
        tuning_method = args.tuning, 
        feature_method = args.feature_method, 
        n_features = args.n_features)
    automl.run()

if __name__ == "__main__":
    main()

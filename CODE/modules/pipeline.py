import logging
from modules.utils import DataLoader, DataSplitter
from modules.modeling import FeatureWrapperSelector, ModelTrainer, HyperparameterTuning
from modules.evaluation import ModelEvaluator, ReportModel, Visualizer

class AutoMLPipeline:
    """
    Quản lý toàn bộ quy trình chạy thử nghiệm trên nhiều mô hình.
    """
    def __init__(self, train_file, test_file, 
                tuning_method="random_search",
                feature_method="rfe",
                n_features=15):
        """
        Khởi tạo AutoMLPipeline.

        Args:
        train_file: file train
        test_file: file test
        tuning_method: phương pháp tuning
        feature_method: phương pháp lựa chọn đặc trưng
        n_features: số lượng đặc trưng
        """
        self.train_file = train_file
        self.test_file = test_file

        self.tuning_method = tuning_method
        self.feature_method = feature_method
        self.n_features = n_features

        self.logger = logging.getLogger("AutoML Pipeline")
        self.all_results_metrics = {}

    def run(self):
        """
        Chạy pipeline.
        """
        self.logger.info("=== BẮT ĐẦU CHẠY AUTOML PIPELINE (MULTI-MODEL) ===")
        self.logger.info(f"Cấu hình: Tuning={self.tuning_method} | Feature={self.feature_method} ({self.n_features})")

        # 1. Load Data
        df_train = DataLoader(self.train_file).data
        df_test = DataLoader(self.test_file).data

        target_col = "OUTCOME"
        cols_drop = [c for c in ["ID", "POSTAL_CODE"] if c in df_train.columns]

        # Tách tập train/test cho X và y
        X_train = df_train.drop(columns=cols_drop + [target_col], errors='ignore')
        y_train = df_train[target_col]

        if target_col in df_test.columns:
            X_test = df_test.drop(columns=cols_drop + [target_col], errors='ignore')
            y_test = df_test[target_col]
        else:
            raise ValueError("File test không có cột target (OUTCOME) để đánh giá!")

        # 2. Lựa chọn đặc trưng cho cả df
        self.logger.info(f">>> STEP: FEATURE SELECTION ({self.feature_method.upper()})")
        selector = FeatureWrapperSelector(method=self.feature_method, n_features_to_select=self.n_features)

        # Fit trên Train
        X_train_selected = selector.fit_transform(X_train, y_train)

        # Transform trên Test
        selected_cols = selector.selected_columns
        X_test_selected = X_test[selected_cols]

        # 3. Loop Models (Chạy vòng lặp qua từng mô hình)
        supported_models = ModelTrainer.get_supported_models()
        self.logger.info(f"Danh sách mô hình sẽ chạy: {supported_models}")

        for model_name in supported_models:
            self.logger.info(f"\n{'='*20} PROCESSING MODEL: {model_name.upper()} {'='*20}")
            try:
                # A. Hyperparameter Tuning (Trên tập Train - dùng KFold nội bộ)
                # Reset index để tránh lỗi khi split
                X_tune = X_train_selected.reset_index(drop=True)
                y_tune = y_train.reset_index(drop=True)

                splitter = DataSplitter(X_tune)
                folds = splitter.kfold_split_data(n_splits=3) # Tuning nhanh với 3 fold

                tuner = HyperparameterTuning(tuning_method="grid_search", scoring="f1")
                tuning_res = tuner.tune_hyperparameters(X_tune, y_tune, folds, model_name)

                # Lấy kết quả trả về từ tuner
                best_params = tuning_res["best_params"] if tuning_res else {}

                # B. Final Training (Train lại trên toàn bộ tập Train với Best Params)
                trainer = ModelTrainer(model_name, **best_params)
                trainer.train_model(X_train_selected, y_train)
                trainer.save_model("")

                # C. Prediction (Dự đoán trên tập Test độc lập)
                y_pred = trainer.predict_y(X_test_selected)

                # D. Evaluation & Visualization
                # Đóng gói kết quả dạng list 1 phần tử
                result_pack = [(trainer.model, X_test_selected, y_test, y_pred)]

                # Tính điểm (Gọi property .metrics)
                evaluator = ModelEvaluator(result_pack)
                metrics = evaluator.metrics

                # Lưu vào danh sách tổng hợp
                self.all_results_metrics[model_name] = metrics
                self.logger.info(f"Kết quả {model_name}: {metrics}")

                # Vẽ biểu đồ (Tên file ảnh sẽ có prefix tên model)
                viz = Visualizer(result_pack, model_name_prefix=model_name)
                viz.plot_all()

            except Exception as e:
                self.logger.error(f"Lỗi khi chạy model {model_name}: {e}")
                continue
            
        # 4. Final Report (Tổng hợp so sánh)
        self.logger.info("\n>>> STEP: TẠO BÁO CÁO TỔNG HỢP")
        reporter = ReportModel("evaluation_report.txt")
        reporter.save_comparision(self.all_results_metrics)
        self.logger.info("=== HOÀN TẤT AUTOML PIPELINE ===")
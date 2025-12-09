import logging
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import RFE, SequentialFeatureSelector

class ModelTrainer:
    """
    Quản lý việc xây dựng, huấn luyện và dự đoán của một mô hình ML. (chỉ dùng tập train, test truyền vào)
    """
    def __init__(self, model_name: str, **kwargs):
        """
        Khởi tạo ModelTrainer và xây dựng mô hình dựa trên tên.

        Args:
        model_name: Tên mô hình ("logistic_regression", "svm", "decision_tree", "xgboost", "random_forest", "knn") dùng cho bài toán classification
        **kwargs: Các tham số (hyperparameters) tùy chọn cho mô hình.
        """
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        # Gọi hàm buid_model để xây dựng mô hình
        self.model = self.build_model(model_name, **kwargs)

    # Dùng để kiểm tra danh sách model hỗ trợ mà không cần khởi tạo class.
    @staticmethod
    def get_supported_models():
        return [
            "logistic_regression", "svm", "decision_tree",
            "random_forest", "xgboost", "knn"
        ]

    def build_model(self, model_name: str, **kwargs):
        """
        Xây dựng mô hình máy học bằng tên mô hình

        Args:
        model_name (str): Tên mô hình cần khởi tạo.
        **kwargs: Các tham số (hyperparameters) tùy chọn cho mô hình.

        Returns:
        Mô hình đã được xây dựng.

        Raises:
        ValueError: Nếu tên mô hình không hợp lệ.
        """
        self.logger.info(f"Đang khởi tạo mô hình: {model_name}")

        # Đặt random_state mặc định nếu chưa được cung cấp cho các mô hình có yếu tố ngẫu nhiên
        if "random_state" not in kwargs and model_name != "knn":
            kwargs["random_state"] = 42

        try:
            if model_name == "logistic_regression":
                return LogisticRegression(**kwargs)
            elif model_name == "svm":
                return LinearSVC(**kwargs)
            elif model_name == "decision_tree":
                return DecisionTreeClassifier(**kwargs)
            elif model_name == "random_forest":
                return RandomForestClassifier(**kwargs)
            elif model_name == "xgboost":
                return XGBClassifier(**kwargs)
            elif model_name == "knn":
                return KNeighborsClassifier(**kwargs)
            else:
                raise ValueError(f"Không hỗ trợ mô hình {model_name}.")
        except Exception as e:
            self.logger.error(f"Lỗi khởi tạo mô hình: {e}")
            raise e

    def train_model(self, X_train, y_train):
        """
        Huấn luyện mô hình sử dụng tập huấn luyện.

        Args:
        X_train: Tập dữ liệu gồm các đặc trưng để huấn luyện.
        y_train: Dữ liệu mục tiêu để huấn luyện.

        Raises:
        ValueError: Nếu mô hình chưa được xây dựng thành công hoặc không hợp lệ.
        """
        if self.model is None:
            self.logger.error("Mô hình chưa được khởi tạo.")
            raise ValueError("Mô hình chưa được xây dựng thành công hoặc không hợp lệ.")

        self.model.fit(X_train, y_train)

    def predict_y(self, X_test):
        """
        Dự đoán bằng mô hình đã được huấn luyện.

        Args:
        X_test: Tập dữ liệu gồm các đặc trưng để dự đoán.

        Returns:
        Tập dữ liệu mục tiêu đã đươc gán nhãn.

        Raises:
        ValueError: Nếu mô hình chưa được xây dựng thành công hoặc không hợp lệ.
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được xây dựng thành công hoặc không hợp lệ.")
        return self.model.predict(X_test)
    
    def save_model(self, save_dir=""): # Mặc định là rỗng để lưu ngay thư mục gốc
        """
        Lưu model ra file .pkl
        """
        if self.model is None:
            return

        # Nếu folder_path rỗng, lưu ngay tại chỗ. Nếu có, thêm dấu gạch chéo
        file_path = os.path.join(save_dir, f"{self.model_name}.pkl")

        # Lưu file
        joblib.dump(self.model, file_path)
        self.logger.info(f"Đã lưu model: {file_path}")
    
class HyperparameterTuning:
    """
    Tìm ra các siêu tham số tốt nhất cho mô hình.
    """
    def __init__(self, tuning_method: str, scoring: str):
        """
        Khởi tạo HyperparameterTuning với phương pháp tuning.

        Args:
        tuning_method (str): Phương pháp tuning
        scoring (str): Đánh giá tối ưu bằng chỉ số nào
        """
        self.tuning_method = tuning_method
        self.scoring = scoring
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_model_config(self, model_name):
        """
        Hàm tạo một lưới các bộ siêu tham số của từng mô hình theo tên.

        Args:
        model_name (str): Tên mô hình.

        Returns:
        Một lưới chứa các siêu tham số và tên mô hình.
        """
        self.logger.info(f"Tạo lưới các siêu tham số cho mô hình: {model_name}")
        model = None
        param_grid = {}

        if model_name == "logistic_regression":
            model = LogisticRegression(max_iter = 1000, random_state = 42)
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'penalty': ['l2']
            }
        elif model_name == "svm":
            model = LinearSVC(random_state=42, dual=False, max_iter=5000)
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2']
            }
        elif model_name == "decision_tree":
            model = DecisionTreeClassifier(random_state = 42)
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == "knn":
            model = KNeighborsClassifier()
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        elif model_name == "xgboost":
            model = XGBClassifier(eval_metric='logloss', booster='gbtree', random_state=42, n_jobs=1)
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        elif model_name == "random_forest":
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        else:
            self.logger.error(f"Model {model_name} không được hỗ trợ!")
            raise ValueError(f"Model {model_name} chưa được hỗ trợ trong cấu hình.")

        return model, param_grid

    def tune_hyperparameters(self, X, y, folds_data, model_name):
        """
        Thực hiện tuning để tìm các bộ siêu tham số tốt nhất.

        Args:
        X: Tập dữ liệu gồm các đặc trưng.
        y: Dữ liệu mục tiêu.
        folds_data: Một danh sách các tuple, mỗi tuple chứa:
                (train_index, test_index) cho mỗi fold.
        model_name: Tên mô hình ("logistic_regression", "svm", "decision_tree", "xgboost", "knn")

        Returns:
        Một dictionary chứa các siêu tham số tốt nhất.
        """
        self.logger.info(f"=== Bắt đầu tuning cho {model_name} bằng {self.tuning_method} ===")
        self.logger.info(f"Metric tối ưu: {self.scoring}")

        try:
            model, param_grid = self._get_model_config(model_name)

            # TRƯỜNG HỢP 1: CHẠY MẶC ĐỊNH (KHÔNG TUNING)
            if self.tuning_method == "default":
                self.logger.info("Đang chạy mô hình với tham số mặc định (Baseline)...")

                # Đánh giá model bằng Cross-Validation để lấy score công bằng so với tuning
                # n_jobs=-1 để chạy đa luồng
                cv_scores = cross_val_score(model, X, y, cv=folds_data, scoring=self.scoring, n_jobs=-1)
                mean_score = cv_scores.mean()

                # Fit model trên toàn bộ dữ liệu để trả về estimator
                model.fit(X, y)

                self.logger.info(f"Hoàn tất chạy mặc định. Score trung bình ({self.scoring}): {mean_score:.4f}")

                return {
                    "best_params": {}, # Không có params cụ thể
                    "best_score": mean_score,
                    "best_estimator": model
                }

            # 2. Chọn phương pháp tuning
            search_engine = None
            if self.tuning_method == "grid_search":
                search_engine = GridSearchCV(
                    estimator = model,
                    param_grid = param_grid,
                    scoring = self.scoring,
                    cv = folds_data,
                    n_jobs = -1,
                    verbose = 1
                )
            elif self.tuning_method == "random_search":
                search_engine = RandomizedSearchCV(
                    estimator = model,
                    param_distributions = param_grid,
                    scoring = self.scoring,
                    cv = folds_data,
                    n_iter = 10,
                    n_jobs = 1,
                    random_state= 42,
                    verbose = 1
                )
            # 3. Fit mô hình:
            self.logger.info(f"Đang chạy fit()... Quá trình này có thể mất chút thời gian, vui lòng chờ...")
            search_engine.fit(X, y)

            # 4. Log kết quả
            self.logger.info(f"Tuning hoàn tất cho {model_name}.")
            self.logger.info(f"Điểm tốt nhất ({self.scoring}): {search_engine.best_score_:.4f}")
            self.logger.info(f"Bộ tham số tốt nhất: {search_engine.best_params_}")

            return {
                "best_params": search_engine.best_params_,
                "best_score": search_engine.best_score_,
                "best_estimator": search_engine.best_estimator_
                }

        except Exception as e:
            self.logger.error(f"Lỗi trong quá trình tuning: {e}")
            return None

class FeatureWrapperSelector:
    """
    Thực hiện lựa chọn đặc trưng bằng phương pháp Wrapper (RFE, Forward/Backward Selection).
    """
    def __init__(self, method = 'rfe', n_features_to_select = 15, base_model = None):
        """
        Khởi tạo bộ lựa chọn đặc trưng.

        Args:
            method (str): 'rfe', 'forward', hoặc 'backward'.
            n_features_to_select (int): Số lượng feature muốn giữ lại.
            base_model: Mô hình dùng để đánh giá feature (mặc định là RandomForest nếu None).
                    Lưu ý: RFE cần model có thuộc tính coef_ hoặc feature_importances_.
        """
        self.method = method
        self.n_features_to_select = n_features_to_select
        self.base_model = base_model
        self.logger = logging.getLogger(self.__class__.__name__)
        self.selected_columns = [] # Lưu tên các cột được chọn


        # Nếu không truyền model, mặc định Random Forest
        if self.base_model is None:
            self.base_model = RandomForestClassifier(random_state=42)


    def fit_transform(self, X, y):
        """
        Thực hiện lựa chọn đặc trưng và trả về dữ liệu.
        Args:
        X: Tập dữ liệu gồm các đặc trưng.
        y: Dữ liệu mục tiêu.
        Returns:
        X_selected: Tập dữ liệu đã được lựa chọn gồm các đặc trưng quan trọng.
        """
        self.logger.info(f"=== Bắt đầu Feature Selection (Method: {self.method.upper()}) ===")
        self.logger.info(f"Số lượng features ban đầu: {X.shape[1]}")
        self.logger.info(f"Số lượng features muốn giữ: {self.n_features_to_select}")

        try:
            selector = None

            # Chọn phương pháp Feature Selection
            if self.method == 'rfe':
                selector = RFE(
                    estimator=self.base_model,
                    n_features_to_select=self.n_features_to_select,
                    step = 1 # Mỗi bước bỏ 1 feature
                )
            elif self.method in ["forward", "backward"]:
                direction = 'forward' if self.method == 'forward' else 'backward'
                selector = SequentialFeatureSelector(
                    estimator=self.base_model,
                    n_features_to_select=self.n_features_to_select,
                    direction=direction,
                    scoring='f1', # Hoặc 'accuracy', 'roc_auc'
                    cv=3,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Phương pháp lựa chọn đặc trưng không hợp lệ: {self.method}")

            # Fit và Transform
            self.logger.info("Đang chạy thuật toán lựa chọn đặc trưng...")
            selector.fit(X, y)

            # Lấy các column được chọn
            selected_mask = selector.get_support()
            self.selected_columns = X.columns[selected_mask].tolist()

            # Tạo dataframe mới chứa các cột đã chọn
            X_selected = X[self.selected_columns]

            self.logger.info(f"Đã chọn được {len(self.selected_columns)} features: {self.selected_columns}")

            return X_selected


        except Exception as e:
            self.logger.error(f"Lỗi trong quá trình lựa chọn đặc trưng: {e}")
            raise e


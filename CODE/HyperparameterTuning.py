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
      # PHẦN THÊM VÀO
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
            "best_params": "default", # Không có params cụ thể
            "best_score": mean_score,
            "best_estimator": model
        }
      ### THÊM VÀO
      # 1. Lấy model và param grid tương ứng
      # model, param_grid = self._get_model_config(model_name)

      search_engine = None

      # 2. Chọn phương pháp tuning
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
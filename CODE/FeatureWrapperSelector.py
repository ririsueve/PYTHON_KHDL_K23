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
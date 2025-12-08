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
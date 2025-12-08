class ModelEvaluator:
  """
  Tính toán các chỉ số để đánh giá một mô hình.
  """
  def __init__(self, cv_results):
    """
    Khởi tạo ModelEvaluator với kết quả từ cross-validation.

    Args:
      cv_results: tuple (mô hình, y_test, y_pred) cho mỗi fold.
    """
    self.cv_results = cv_results
    self.logger = logging.getLogger(self.__class__.__name__)
    self._metrics = None

  # Biến việc tính toán thành một thuộc tính.
  # Gọi evaluator.metrics thay vì evaluator.evaluate_score()
  @property
  def metrics(self):
    """
    Tính toán các chỉ số cho từng fold và tổng hợp kết quả trung bình.
    Returns:
      Một dictionary chứa các chỉ số trung bình trên tất cả fold.
    """
    if self._metrics is None:
      self.logger.info("Tính toán metrics...")
      fold_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

      for i, (model, X_test, y_test, y_pred) in enumerate(self.cv_results):
        fold_results['accuracy'].append(accuracy_score(y_test, y_pred))
        fold_results['precision'].append(precision_score(y_test, y_pred, average="binary", zero_division=0))
        fold_results['recall'].append(recall_score(y_test, y_pred, average="binary", zero_division=0))
        fold_results['f1_score'].append(f1_score(y_test, y_pred, average="binary", zero_division=0))

      final_metrics = {}
      for metric, scores in fold_results.items():
        final_metrics[f"mean_{metric}"] = np.mean(scores) if scores else 0

      self._metrics = final_metrics
    return self._metrics
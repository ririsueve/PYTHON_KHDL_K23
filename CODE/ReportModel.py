class ReportModel:
  """
  Lưu các chỉ số đánh giá mô hình vào file txt
  """
  def __init__(self, file_name = "evaluation_report.txt"):
    """
    Khởi tạo ReportModel với tên file báo cáo.

    Args:
      file_name (str): Tên file text để lưu báo cáo.
    """
    self.file_name = file_name
    self.logger = logging.getLogger(self.__class__.__name__)

  def save_comparision(self, all_models_metrics):
    """
    Lưu bảng so sánh tất cả các model vào file.
    Args:
      all_models_metrics: dict { "model_name": {metrics...}, ... }
    """
    if not all_models_metrics:
      self.logger.warning("Không có kết quả nào để báo cáo.")
      return

    df_report = pd.DataFrame(all_models_metrics).T

    # Sắp xếp theo F1-score
    sort_key = "mean_f1_score" if "mean_f1_score" in df_report.columns else df_report.columns[0]
    df_report = df_report.sort_values(by=sort_key, ascending=False)

    # 1. Định dạng tiêu đề file báo cáo
    report_content = []
    report_content.append("="*80)
    report_content.append(f"{'BẢNG XẾP HẠNG HIỆU SUẤT MÔ HÌNH':^80}")
    report_content.append("="*80)
    report_content.append(df_report.to_string())
    report_content.append("\n" + "-"*80)

    # Tìm Model tốt nhất
    if not df_report.empty:
      best_model = df_report.index[0]
      report_content.append(f"\n>>> CHAMPION MODEL: {best_model.upper()}")
      report_content.append(f">>> {sort_key}: {df_report.loc[best_model, sort_key]:.4f}")

    full_text = "\n".join(report_content)
    # 3. In ra log
    self.logger.info("\n" + full_text)

    # 4. Lưu vào file
    try:
      with open(self.file_name, "w", encoding='utf-8') as f:
        f.write(full_text)
      self.logger.info(f"Đã lưu báo cáo tổng hợp tại: {self.file_name}")
    except Exception as e:
      self.logger.error(f"Không thể lưu file báo cáo: {e}")
import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

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

class Visualizer:
    """
    Vẽ các biểu đồ để so sánh giữa các mô hình.
    """
    def __init__(self, cv_results, model_name_prefix="", save_dir="RESULT"):
        """
        Khởi tạo Visualize với kết quả từ cross-validation.

        Args:
        cv_results: tuple (mô hình, X_test, y_test, y_pred) cho mỗi fold.
        model_name_prefix (str): Prefix để thêm vào tên mô hình.
        save_dir: thư mục lưu biểu đồ.
        """
        self.cv_results = cv_results
        self.prefix = model_name_prefix
        self.save_dir = save_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def plot_all(self):
        """Hàm để vẽ tất cả biểu đồ"""
        self.plot_confusion_matrix()
        self.plot_ROC_curve()

    def plot_confusion_matrix(self):
        """
        Vẽ confusion matrix.
        """
        self.logger.info("Đang vẽ Confusion Matrix")
        try:
            # 1. Gom tất cả y_true và y_pred từ các fold lại thành 1 list dài
            y_test_all = []
            y_pred_all = []

            for i, (model, X_test, y_test, y_pred) in enumerate(self.cv_results):
                y_test_all.extend(y_test)
                y_pred_all.extend(y_pred)

            # 2. Tính Confusion Matrix tổng
            cm = confusion_matrix(y_test_all, y_pred_all)

            # 3. Vẽ biểu đồ dùng Seaborn
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels= [0, 1], yticklabels= [0, 1])

            plt.title(f'Confusion Matrix - {self.prefix}')
            plt.ylabel('True')
            plt.xlabel('Pred')
            plt.tight_layout()

            # 4. Lưu hình ảnh
            filename = f"confusion_matrix_{self.prefix}.png"
            full_path = os.path.join(self.save_dir, filename)

            plt.savefig(full_path)
            self.logger.info(f"Đã lưu ảnh confusion_matrix_{self.prefix}.png")

            # plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ Confusion Matrix: {e}")

    def plot_ROC_curve(self):
        """
        Vẽ ROC curve.
        """
        self.logger.info("Đang vẽ ROC Curve")

        try:
            y_test_all = []
            y_score_all = []

            for model, X_test, y_test, y_pred in self.cv_results:
                # Đối với các mô hình cần predict_proba
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                else:
                    # Trường hợp SVM không dùng prob, ta dùng decision_function hoặc y_pred
                    if hasattr(model, "decision_function"):
                        y_score = model.decision_function(X_test)
                    else:
                        y_score = y_pred

                y_test_all.extend(y_test)
                y_score_all.extend(y_score)

            # Tính toán ROC curve và AUC
            fpr, tpr, thresholds = roc_curve(y_test_all, y_score_all)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.prefix}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)

            # Lưu hình ảnh
            filename = f"roc_{self.prefix}.png"
            full_path = os.path.join(self.save_dir, filename)

            plt.savefig(full_path)
            # plt.show()
            self.logger.info(f"Đã lưu ảnh roc_curve_{self.prefix}.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ ROC Curve: {e}")
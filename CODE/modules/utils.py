import pandas as pd
import logging
from sklearn.model_selection import KFold, train_test_split

class SystemLogger:
    """
    Cấu hình hệ thống log: ghi vào file và in ra màn hình.
    """
    def __init__(self, log_file="training_process.log"):
        """
        Hàm khởi tạo với tên file log.
        Args:
            log_file (str): Tên file log.
        """
        self.log_file = log_file
        self.setup_logging()

    def setup_logging(self):
        """
        Hàm thiết lập cấu hình logging.
        """
        # Xóa cấu hình cũ nếu có
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Định dạng log: [Thời gian] - [Tên Class/Module] - [Level] - Nội dung
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Cấu hình logging cơ bản
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'), # Ghi vào file
                logging.StreamHandler()  # In ra màn hình
            ]
        )

class DataLoader:
    """
    Quản lý việc tải file csv và tạo file dataframe từ file csv
    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self._dataframe = None

    # Biến việc load data thành một thuộc tính.
    # Giúp truy cập data bằng cách gọi: loader.data (thay vì loader.load_data())
    @property
    def data(self):
        """
        Trả về dataframe. Nếu chưa load thì thực hiện load.
        """
        if self._dataframe is None:
            self.logger.info(f"Đang tải dữ liệu từ: {self.file_name}")
        try:
            self._dataframe = pd.read_csv(self.file_name)
            self.logger.info(f"Tải thành công. Shape: {self._dataframe.shape}")
        except FileNotFoundError:
            self.logger.error(f"Lỗi: Không tìm thấy file '{self.file_name}'")
            raise
        except Exception as e:
            self.logger.error(f"Lỗi đọc file: {e}")
            raise e
        return self._dataframe

    @property
    def shape(self):
        """Trả về kích thước dữ liệu mà không cần truy cập trực tiếp biến private"""
        return self.data.shape if self._dataframe is not None else (0, 0)
  
class DataSplitter:
    """
    Quản lý việc chia dataframe thành các tập train, tập test băng k_fold_cross_validation
    """
    def __init__(self, dataframe,target_col = None):
        self.dataframe = dataframe
        self.target_col = target_col
        self.logger = logging.getLogger(self.__class__.__name__)

    def kfold_split_data(self, n_splits):
        """
        Chia train/test bằng k_fold_cross_validation

        Args:
        n_splits: số lượng fold

        Returns:
        folds_indices: Một danh sách các tuple, mỗi tuple chứa:
              (train_index, test_index) cho mỗi fold.
        """
        self.logger.info(f"Bắt đầu chia dữ liệu với KFold (n_splits={n_splits}).")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds_indices = []

        for train_index, test_index in kf.split(self.dataframe):
            folds_indices.append((train_index, test_index))

        self.logger.info(f"Đã chia thành {len(folds_indices)} folds thành công.")
        return folds_indices

    def simple_split(self, test_size=0.2):
        self.logger.info(f" Bắt đầu chuẩn bị cắt dữ liệu (Test size: {test_size})...")
        # Có target thì dùng, không có thì thôi
        y_stratify = None

        # Kiểm tra 2 điều kiện:
        # a. Có khai báo target_col không?
        # b. Cột đó có thực sự nằm trong bảng dữ liệu không?
        if self.target_col and self.target_col in self.dataframe.columns:
            self.logger.info(f"[INFO] Chia theo tỷ lệ cân bằng (Stratify) cột: {self.target_col}")
            y_stratify = self.dataframe[self.target_col]
        else:
            self.logger.info("[INFO] Chia ngẫu nhiên (Random Split)")

        # 3. Cắt
        train_df, test_df = train_test_split(
            self.dataframe,
            test_size=test_size,
            random_state=42,
            shuffle=True,
            stratify=y_stratify
        )
        return train_df, test_df
import pandas as pd
import numpy as np
import logging
import re
import os
from fuzzywuzzy import process
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, PowerTransformer, KBinsDiscretizer
from modules.utils import DataLoader, DataSplitter # Import từ file utils.py

#=======================================================================================
#-----------------------CLASS : DataCleaner --------------------------------------------
#=======================================================================================
class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Chuẩn hóa, làm sạch, và tiền xử lý dữ liệu thô (Raw Data) bao gồm: xử lý giá trị ngoại lai (Outliers),
    lỗi chính tả (Typos), giá trị thiếu (Missing Values), và logic nghiệp vụ.
    """
    def __init__(self, golden_specs=None, range_rules=None,cols_to_drop = None, fuzzy_threshold=90, max_drop_ratio=0.05,imputation_strategy='auto'):
        """
        Khởi tạo bộ làm sạch dữ liệu.
        Args:
            golden_specs (dict): Quy tắc chuẩn hóa giá trị phân loại (Key: Tên cột, Value: List giá trị chuẩn).
            range_rules (dict): Quy tắc giới hạn Min/Max cho cột số (Key: Tên cột, Value: (min, max)).
            cols_to_drop (list): Danh sách tên cột cần loại bỏ.
            fuzzy_threshold (int): Ngưỡng độ chính xác (0-100) cho Fuzzy Matching.
            max_drop_ratio (float): Tỷ lệ lỗi tối đa cho phép xóa hàng (Drop Row). 
                                    Nếu vượt ngưỡng này sẽ giữ lại và gán NaN.
            imputation_strategy (str): Chiến lược điền dữ liệu thiếu.
                - 'auto': Chỉ dùng Median (số) / Mode (chữ).
                - 'ffill': Ưu tiên Forward-fill trước, sau đó vét lại bằng Auto.
                - 'bfill': Ưu tiên Backward-fill trước, sau đó vét lại bằng Auto.
        """
        self.golden_specs = golden_specs if golden_specs else {}
        self.range_rules = range_rules if range_rules else {}
        self.fuzzy_threshold = fuzzy_threshold
        self.max_drop_ratio = max_drop_ratio
        self.cols_to_drop = cols_to_drop if cols_to_drop else []
        self.imputation_strategy = imputation_strategy
        self.logger = logging.getLogger(self.__class__.__name__)
      
        # Biến nội bộ để học tham số
        self.medians_ = {}
        self.modes_ = {}
        self.numeric_cols = []
        self.categorical_cols = []

    def __repr__(self):
        #1. Kiểm tra trạng thái
        # Nếu dict medians_ có dữ liệu -> Đã fit
        if hasattr(self, 'medians_') and self.medians_:
            status_icon = "ĐÃ FIT (Ready)"
            stats_info = f"{len(self.medians_)} cột số, {len(self.modes_)} cột chữ"
        else:
            status_icon = "CHƯA FIT (Not Fitted)"
            stats_info = "Cần chạy .fit(X) để học dữ liệu"

        #2. Tóm tắt các luật
        n_golden = len(self.golden_specs) if self.golden_specs else 0
        n_ranges = len(self.range_rules) if self.range_rules else 0

        #3. Trả về chuỗi kết quả
        return (
            f"DataCleaner(\n"
            f"  |__ Trạng thái:  {status_icon}\n"
            f"  |__ Kiến thức:   {stats_info}\n"
            f"  |__ Cấu hình:    [Fuzzy: {self.fuzzy_threshold}] - [Outlier Drop: {self.max_drop_ratio}]\n"
            f"                   [Luật chuẩn hóa: {n_golden}] - [Luật giới hạn: {n_ranges}]\n"
            f")"
        )

    # =========================================================================
    # GIAI ĐOẠN 1: FIT
    # =========================================================================
    def fit(self, X, y=None):
        """
        Học các tham số thống kê (Median, Mode) từ tập dữ liệu huấn luyện.
        Args:
            X (pd.DataFrame): Dữ liệu huấn luyện.
        Returns:
            self: Đối tượng đã được huấn luyện.
        """
        self.logger.info("=====[Cleaner] FIT: Bắt đầu quá trình học các thông số.=====")

        # 1. Kiểm tra đầu vào
        if not isinstance(X, pd.DataFrame):
             raise TypeError("Dữ liệu đầu vào phải là một Dataframe")
        self.logger.info("[Cleaner] FIT: Đang học các thông số.......")

        # 2. Reset bộ nhớ để cho phép lớp DataCleaner học lại nhiều lần, mà không bị ảnh hưởng bởi việc học trước đó
        self.medians_ = {}
        self.modes_ = {}

        # 3. Phân loại các cột : numeric_col và categorical_col
        self.numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # 4. Học các tham số cần dùng : Mean cho numeric_col, Mode cho categorical_col
        for col in self.numeric_cols:
            self.medians_[col] = X[col].median()
        for col in self.categorical_cols:
            mode_val = X[col].mode()
            if not mode_val.empty:
                # Lấy mode đầu tiên nếu tồn tại nhiều mode
                self.modes_[col] = mode_val.sort_values().iloc[0]
            else:
                # Nếu tất cả là NaN, mode là Unknow
                self.modes_[col] = "Unknown"

        self.logger.info("[Cleaner] FIT: Hoàn thành quá trình học tham số.")
        return self

    # =========================================================================
    # GIAI ĐOẠN 2: TRANSFORM
    # =========================================================================
    def transform(self, X):
        """
        Áp dụng quy trình làm sạch dữ liệu lên DataFrame.
        Quy trình:
        1. Xóa cột không cần thiết.
        2. Xóa hàng trùng lặp.
        3. Lấy trị tuyệt đối cột số (sửa lỗi nhập liệu âm).
        4. Chuẩn hóa text và sửa lỗi chính tả (Fuzzy).
        5. Xử lý logic cột Tuổi (AGE).
        6. Kiểm tra khoảng giá trị (Range Rules) -> Xóa hàng hoặc gán NaN.
        7. Điền giá trị thiếu (Imputation) theo chiến lược đã chọn.
        Args:
            X (pd.DataFrame): Dữ liệu cần xử lý.

        Returns:
            pd.DataFrame: Dữ liệu sạch.
        """
        self.logger.info(f"[Cleaner] TRANSFORM: Bắt đầu xử lý dữ liệu...")
        df = X.copy()
        df = self._drop_cols(df)
        df = self._remove_duplicates(df)
        df = self._abs_numerical_col(df)
        df = self._standardize_categories(df)
        df = self._fix_logic_age(df)
        df = self._enforce_numerical_ranges(df)
        df = self._impute_missing_values(df)

        self.logger.info("====[Cleaner] TRANSFORM: Hoàn thành quá trình quá xử lí dữ liệu.====")
        return df

    # =========================================================================
    # PHẦN 3: CÁC HÀM CON CHI TIẾT (HELPER FUNCTIONS)
    # =========================================================================
    def _drop_cols(self, df):
        """
        Loại bỏ các cột không cần thiết được chỉ định trong `self.cols_to_drop`.
        Args:
            df (pd.DataFrame): DataFrame đầu vào.
        Returns:
            pd.DataFrame: DataFrame sau khi đã xóa cột.
        """

        if self.cols_to_drop:
            df = df.drop(columns=self.cols_to_drop, errors='ignore')
            self.logger.info(f"Đã xóa các cột: {self.cols_to_drop}")
        return df

    def _remove_duplicates(self, df):
        """
        Loại bỏ các hàng trùng lặp hoàn toàn trong DataFrame.
        Args:
            df (pd.DataFrame): DataFrame đầu vào.
        Returns:
            pd.DataFrame: DataFrame duy nhất, không còn dòng trùng lặp.
        """
        n_rows_before = len(df)
        df = df.drop_duplicates()
        n_dups = n_rows_before - len(df)
        if n_dups > 0:
            self.logger.info(f"Đã xóa {n_dups} dòng trùng lặp.")
        return df

    def _abs_numerical_col(self, df):
        """
        Chuyển đổi tất cả giá trị âm trong cột số thành số dương (Trị tuyệt đối).
        Mục đích: Sửa lỗi nhập liệu (ví dụ: nhập giá tiền là -5000).
        Args:
            df (pd.DataFrame): DataFrame đầu vào.    
        Returns:
            pd.DataFrame: DataFrame với các cột số đã lấy trị tuyệt đối.
        """
        self.logger.info("Lấy trị tuyệt đối (Abs) cho cột số...")

        # Lọc ra các cột hợp lệ:
        valid_cols = [c for c in self.numeric_cols if c in df.columns]

        # Tính số giá trị và lấy trị tuyệt đối các cột
        for col in valid_cols:
            n_negative = (df[col] < 0).sum()
            if n_negative > 0:
                df[col] = df[col].abs()
                self.logger.info(f"   - {col}: Sửa {n_negative} giá trị")

        self.logger.info("Hoàn thành lấy trị tuyệt đối (Abs) cho cột số.")
        return df


    def _standardize_categories(self, df):
        """
        Chuẩn hóa dữ liệu phân loại dựa trên `golden_specs`.
        Quy trình:
        1. Chuyển về chữ thường (lowercase) và xóa khoảng trắng (strip).
        2. Chuyển các giá trị rỗng/null về NaN.
        3. Áp dụng Fuzzy Matching để sửa lỗi chính tả nếu giá trị không nằm trong danh sách chuẩn.
        Args:
            df (pd.DataFrame): DataFrame đầu vào.  
        Returns:
            pd.DataFrame: DataFrame đã chuẩn hóa cột phân loại.
        """
        self.logger.info("Đang chuẩn hóa dữ liệu phân loại (Standardizing)...")

        # Duyệt qua các cột có trong danh sách chuẩn hóa:
        for col, valid_list in self.golden_specs.items():

            # Chỉ xử lý nếu cột đó CÓ trong dữ liệu hiện tại
            if col in df.columns:
               # Xử lí theo logic khác, phải bỏ qua do chuẩn Age cũng nằm trong gold_specs
                if col == 'AGE':
                  continue
                # 1. Xử lý sơ bộ
                df[col] = df[col].astype(str).str.lower().str.strip()

                # 2. Dữ liệu rỗng (astype() biến rỗng thành giá trị string) thì chuyển thành để điền sau
                df[col] = df[col].replace(['nan','none','null',''],np.nan)

                # 3. Áp dụng Fuzzy Logic để sửa lỗi chính tả
                # (Chỉ chạy fuzzy trên các giá trị không bị NaN)
                mask_not_null = df[col].notna()

                # Apply fuzzy chỉ trên các dòng có dữ liệu
                df.loc[mask_not_null, col] = df.loc[mask_not_null, col].apply(
                    lambda x: self._fuzzy_helper(x, valid_list)
                )
        self.logger.info("Hoàn thành chuẩn hóa dữ liệu phân loại (Standardizing)")
        return df

    def _fix_logic_age(self, df):
        """
        Áp dụng logic làm sạch đặc thù cho cột 'AGE'.
        Trích xuất số từ chuỗi nhập liệu và ánh xạ vào các khoảng nhóm tuổi chuẩn (ví dụ: '16-25')
        Args:
            df (pd.DataFrame): DataFrame đầu vào.
        Returns:
            pd.DataFrame: DataFrame sau khi xử lý cột AGE.
        """
        if 'AGE' not in df.columns:
            return df

        self.logger.info("Đang xử lý logic đặc thù cột AGE...")

        # Làm sạch sơ bộ dữ liệu
        df['AGE'] = df['AGE'].astype(str).str.lower().str.strip()
        df['AGE'] = df['AGE'].replace(['nan', 'none', 'null', ''], np.nan)

        # Map tuổi bằng hàm helper
        valid_ranges = self.golden_specs['AGE']
        df['AGE'] = df['AGE'].apply(lambda x: self._age_logic_helper(x, valid_ranges))

        # Double check và xử lí lỗi còn sót lại bằng nan
        mask_invalid = ~df['AGE'].isin(valid_ranges)
        if mask_invalid.any():
            df.loc[mask_invalid, 'AGE'] = np.nan
        self.logger.info("Hoàn thành xử lý cột AGE")
        return df


    def _enforce_numerical_ranges(self, df):
        """
        Kiểm tra và xử lý giá trị ngoại lai thô dựa trên `range_rules`.
        Logic quyết định:
        1. Tính tỷ lệ dòng lỗi trên tổng số dòng.
        2. Nếu tỷ lệ lỗi < `max_drop_ratio`: XÓA các dòng lỗi (Drop Rows).
        3. Nếu tỷ lệ lỗi >= `max_drop_ratio`: GIỮ lại dòng, nhưng gán giá trị lỗi thành NaN (để Impute sau).
        Args:
            df (pd.DataFrame): DataFrame đầu vào. 
        Returns:
            pd.DataFrame: DataFrame đã xử lý ngoại lai thô.
        """
        self.logger.info(f"Enforcing Numerical Ranges (Ngưỡng xóa: {self.max_drop_ratio:.0%})")

        # 1. Tạo Global Mask (Lưu vết những dòng cần xử lý)
        # Ban đầu tất cả là False (Sạch)
        bad_rows_mask = pd.Series(False, index=df.index)

        # 2. Duyệt từng cột có luật
        for col, (min_v, max_v) in self.range_rules.items():
            if col not in df: continue
            # Ép kiểu số
            s = pd.to_numeric(df[col], errors='coerce')
            #  Khởi tạo mask cho cột này là Series False
            col_bad_mask = pd.Series(False, index=df.index)
            # Cộng dồn các điều kiện vi phạm
            if min_v is not None: col_bad_mask |= (s < min_v)
            if max_v is not None: col_bad_mask |= (s > max_v)
            # Nếu có lỗi ở cột này
            if col_bad_mask.any():
                err_count = col_bad_mask.sum()
                self.logger.warming(f"Cột '{col}': Phát hiện {err_count} giá trị vi phạm khoảng ({min_v} - {max_v}).")
                # a. Biến giá trị lỗi thành NaN NGAY LẬP TỨC, để nếu có giữ lại thì impute sau
                df.loc[col_bad_mask, col] = np.nan
                # b. Cập nhật vào sổ đen tổng
                bad_rows_mask |= col_bad_mask

        # 3. Tính toán thiệt hại
        if len(df) == 0: return df
        bad_count = bad_rows_mask.sum()
        bad_ratio = bad_count / len(df)

        # 4. Ra quyết định (Decision Making)
        # Trường hợp 1: Lỗi ít (trong ngưỡng cho phép) -> XÓA BỎ DÒNG
        if 0 < bad_ratio < self.max_drop_ratio:
            self.logger.info(f"==> QUYẾT ĐỊNH: Xóa {bad_count} dòng (Tỷ lệ {bad_ratio:.2%} < Ngưỡng).")
            # Trả về những dòng KHÔNG nằm trong bad_rows_mask
            return df[~bad_rows_mask]

        # Trường hợp 2: Lỗi nhiều (quá ngưỡng) -> GIỮ LẠI DÒNG
        if bad_count > 0:
            self.logger.info(f"==> QUYẾT ĐỊNH: Giữ lại {bad_count} dòng để điền Mode/Median (Tỷ lệ {bad_ratio:.2%} >= Ngưỡng).")
            # Trả về df nguyên vẹn (nhưng đã gán NaN) để bước sau Impute.
        return df

    def _impute_missing_values(self, df):
        """
        Điền giá trị thiếu theo chiến lược `imputation_strategy` kết hợp vét lại (fallback)
        Các chế độ:
        - 'ffill': Forward Fill trước -> vét lại phần còn sót bằng Median/Mode.
        - 'bfill': Backward Fill trước -> vét lại phần còn sót bằng Median/Mode.
        - 'auto': Chỉ sử dụng Median (cột số) và Mode (cột phân loại).
        Returns:
            pd.DataFrame: Dữ liệu sạch 100% không còn NaN.
        """
        self.logger.info("Điền giá trị thiếu...")
        if self.imputation_strategy == 'ffill':
            self.logger.info("Áp dụng Forward Fill.")
            df = df.ffill() 
            
        elif self.imputation_strategy == 'bfill':
            self.logger.info("Áp dụng Backward Fill.")
            df = df.bfill() 

        #Dành cho auto hoặc điền vào với những ô bị thiếu ở bằng bfill/ffill bằng Median/Mode 
        self.logger.info("Áp dụng Median/Mode cho các ô còn sót lại.")

        # Điền số bằng Median
        for col in self.numeric_cols:
            if col in df.columns:
                fill_val = self.medians_.get(col, 0)
                df[col] = df[col].fillna(fill_val)

        # Điền chữ bằng Mode
        for col in self.categorical_cols:
            if col in df.columns:
                fill_val = self.modes_.get(col, 'Unknown')
                df[col] = df[col].fillna(fill_val)
                
        return df
    # =========================================================================

    # --- HÀM HỖ TRỢ LOGIC (STATIC) ---
    def _fuzzy_helper(self, value, valid_list):
        """
        (Hàm hỗ trợ) Tìm giá trị chuẩn nhất trong danh sách dựa trên độ tương đồng văn bản (Fuzzy Logic).
        Args:
            value (str): Giá trị cần kiểm tra.
            valid_list (list): Danh sách các giá trị chuẩn.  
        Returns:
            str: Giá trị chuẩn nếu độ tương đồng >= `fuzzy_threshold`, ngược lại trả về NaN.
        """
        # 1. Nếu giá trị rỗng -> Trả về NaN
        if pd.isna(value):
            return np.nan
        # 2. Nếu giá trị đã nằm trong list chuẩn -> Giữ nguyên
        if value in valid_list:
            return value

        # 3. Dùng fuzzy để sửa lỗi
        match, score = process.extractOne(value, valid_list)
        if score >= self.fuzzy_threshold:
            return match

        # 5. Nếu không đủ điểm tin cậy -> Trả về NaN
        return np.nan


    @staticmethod
    def _age_logic_helper(val, valid_ranges):
        """
        (Static Method) Trích xuất số từ chuỗi nhập liệu và ánh xạ vào các khoảng nhóm tuổi quy định.
        Args:
            val (str/int): Giá trị tuổi đầu vào (có thể lẫn ký tự).
            valid_ranges (list): Danh sách các nhóm tuổi hợp lệ (để kiểm tra nhanh).
        Returns:
            str: Nhóm tuổi chuẩn (ví dụ '16-25') hoặc NaN nếu không hợp lệ.
        """
        if pd.isna(val): return np.nan

        # 2. Làm sạch sơ bộ
        val = str(val).strip()
        #  Nếu đúng chuẩn rồi thì trả về luôn
        if val in valid_ranges: return val

        try:
            match = re.search(r'\d+', val)
            # Nếu không có số thì trả về nan
            if not match: return np.nan

            # Lấy giá trị và mapping vào nhóm tương ứng
            num_str = match.group()
            n = int(num_str)
            if 16 <= n <= 25: return '16-25'
            if 26 <= n <= 39: return '26-39'
            if 40 <= n <= 64: return '40-64'
            if n >= 65: return '65+'

            return np.nan # Dưới 16 tuổi, tránh bị lỗi

        except Exception:
            return np.nan

        return np.nan

#=======================================================================================
#-----------------------CLASS : FeatureEngineer ----------------------------------------
#=======================================================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Class chịu trách nhiệm Feature Engineering (Tạo đặc trưng mới).
    Thực hiện các phép biến đổi logic, tương tác giữa các cột hoặc tổng hợp thông tin 
    để tạo ra các biến mới giúp mô hình học tốt hơn.
    """

    def __init__(self):
        """
        Khởi tạo FeatureEngineer.
        Hiện tại class này không yêu cầu tham số cấu hình phức tạp.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        pass

    # =========================================================================
    # GIAI ĐOẠN 1: FIT
    # =========================================================================
    def fit(self, X, y=None):
        """
        Phương thức fit theo chuẩn Scikit-learn.
        Lưu ý:
            Feature Engineering trong class này là 'stateless' (phi trạng thái).
            Các phép toán cố định và không phụ thuộc vào phân phối 
            của tập huấn luyện (không cần học Mean/Mode).
            Do đó, hàm này không làm gì cả ngoài việc trả về chính nó.
        Args:
            X (pd.DataFrame): Dữ liệu huấn luyện (không sử dụng).
            y (pd.Series, optional): Biến mục tiêu (không sử dụng).
        Returns:
            self: Trả về chính đối tượng này.
        """
        return self

    # =========================================================================
    # GIAI ĐOẠN 2: TRANSFORM 
    # =========================================================================
    def transform(self, X):
        """
        Thực hiện quy trình tạo đặc trưng mới trên dữ liệu.
        Quy trình:
        1. Kiểm tra tính hợp lệ của dữ liệu đầu vào.
        2. Gọi các hàm nội bộ để tạo các nhóm đặc trưng cụ thể (ví dụ: Family Features).
        Args:
            X (pd.DataFrame): Dữ liệu đầu vào cần tạo đặc trưng.
        Returns:
            pd.DataFrame: Dữ liệu đã được bổ sung các cột đặc trưng mới.
        """
        self.logger.info(">>>[FeatureEngineer] TRANSFORM: Đang tạo đặc trưng tương tác...")
        
        # Kiểm tra đầu vào
        if not isinstance(X, pd.DataFrame):
             self.logger.warning("[Warning] Input không phải DataFrame, feature engineer có thể lỗi.")
        df = X.copy()
        # Gọi lần lượt các hàm tính toán con
        df = self._create_family_features(df)
        return df

    # =========================================================================
    # CÁC HÀM TÍNH TOÁN CHI TIẾT (HELPER METHODS)
    # =========================================================================
    def _create_family_features(self, df):
        """
        Tính toán chỉ số 'FAMILY_STABILITY' (Sự ổn định gia đình).
        Logic:
            Chỉ số này là tổng hợp của các yếu tố liên quan đến gia đình và tài sản cơ bản.
            Công thức: FAMILY_STABILITY = MARRIED + CHILDREN + VEHICLE_OWNERSHIP
            (Các giá trị sẽ được ép kiểu về số, lỗi hoặc thiếu sẽ coi là 0).
        Args:
            df (pd.DataFrame): DataFrame đầu vào.
        Returns:
            pd.DataFrame: DataFrame đã được thêm cột 'FAMILY_STABILITY'.
        """
        fam_cols = ['MARRIED', 'CHILDREN', 'VEHICLE_OWNERSHIP']

        # Kiểm tra xem đủ cột để tính không
        if all(c in df.columns for c in fam_cols):
            # Ép kiểu số, nếu lỗi (text lạ) thì thành NaN, sau đó fillna(0)
            temp = df[fam_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

            # Công thức: Tổng điểm ổn định
            df['FAMILY_STABILITY'] = temp.sum(axis=1)
            self.logger.info("-> Đã tạo feature: FAMILY_STABILITY")
            
        return df
    
#=======================================================================================
#-----------------------CLASS : DataTransformer----------------------------------------
#=======================================================================================
class DataTransformer(BaseEstimator, TransformerMixin):
    """
    Class thực hiện việc biến đổi và chuẩn hóa và mã hóa dữ liệu (Data Transformation).
    1. Auto Scaling: Tự động chọn thuật toán chuẩn hóa (Power, Robust, MinMax) dựa trên độ lệch (Skewness) của dữ liệu.
    2. Outlier Handling: Xử lý ngoại lai bằng phương pháp Capping (kẹp biên) hoặc Log Transform.
    3. **Binning & Encoding:** Chia nhóm dữ liệu số và mã hóa dữ liệu phân loại.
    """
    def __init__(self,
                 scaling_strategy='auto',
                 default_outlier_strategy='capping',
                 capping_quantiles=(0.01, 0.99),
                 outlier_strategies=None,
                 binning_cols=None,
                 ordinal_mappings=None,
                 nominal_cols=None,
                 ignore_cols=None):
        """
        Khởi tạo Transformer.
        Args:
            scaling_strategy (str): Chiến lược chuẩn hóa ('auto', 'standard', 'robust', 'minmax', 'none').
            default_outlier_strategy (str): Chiến lược xử lý ngoại lai mặc định ('capping', 'log', 'none').
            capping_quantiles (tuple): Cặp phân vị (min, max) dùng để tính ngưỡng cắt ngọn (ví dụ: 1% và 99%).
            outlier_strategies (dict): Cấu hình xử lý ngoại lai riêng cho từng cột (Key: Tên cột, Value: Chiến lược).
            binning_cols (list): Danh sách các cột số cần chia thành nhóm (Bins).
            ordinal_mappings (dict): Mapping thứ tự cho biến Ordinal (Key: Tên cột, Value: Dict {giá trị: số}).
            nominal_cols (list): Danh sách các cột định danh cần One-Hot Encoding.
            ignore_cols (list): Danh sách các cột giữ nguyên, không xử lý.
        """
        self.scaling_strategy = scaling_strategy
        self.default_outlier_strategy = default_outlier_strategy
        self.capping_quantiles = capping_quantiles
        self.outlier_strategies = outlier_strategies if outlier_strategies else {}
        self.binning_cols = binning_cols if binning_cols else []
        self.ordinal_mappings = ordinal_mappings if ordinal_mappings else {}
        self.nominal_cols = nominal_cols if nominal_cols else []
        self.ignore_cols = ignore_cols if ignore_cols else []
        self.logger = logging.getLogger(self.__class__.__name__)

        # Các tham số sẽ được học (Attributes)
        self.binners_ = {}
        self.capping_limits_ = {}
        self.scalers_ = {}
        self.onehot_encoder_ = None
        self.numeric_cols_ = []

    def __repr__(self):
        is_fitted = hasattr(self, 'scalers_') and bool(self.scalers_)
        status_icon = "ĐÃ FIT (Ready)" if is_fitted else "CHƯA FIT (Not Fitted)"
        n_scaled = len(self.scalers_)
        
        return (
            f"DataTransformer(\n"
            f"  |___ Trạng thái:  {status_icon}\n"
            f"  |___ Chiến lược:  [Scale: '{self.scaling_strategy}'] - [Outlier: '{self.default_outlier_strategy}']\n"
            f"  |___ Cấu hình:    [Cần Scale: {n_scaled} cột] - [OneHot: {len(self.nominal_cols)} cột]\n"
            f")"
        )

    # =========================================================================
    # FIT (HỌC THAM SỐ)
    # =========================================================================

    def fit(self, X, y=None):
        """
        Học toàn bộ tham số từ dữ liệu huấn luyện.
        Quy trình:
        1. Nhận diện cột số.
        2. Học quy tắc Binning (chia nhóm).
        3. Học ngưỡng Outlier (Capping thresholds).
        4. Học One-Hot Encoder.
        5. Áp dụng tạm thời các bước trên để làm sạch dữ liệu -> Tính Skewness -> Học Scaler phù hợp.
        Args:
            X (pd.DataFrame): Dữ liệu huấn luyện.
        Returns:
            self: Đối tượng đã fit.
        """
        self.logger.info(f"[Transformer] FIT: Bắt đầu học tham số (Chiến lược Scale: {self.scaling_strategy})...")
        df = X.copy()

        # 1. Định danh cột số
        self._identify_numeric_cols(df)
        self.logger.info(f"->Đã xác định {len(self.numeric_cols_)} cột số cần chuẩn hóa.")

        # 2. Học Binning
        self._fit_binning(df)
        
        # 3. Học Outliers
        self._fit_outliers(df)
        
        # 4. Học Encoding
        self._fit_encoding(df)

        # 5. Học Scaling dựa trên dữ liệu đã được biến đổi sơ bộ
        self.logger.info("Đang chuẩn bị dữ liệu tạm để học Scaler...")
        df_transformed = self._apply_pre_scaling_transforms(df)
        self._fit_scaling(df_transformed)

        self.logger.info(">>> [Transformer] FIT: Hoàn thành.")
        return self

    def _apply_pre_scaling_transforms(self, df):
        """Hàm nội bộ: Áp dụng các biến đổi sơ bộ (không log) để chuẩn bị dữ liệu cho việc học Scaler."""
        df_temp = df.copy()
        df_temp = self._transform_binning(df_temp, logging_enabled=False)
        df_temp = self._transform_outliers(df_temp, logging_enabled=False)
        df_temp = self._transform_encoding(df_temp, logging_enabled=False)
        return df_temp

    # =========================================================================
    # TRANSFORM
    # =========================================================================
    def transform(self, X):
        """
        Áp dụng các biến đổi đã học lên dữ liệu (Train hoặc Test).
        Có ghi log chi tiết số lượng giá trị bị thay đổi (Capping).
        Args:
            X (pd.DataFrame): Dữ liệu cần biến đổi.
        Returns:
            pd.DataFrame: Dữ liệu đã được chuẩn hóa.
        """
        self.logger.info("[Transformer] TRANSFORM: Đang biến đổi dữ liệu thực tế...")
        df = X.copy()

        # Bật log cho các bước transform chính thức
        df = self._transform_binning(df, logging_enabled=True)
        df = self._transform_outliers(df, logging_enabled=True)
        df = self._transform_encoding(df, logging_enabled=True)
        df = self._transform_scaling(df, logging_enabled=True)
        
        self.logger.info("[Transformer] TRANSFORM: Hoàn tất.")
        return df

    # =========================================================================
    # 1. SETUP & BINNING (CHIA NHÓM)
    # =========================================================================
    def _identify_numeric_cols(self, df):
        """
        Xác định danh sách các cột số thực sự cần được chuẩn hóa (Scaling).
        Logic:
            Lấy tất cả cột số, TRỪ ĐI:
            - Các cột dùng để Binning (sẽ thành Ordinal).
            - Các cột Nominal (sẽ thành One-Hot).
            - Các cột nằm trong danh sách bỏ qua (Ignore).
        Args:
            df (pd.DataFrame): DataFrame đầu vào.
        """
        all_nums = df.select_dtypes(include=np.number).columns.tolist()
        self.numeric_cols_ = [c for c in all_nums
                              if c not in self.binning_cols
                              and c not in self.nominal_cols
                              and c not in self.ignore_cols]

    def _fit_binning(self, df):
        """
        Học quy tắc chia nhóm (Binning) cho các cột số liên tục.
        Sử dụng chiến lược Quantile để chia thành 5 nhóm có số lượng mẫu tương đương.
        Args:
            df (pd.DataFrame): Dữ liệu huấn luyện.
        """
        if not self.binning_cols: return
        self.logger.info("Học quy tắc chia nhóm (Binning)...")
        for col in self.binning_cols:
            if col in df.columns:
                # n_bins=5: Chia làm 5 khúc
                # strategy='quantile': Chia đều quân số cho mỗi khúc
                binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                binner.fit(df[[col]].dropna())
                self.binners_[col] = binner
                self.logger.info(f"Cột '{col}': Đã chia thành 5 nhóm (bins).")

    def _transform_binning(self, df, logging_enabled=False):
        """
        Áp dụng quy tắc chia nhóm đã học.
        Biến đổi giá trị số liên tục thành chỉ số nhóm (0, 1, 2, 3, 4).
        Args:
            df (pd.DataFrame): Dữ liệu cần biến đổi.
            logging_enabled (bool): Có ghi log hay không. 
        Returns:
            pd.DataFrame: Dữ liệu đã được chia nhóm.
        """
        if logging_enabled and self.binners_:
            self.logger.info("Áp dụng Binning...")
        
        for col, binner in self.binners_.items():
            if col in df.columns:
                df[col] = binner.transform(df[[col]]).flatten()
        return df

    # =========================================================================
    # 2. OUTLIER HANDLING (XỬ LÝ NGOẠI LAI)
    # =========================================================================
    def _get_strategy(self, col):
        """
        Lấy chiến lược xử lý ngoại lai cho một cột cụ thể.
        Ưu tiên cấu hình riêng (outlier_strategies), nếu không có thì dùng mặc định.
        """
        return self.outlier_strategies.get(col, self.default_outlier_strategy)

    def _fit_outliers(self, df):
        """
        Học các ngưỡng giới hạn (Lower/Upper Bound) để xử lý ngoại lai theo phương pháp Capping.
        Dựa trên phân vị (Quantiles) đã cấu hình (ví dụ: 1% và 99%).
        
        Args:
            df (pd.DataFrame): Dữ liệu huấn luyện.
        """
        self.logger.info(f"Học ngưỡng xử lý ngoại lai (Strategy mặc định: {self.default_outlier_strategy})...")
        q_min, q_max = self.capping_quantiles

        for col in self.numeric_cols_:
            if col in df.columns:
                strategy = self._get_strategy(col)
                if strategy == 'capping':
                    upper = df[col].quantile(q_max)
                    lower = df[col].quantile(q_min)
                    self.capping_limits_[col] = (lower, upper)
                    self.logger.info(f"      + Cột '{col}': Capping Range [{lower:.2f} - {upper:.2f}]")

    def _transform_outliers(self, df, logging_enabled=False):
        """
        Áp dụng xử lý ngoại lai dựa trên tham số đã học.
        Các phương pháp:
        1. Log Transform: Giảm độ lệch cho dữ liệu phân phối lệch phải (chỉ áp dụng số dương).
        2. Capping: Gán giá trị vượt ngưỡng về giá trị biên (Lower/Upper).
        Args:
            df (pd.DataFrame): Dữ liệu cần biến đổi.
            logging_enabled (bool): Có ghi log hay không.
            
        Returns:
            pd.DataFrame: Dữ liệu đã xử lý ngoại lai.
        """
        if logging_enabled:
            self.logger.info("Áp dụng xử lý ngoại lai (Outlier Handling)...")

        for col in self.numeric_cols_:
            if col in df.columns:
                strategy = self._get_strategy(col)

                # 1. Log Transform
                if strategy == 'log':
                    mask = df[col] > 0
                    if mask.any():
                        df.loc[mask, col] = np.log1p(df.loc[mask, col])
                        if logging_enabled:
                            self.logger.info(f"Log Transform: '{col}'")

                # 2. Capping
                elif strategy == 'capping':
                    if col in self.capping_limits_:
                        low, high = self.capping_limits_[col]
                        if logging_enabled:
                            # Đếm số lượng bị thay đổi để log cho dễ debug
                            n_capped = ((df[col] < low) | (df[col] > high)).sum()
                            if n_capped > 0:
                                self.logger.info(f"Capping '{col}': Sửa {n_capped} giá trị.")
                        
                        df[col] = df[col].clip(lower=low, upper=high)
        return df

    # =========================================================================
    # 3. ENCODING (MÃ HÓA BIẾN PHÂN LOẠI)
    # =========================================================================
    def _fit_encoding(self, df):
        """
        Học bộ mã hóa One-Hot (OneHotEncoder) cho các biến định danh (Nominal).
        """
        if self.nominal_cols:
            self.logger.info(f"Học One-Hot Encoding cho: {self.nominal_cols}")
            self.onehot_encoder_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.onehot_encoder_.fit(df[self.nominal_cols].astype(str))

    def _transform_encoding(self, df, logging_enabled=False):
        """
        Thực hiện mã hóa biến phân loại:
        1. Ordinal Encoding: Ánh xạ theo thứ tự điển (Map Dictionary).
        2. One-Hot Encoding: Biến đổi cột định danh thành vector nhị phân.
        Args:
            df (pd.DataFrame): Dữ liệu cần biến đổi.
        Returns:
            pd.DataFrame: Dữ liệu đã mã hóa.
        """
        if logging_enabled and (self.ordinal_mappings or self.nominal_cols):
             self.logger.info("Áp dụng Encoding (Ordinal & One-Hot)...")

        # Ordinal
        for col, mapping in self.ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)

        # One-Hot
        if self.nominal_cols and self.onehot_encoder_:
            vals = self.onehot_encoder_.transform(df[self.nominal_cols].astype(str))
            enc_df = pd.DataFrame(
                vals,
                columns=self.onehot_encoder_.get_feature_names_out(self.nominal_cols),
                index=df.index
            )
            df = pd.concat([df, enc_df], axis=1)
            df.drop(columns=self.nominal_cols, inplace=True)
        return df

    # =========================================================================
    # 4. SCALING (CHUẨN HÓA DỮ LIỆU)
    # =========================================================================
    def _fit_scaling(self, df):
        """
        Học tham số chuẩn hóa (Scaler) cho từng cột số dựa trên chiến lược `scaling_strategy`.
        Hàm hỗ trợ các tùy chọn (Options) sau:
        1. 'auto' (Chế độ thông minh): Tự động chọn thuật toán dựa trên độ lệch (Skewness) của dữ liệu.
        2. Chế độ thủ công (Manual):
           - 'standard': Luôn dùng StandardScaler (Z-score normalization).
           - 'robust': Luôn dùng RobustScaler (Scaling dựa trên IQR).
           - 'minmax': Luôn dùng MinMaxScaler.
           - 'none': Bỏ qua bước chuẩn hóa.
        Args:
            df (pd.DataFrame): Dữ liệu huấn luyện dùng để học Scaler.
        """
        self.logger.info("Học Scaler cho các cột số...")
        
        if self.scaling_strategy is None or self.scaling_strategy == 'none':
            self.logger.info("-> Scaling bị tắt (None).")
            return

        # 1: SỐ THỰC
        for col in self.numeric_cols_:
            if col in df.columns:
                scaler_name = ""
                
                # Logic Auto
                s = skew(df[col]) if len(df[col]) > 0 else 0
                if self.scaling_strategy == 'auto':
                    if abs(s) > 1.0: 
                        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                        scaler_name = f"PowerTransformer (Skew={s:.2f})"
                    elif abs(s) > 0.5: 
                        scaler = RobustScaler()
                        scaler_name = f"RobustScaler (Skew={s:.2f})"
                    else: 
                        scaler = MinMaxScaler()
                        scaler_name = f"MinMaxScaler (Skew={s:.2f})"
                
                # Logic Manual
                elif self.scaling_strategy == 'standard': 
                    scaler = StandardScaler()
                    scaler_name = "StandardScaler"
                elif self.scaling_strategy == 'robust': 
                    scaler = RobustScaler()
                    scaler_name = "RobustScaler"
                else: 
                    scaler = MinMaxScaler()
                    scaler_name = "MinMaxScaler "

                # Fit & Log
                scaler.fit(df[[col]].fillna(df[col].median()))
                self.scalers_[col] = scaler
                self.logger.info(f"Cột '{col}': Chọn {scaler_name}")

        # 2: THỨ TỰ (Binning/Ordinal luôn dùng MinMaxScaler)
        ordinal_cols = list(self.ordinal_mappings.keys()) + self.binning_cols
        if ordinal_cols:
            self.logger.info(f"Cột thứ tự/bin ({len(ordinal_cols)} cột): Dùng MinMaxScaler.")
            for col in ordinal_cols:
                if col in df.columns:
                    scaler = MinMaxScaler()
                    scaler.fit(df[[col]].fillna(0))
                    self.scalers_[col] = scaler

    def _transform_scaling(self, df, logging_enabled=False):
        """
        Áp dụng các bộ chuẩn hóa (Scaler) đã học lên dữ liệu.
        Đưa dữ liệu về cùng một miền giá trị hoặc phân phối chuẩn.
        Args:
            df (pd.DataFrame): Dữ liệu cần biến đổi.
            
        Returns:
            pd.DataFrame: Dữ liệu đã chuẩn hóa.
        """
        if logging_enabled and self.scalers_:
             self.logger.info("-> Áp dụng Scaling (Chuẩn hóa)...")
             
        for col, scaler in self.scalers_.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]]).flatten()
        return df
    
#=======================================================================================
#-----------------------CLASS :DataPreprationPipeline ----------------------------------
#=======================================================================================
class DataPreparationPipeline:
    """
    Class quản lý toàn bộ quy trình chuẩn bị dữ liệu (End-to-End Pipeline).
    1. DataLoader: Tải dữ liệu đa nguồn.
    2. DataCleaner & FeatureEngineer: Làm sạch và tạo đặc trưng (áp dụng toàn cục).
    3. DataSplitter: Chia tập dữ liệu.
    4. DataTransformer: Chuẩn hóa, mã hóa (Scale/Encode) theo nguyên tắc chống rò rỉ dữ liệu (Data Leakage).
    """

    def __init__(self,
                 file_path, 
                 cleaner,
                 featuring,
                 transformer,
                 target_col=None):
        """
        Khởi tạo Pipeline với các thành phần xử lý.

        Args:
            file_path (str): Đường dẫn tới file dữ liệu gốc (CSV, Excel, JSON).
            cleaner (DataCleaner): Đối tượng đã cấu hình để làm sạch dữ liệu.
            featuring (FeatureEngineer): Đối tượng đã cấu hình để tạo đặc trưng mới.
            transformer (DataTransformer): Đối tượng đã cấu hình để chuẩn hóa và mã hóa.
            target_col (str, optional): Tên cột biến mục tiêu (Target) dùng để chia tầng (Stratify Split).
        """
        self.file_path = file_path
        self.cleaner = cleaner
        self.featuring = featuring
        self.transformer = transformer
        self.target_col = target_col
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, test_size=0.2):
        """
        Kích hoạt luồng xử lý dữ liệu tự động.
        Quy trình chi tiết:
        1. Load: Tải dữ liệu thô từ file.
        2. Pre-processing: Làm sạch và tạo đặc trưng trên toàn bộ tập dữ liệu (để đảm bảo tính nhất quán logic).
        3. Splitting: Chia tập Train/Test (giữ nguyên tỷ lệ Target nếu có `target_col`).
        4. Transforming: - Học tham số (Fit) CHỈ trên tập Train.
           - Áp dụng tham số (Transform) cho cả Train và Test.
           -> Đảm bảo không lấy thông tin tương lai từ tập Test (Data Leakage Prevention).

        Args:
            test_size (float): Tỷ lệ phần trăm cho tập kiểm tra (mặc định 0.2 = 20%).

        Returns:
            tuple(pd.DataFrame, pd.DataFrame): 
                - df_train_final: Tập huấn luyện đã xử lý hoàn chỉnh.
                - df_test_final: Tập kiểm tra đã xử lý hoàn chỉnh.
        """
        self.logger.info("="*60)
        self.logger.info("[PIPELINE] KHỞI ĐỘNG HỆ THỐNG XỬ LÝ DỮ LIỆU")
        self.logger.info("="*60)

        #------------1: Load dữ liệu -------------------------------
        self.logger.info(f"1. Đang tải dữ liệu từ: {self.file_path}")
        # Sử dụng DataLoader thông minh (tự nhận diện file)
        loader = DataLoader(self.file_path)
        df = loader.data

        if df is None:
            self.logger.error("Dữ liệu tải về bị rỗng (None). Dừng Pipeline.")
            raise ValueError("Lỗi: Không tải được dữ liệu!")

        #------------2: Tiền xử lí dữ liệu (Sơ chế toàn cục) -------------------------------
        self.logger.info("2. Đang xử lí (Cleaner + Engineer)...")
        self.logger.info(f"-> Kích thước ban đầu: {df.shape} (Dòng, Cột)")
        # Làm sạch (Clean)
        df = self.cleaner.fit_transform(df)
        
        # Tạo đặc trưng (Engineer)
        df_clean = self.featuring.fit_transform(df)
        
        self.logger.info(f"-> Kết quả làm sạch và tạo thêm feature mới : {df_clean.shape} (Dòng, Cột)")

        #------------3: Chia tập train, test -------------------------------
        self.logger.info(f"3. Đang chia dữ liệu (Split Train/Test) với tỷ lệ Test={test_size}...")
        splitter = DataSplitter(df_clean, self.target_col)
        train_df, test_df = splitter.simple_split(test_size=test_size)

        #---------4: Transforming dữ liệu (Tinh chế) -------------------------------
        self.logger.info("4. Đang chuyển đổi dữ liệu (Transforming).")
        self.logger.info("-> Nguyên tắc: Học tham số từ Train, Áp dụng cho cả Train & Test.")

        # Fit (Học) chỉ trên Train
        self.transformer.fit(train_df)
        
        # Transform (Áp dụng) trên cả hai
        df_train_final = self.transformer.transform(train_df)
        df_test_final = self.transformer.transform(test_df)

        # -----------------------Hoàn thành----------------------------------
        self.logger.info("="*60)
        self.logger.info("[PIPELINE] HOÀN TẤT!")
        self.logger.info(f"   Train Set: {df_train_final.shape}")
        self.logger.info(f"   Test Set:  {df_test_final.shape}")
        self.logger.info("="*60)

        return df_train_final, df_test_final

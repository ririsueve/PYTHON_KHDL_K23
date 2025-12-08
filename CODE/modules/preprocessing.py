import pandas as pd
import numpy as np
import logging
import re
from fuzzywuzzy import process
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, PowerTransformer, KBinsDiscretizer
from modules.utils import DataLoader, DataSplitter # Import từ file utils.py

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, golden_specs=None, range_rules=None,cols_to_drop = None, fuzzy_threshold=90, max_drop_ratio=0.05):
        self.golden_specs = golden_specs if golden_specs else {}
        self.range_rules = range_rules if range_rules else {}
        self.fuzzy_threshold = fuzzy_threshold
        self.max_drop_ratio = max_drop_ratio
        self.cols_to_drop = cols_to_drop if cols_to_drop else []
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
            f"  ├── Kiến thức:   {stats_info}\n"
            f"  └── Cấu hình:    [Fuzzy: {self.fuzzy_threshold}] - [Outlier Drop: {self.max_drop_ratio}]\n"
            f"                   [Luật chuẩn hóa: {n_golden}] - [Luật giới hạn: {n_ranges}]\n"
            f")"
        )

    # =========================================================================
    # GIAI ĐOẠN 1: FIT
    # =========================================================================
    def fit(self, X, y=None):
        self.logger.info(">>> [Cleaner] FIT: Bắt đầu quá trình học các thông số.")

        # 1. Kiểm tra đầu vào
        if not isinstance(X, pd.DataFrame):
             raise TypeError("Dữ liệu đầu vào phải là một Dataframe")
        self.logger.info(">>> [Cleaner] FIT: Đang học các thông số.......")

        # 2. Reset bộ nhớ để cho phép lớp DataCleaner học lại nhiều lần, mà không bị ảnh hưởng bởi việc học trước đó
        self.medians_ = {}
        self.modes_ = {}

        # 3. Phân loại các cột : numeric_col và categorical_col
        self.numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # 4. Học các tham số cần dùng : Mean cho numeric_col, Mode cho categorical_col
        # Median
        for col in self.numeric_cols:
            self.medians_[col] = X[col].median()
        # Mode
        for col in self.categorical_cols:
            mode_val = X[col].mode()
            if not mode_val.empty:
                # Lấy mode đầu tiên nếu tồn tại nhiều mode
                self.modes_[col] = mode_val.sort_values().iloc[0]
            else:
                # Nếu tất cả là NaN, mode là Unknow
                self.modes_[col] = "Unknown"

        self.logger.info(">>> [Cleaner] FIT: Hoàn thành quá trình học tham số.")
        return self

    # =========================================================================
    # GIAI ĐOẠN 2: TRANSFORM
    # =========================================================================
    def transform(self, X):
        self.logger.info(f">>> [Cleaner] TRANSFORM: Bắt đầu xử lý dữ liệu...")
        df = X.copy()

        # Gọi lần lượt các hàm con xử lý từng phần việc
        df = self._drop_cols(df)
        df = self._remove_duplicates(df)
        df = self._abs_numerical_col(df)
        df = self._standardize_categories(df)
        df = self._fix_logic_age(df)
        df = self._enforce_numerical_ranges(df)
        df = self._impute_missing_values(df)

        self.logger.info(">>> [Cleaner] TRANSFORM: Hoàn thành quá trình quá xử lí dữ liệu.")
        return df

    # =========================================================================
    # PHẦN 3: CÁC HÀM CON CHI TIẾT (HELPER FUNCTIONS)
    # =========================================================================
    def _drop_cols(self, df):
        """Xóa cột không cần thiết"""

        if self.cols_to_drop:
            df = df.drop(columns=self.cols_to_drop, errors='ignore')
            self.logger.info(f">>>---Đã xóa các cột: {self.cols_to_drop}")
        return df

    def _remove_duplicates(self, df):
        """Xóa trùng lặp (Duplicates)"""
        n_rows_before = len(df)
        df = df.drop_duplicates()
        n_dups = n_rows_before - len(df)
        if n_dups > 0:
            self.logger.info(f"---Đã xóa {n_dups} dòng trùng lặp.")
        return df

    def _abs_numerical_col(self, df):
        """Lấy trị tuyệt đối (Sửa lỗi typo dấu trừ)"""
        self.logger.info("---Lấy trị tuyệt đối (Abs) cho cột số...")

        # Lọc ra các cột hợp lệ:
        valid_cols = [c for c in self.numeric_cols if c in df.columns]

        # Tính số giá trị và lấy trị tuyệt đối các cột
        for col in valid_cols:
            n_negative = (df[col] < 0).sum()
            if n_negative > 0:
                df[col] = df[col].abs()
                self.logger.info(f"   - {col}: Sửa {n_negative} giá trị")

        self.logger.info("---Hoàn thành lấy trị tuyệt đối (Abs) cho cột số.")
        return df


    def _standardize_categories(self, df):
        """
        Chuẩn hóa dữ liệu phân loại dựa trên Golden Specs.
        Quy trình: Lowercase -> Strip -> Map về giá trị chuẩn (Fuzzy).
        """
        self.logger.info("----Đang chuẩn hóa dữ liệu phân loại (Standardizing)...")

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

        return df

    def _fix_logic_age(self, df):
        """Xử lý logic đặc thù cho cột AGE"""
        if 'AGE' not in df.columns:
            return df

        self.logger.info("----Đang xử lý logic đặc thù cột AGE...")

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

        return df


    def _enforce_numerical_ranges(self, df):
        """
        Kiểm tra các cột số phải nằm trong khoảng cho phép (min-max). Nếu không thì biến thành Nan hoặc xóa
        1. Quét lỗi toàn bộ bảng.
        2. Nếu tỷ lệ lỗi thấp (< threshold): Xóa bỏ dòng lỗi (Drop).
        3. Nếu tỷ lệ lỗi cao (>= threshold): Giữ lại dòng, chỉ xóa giá trị sai (về NaN) để điền Mode sau.
        """
        self.logger.info(f"----Enforcing Numerical Ranges (Ngưỡng xóa: {self.max_drop_ratio:.0%})")

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
                self.logger.info(f"---Cột '{col}': Phát hiện {err_count} giá trị vi phạm khoảng ({min_v} - {max_v}).")
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
        """ Điền giá trị thiếu (Missing Values):
          - Điền Median với cột numerical_cols
          - Điềm Mode với categorical_cols
        """
        self.logger.info("----Điền giá trị thiếu...")
        # Điền số bằng Median
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.medians_.get(col, 0))

        # Điền chữ bằng Mode:
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.modes_.get(col, 'unknown'))
        return df

    # --- HÀM HỖ TRỢ LOGIC (STATIC) ---
    def _fuzzy_helper(self, value, valid_list):
        """
        Trả về từ chuẩn trong valid_list nếu độ giống >= threshold, nếu không trả về NaN
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

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        pass

    # =========================================================================
    # GIAI ĐOẠN 1: FIT
    # =========================================================================
    def fit(self, X, y=None):
        """
        Feature Engineering thường là 'stateless' (không cần học tham số từ dữ liệu).
        Ví dụ: A + B luôn là A + B, không phụ thuộc vào Mean/Mode của tập Train.
        Nên hàm này chỉ cần return self.
        """
        return self

    # =========================================================================
    # GIAI ĐOẠN 2: TRANSFORM 
    # =========================================================================
    def transform(self, X):
        self.logger.info(">>> [FeatureEngineer] TRANSFORM: Đang tạo đặc trưng tương tác...")
        # Kiểm tra đầu vào
        if not isinstance(X, pd.DataFrame):
             self.logger.info("   [Warning] Input không phải DataFrame, feature engineer có thể lỗi.")
             # Nếu X là numpy array (do bước trước trả về), cần convert lại DF (tùy tình huống)

        df = X.copy()

        # Gọi lần lượt các hàm tính toán con
        df = self._create_risk_features(df)
        df = self._create_family_features(df)

        return df

    # =========================================================================
    # CÁC HÀM TÍNH TOÁN CHI TIẾT (HELPER METHODS)
    # =========================================================================

    def _create_risk_features(self, df):
        """Hàm 1: Tính toán các chỉ số rủi ro hành vi"""
        # Danh sách cột cần dùng
        risk_cols = ['DUIS', 'PAST_ACCIDENTS', 'SPEEDING_VIOLATIONS']

        # Kiểm tra xem có đủ cột không
        if all(c in df.columns for c in risk_cols):
            # Fill 0 tạm thời để tính toán, đảm bảo không cộng với NaN
            # Ép kiểu numeric để tránh lỗi nếu dữ liệu đang là string
            temp = df[risk_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

            # 1.1 Tổng số sự cố
            df['TOTAL_INCIDENTS'] = temp.sum(axis=1)

            # 1.2 Điểm rủi ro trọng số (5-3-1)
            # Công thức: (DUI*5) + (Accidents*3) + (Speeding*1)
            df['WEIGHTED_RISK_SCORE'] = (temp['DUIS'] * 5) + \
                                        (temp['PAST_ACCIDENTS'] * 3) + \
                                        (temp['SPEEDING_VIOLATIONS'] * 1)
            self.logger.info("   -> Đã tạo feature: TOTAL_INCIDENTS, WEIGHTED_RISK_SCORE")
        else:
            self.logger.info(f"   [Skip] Không đủ cột để tạo Risk Features. Cần: {risk_cols}")
        return df

    def _create_family_features(self, df):
        """Hàm 2: Tính toán chỉ số ổn định gia đình"""
        fam_cols = ['MARRIED', 'CHILDREN', 'VEHICLE_OWNERSHIP']

        if all(c in df.columns for c in fam_cols):
            temp = df[fam_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

            # Công thức: Tổng điểm ổn định
            df['FAMILY_STABILITY'] = temp.sum(axis=1)
            self.logger.info("   -> Đã tạo feature: FAMILY_STABILITY")
        return df
    

class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 scaling_strategy='auto',
                 default_outlier_strategy='capping',
                 capping_quantiles=(0.01, 0.99),
                 outlier_strategies=None,
                 binning_cols=None,
                 ordinal_mappings=None,
                 nominal_cols=None,
                 ignore_cols=None):

        self.scaling_strategy = scaling_strategy
        self.default_outlier_strategy = default_outlier_strategy
        self.capping_quantiles = capping_quantiles
        self.outlier_strategies = outlier_strategies if outlier_strategies else {}
        self.binning_cols = binning_cols if binning_cols else []
        self.ordinal_mappings = ordinal_mappings if ordinal_mappings else {}
        self.nominal_cols = nominal_cols if nominal_cols else []
        self.ignore_cols = ignore_cols if ignore_cols else []
        self.logger = logging.getLogger(self.__class__.__name__)

        # Các tham số cần học : tên_ :
        self.binners_ = {}
        self.capping_limits_ = {}
        self.scalers_ = {}
        self.onehot_encoder_ = None
        self.numeric_cols_ = []

    def __repr__(self):
        # 1. Kiểm tra trạng thái đã fit chưa
        is_fitted = hasattr(self, 'scalers_') and bool(self.scalers_)
        if is_fitted:
            status_icon = "ĐÃ FIT (Ready)"
            # Tổng hợp kiến thức đã học
            n_scaled = len(self.scalers_)
            n_capped = len(self.capping_limits_)
            n_binned = len(self.binners_)
            stats_info = f"Đã học: [Scale: {n_scaled}] - [Cap: {n_capped}] - [Bin: {n_binned}]"
        else:
            status_icon = "CHƯA FIT (Not Fitted)"
            stats_info = "Cần chạy .fit(X_train) để học tham số"

        # 2. Tóm tắt cấu hình đầu vào
        n_nominal = len(self.nominal_cols)
        n_ordinal = len(self.ordinal_mappings)

        # 3. Trả về chuỗi định dạng đẹp
        return (
            f"DataTransformer(\n"
            f"  |___ Trạng thái:  {status_icon}\n"
            f"  |___ Chiến lược:  [Scale: '{self.scaling_strategy}'] - [Outlier: '{self.default_outlier_strategy}']\n"
            f"  |___ Cấu hình:    [OneHot: {n_nominal} cột] - [Ordinal: {n_ordinal} cột]\n"
            f"  |___ Kiến thức:   {stats_info}\n"
            f")"
        )

    # =========================================================================
    # FIT (HỌC THAM SỐ)
    # =========================================================================

    def fit(self, X, y=None):
        self.logger.info(f">>> [Transformer] FIT: Học tham số (Scale: {self.scaling_strategy})...")
        df = X.copy()

        # Học các tham số cơ bản
        self._identify_numeric_cols(df)
        self._fit_binning(df)
        self._fit_outliers(df)
        self._fit_encoding(df)

        # Muốn học tham số scaling thì phải thực hiện bước chuyển đổi, sau đó mới học
        # Chuẩn bị dữ liệu học scaling
        df_transformed = self._apply_pre_scaling_transforms(df)
        # Học dữ liệu cho scaling
        self._fit_scaling(df_transformed)
        self.logger.info("Hoàn thành quá trình học tham số")
        return self

    # -- Hàm hỗ trợ ----------------------------------------
    def _apply_pre_scaling_transforms(self, df):
        """
        Áp dụng tạm thời các biến đổi (Binning, Outlier, Encoding)
        để tạo ra dữ liệu sạch phục vụ cho việc học Scaler.
        """
        df_temp = df.copy()
        df_temp = self._transform_binning(df_temp)
        df_temp = self._transform_outliers(df_temp)
        df_temp = self._transform_encoding(df_temp)
        return df_temp

    # =========================================================================
    # TRANSFORM
    # =========================================================================
    def transform(self, X):
        self.logger.info(">>> [Transformer] TRANSFORM: Đang biến đổi dữ liệu...")
        df = X.copy()

        df = self._transform_binning(df)
        df = self._transform_outliers(df)
        df = self._transform_encoding(df)
        df = self._transform_scaling(df)
        self.logger.info(" [Transformer] TRANSFORM: Hoàn thành quá trình biến đổi dữ liệu.")
        return df

    # =========================================================================
    # 1. SETUP & BINNING
    # =========================================================================
    def _identify_numeric_cols(self, df):
        """
        Lọc ra các cột số thực sự cần chuẩn hóa (Scaling).
        (Loại trừ các cột sẽ dùng để chia nhóm, cột định danh hoặc cột cần bỏ qua).
        """
        all_nums = df.select_dtypes(include=np.number).columns.tolist()
        self.numeric_cols_ = [c for c in all_nums
                              if c not in self.binning_cols
                              and c not in self.nominal_cols
                              and c not in self.ignore_cols]

    def _fit_binning(self, df):
        """
        Học cách chia cột số thành 5 nhóm (bins) dựa trên mật độ dữ liệu (Quantile).
        """
        for col in self.binning_cols:
            if col in df.columns:
                # n_bins=5: Chia làm 5 khúc
                # strategy='quantile': Chia đều quân số cho mỗi khúc
                binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                binner.fit(df[[col]].dropna()) # dropna để học cho chuẩn
                self.binners_[col] = binner

    def _transform_binning(self, df):
        """
        Áp dụng bộ chia nhóm đã học: Biến đổi số liên tục thành số thứ tự nhóm (0, 1, 2, 3, 4).
        """
        for col, binner in self.binners_.items():
            if col in df.columns:
                df[col] = binner.transform(df[[col]]).flatten()
        return df


    # =========================================================================
    # 2. OUTLIER HANDLING (Logic tùy chỉnh)
    # =========================================================================
    # Hàm hỗ trợ lấy chiến lược cho từng cột
    def _get_strategy(self, col):
        # Ưu tiên lấy strategy riêng của cột, nếu không có thì lấy default
        return self.outlier_strategies.get(col, self.default_outlier_strategy)

    def _fit_outliers(self, df):
        """ Học ngưỡng cắt (lower, upper) dựa trên Quantile động """
        q_min, q_max = self.capping_quantiles

        for col in self.numeric_cols_:
            if col in df.columns:
                strategy = self._get_strategy(col)

                if strategy == 'capping':
                    # Tính toán động
                    upper = df[col].quantile(q_max)
                    lower = df[col].quantile(q_min)
                    self.capping_limits_[col] = (lower, upper)

                # Log không cần học nên bỏ qua

    def _transform_outliers(self, df):
        """ Áp dụng biến đổi log hoặc cắt ngọn(capping) """
        for col in self.numeric_cols_:
            if col in df.columns:
                strategy = self._get_strategy(col)

                # 1. Chiến lược LOG (Giảm độ lệch phân phối)
                if strategy == 'log':
                    # Chỉ log những số dương để tránh lỗi toán học
                    mask = df[col] > 0
                    if mask.any():
                        # np.log1p = log(x + 1) -> Tốt cho số gần 0
                        df.loc[mask, col] = np.log1p(df.loc[mask, col])

                # 2. Chiến lược CAPPING (Cắt ngọn)
                elif strategy == 'capping':
                    if col in self.capping_limits_:
                        low, high = self.capping_limits_[col]
                        df[col] = df[col].clip(lower=low, upper=high)

                # strategy == None: Giữ nguyên
        return df

    # =========================================================================
    # 3. ENCODING
    # =========================================================================
    def _fit_encoding(self, df):
        if self.nominal_cols:
            self.onehot_encoder_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.onehot_encoder_.fit(df[self.nominal_cols].astype(str))

    def _transform_encoding(self, df):
        # Ordinal
        for col, mapping in self.ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0) # Fill 0 nếu lạ

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
    # 4. SCALING
    # =========================================================================
    def _fit_scaling(self, df):
        # Nếu scaling_statrategy là None thì dừng lại ngày
        if self.scaling_strategy is None or self.scaling_strategy == 'none':
            return

        # 1: SỐ THỰC :chiến lược có thể là : Auto hoặc là theo ý chọn : Standard/Robust/MinMax
        for col in self.numeric_cols_:
            if col in df.columns:
                #Logic để chọn theo "auto"
                s = skew(df[col]) if len(df[col]) > 0 else 0
                if self.scaling_strategy == 'auto':
                    if abs(s) > 1.0: scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                    elif abs(s) > 0.5: scaler = RobustScaler()
                    else: scaler = MinMaxScaler()

                # Logic Manual
                elif self.scaling_strategy == 'standard': scaler = StandardScaler()
                elif self.scaling_strategy == 'robust': scaler = RobustScaler()
                else: scaler = MinMaxScaler()

                # Fit
                scaler.fit(df[[col]].fillna(df[col].median()))
                self.scalers_[col] = scaler

        # 2: THỨ TỰ (Ordinal + Binning) -> Chỉ dùng MinMaxScaler
        ordinal_cols = list(self.ordinal_mappings.keys()) + self.binning_cols
        for col in ordinal_cols:
            if col in df.columns:
                scaler = MinMaxScaler()
                scaler.fit(df[[col]].fillna(0))
                self.scalers_[col] = scaler


    def _transform_scaling(self, df):
        for col, scaler in self.scalers_.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]]).flatten()
        return df

class DataPreparationPipeline:
    """
    Cỗ máy tổng hợp:
    1. Load dữ liệu
    2. Sơ chế (Clean + Engineer) toàn cục
    3. Cắt Train/Test (Có cân bằng Target nếu muốn)
    4. Tinh chế (Scale/Encode)
    """
    def __init__(self,
                 file_path,          # Đường dẫn file CSV
                 cleaner,
                 featuring,       # Class Sơ chế (Cleaner + Engineer)
                 transformer,        # Class Tinh chế (Scaler + Encoder)
                 target_col=None):   # Tên cột Target (để chia Stratify)

        self.file_path = file_path
        self.cleaner = cleaner
        self.featuring = featuring
        self.transformer = transformer
        self.target_col = target_col
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, test_size=0.2):
        self.logger.info("="*60)
        self.logger.info("[PIPELINE] KHỞI ĐỘNG HỆ THỐNG XỬ LÝ DỮ LIỆU")
        self.logger.info("="*60)

        #------------1: Load dữ liệu -------------------------------
        self.logger.info(f"1. Đang tải dữ liệu từ: {self.file_path}")
        loader = DataLoader(self.file_path)
        df = loader.data

        if df is None:
            raise ValueError("Lỗi: Không tải được dữ liệu!")

        #------------2: Tiền xử lí dữ liệu -------------------------------
        self.logger.info("2. Đang xử lí (Cleaner + Engineer)...")
        df = self.cleaner.fit_transform(df)
        df_clean = self.featuring.fit_transform(df)
        self.logger.info(f"-> Kết quả: {df_clean.shape} (Dòng, Cột)")

        #------------3: Chia tập train, test -------------------------------
        self.logger.info("3. Đang chia dữ liệu (Split Train/Test)...")
        splitter = DataSplitter(df_clean, self.target_col)
        train_df, test_df = splitter.simple_split(test_size=test_size)

        #---------4: Transforming dữ liệu: Học tập train để chuyển đổi cả train, test----------
        self.logger.info("4.Đang chuyển đổi dữ liệu.")
        self.logger.info(" Nguyên tắc: Học từ Train, Áp dụng cho Test.")

        self.transformer.fit(train_df)
        df_train_final = self.transformer.transform(train_df)
        df_test_final = self.transformer.transform(test_df)

        # -----------------------Hoàn thành----------------------------------
        self.logger.info("="*60)
        self.logger.info("[PIPELINE] HOÀN TẤT!")
        self.logger.info(f"   Train Set: {df_train_final.shape}")
        self.logger.info(f"   Test Set:  {df_test_final.shape}")
        self.logger.info("="*60)

        return df_train_final, df_test_final
    

    

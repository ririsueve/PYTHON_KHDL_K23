## GIỚI THIỆU VỀ ĐỒ ÁN CỦA NHÓM:

Đây là hệ thống Machine Learning tự động (AutoML Pipeline) được thiết kế theo cấu trúc Modular, giúp tự động hóa quy trình từ xử lý dữ liệu, lựa chọn đặc trưng, tinh chỉnh tham số (Hyperparameter Tuning) đến huấn luyện và đánh giá nhiều mô hình khác nhau.

Mục tiêu chính: Dự đoán rủi ro (Risk Prediction) dựa trên dữ liệu hành vi và nhân khẩu học.

## CẤU TRÚC REPO:
```text
PYTHON_KHDL_K23/
│
├── 📂 CODE/                   # THƯ MỤC MÃ NGUỒN
│   ├── 📂 modules/            # Các module chức năng
│   │   ├── __init__.py
│   │   ├── utils.py          # Công cụ: Logger, Loader
│   │   ├── preprocessing.py  # Xử lý: Cleaner, Transformer
│   │   ├── modeling.py       # Model: Trainer, Tuner, Feature Selector
│   │   ├── evaluation.py     # Đánh giá: Evaluator, Reporter, Visualizer
│   │   └── pipeline.py       # Luồng chạy chính (AutoML)
│   │
│   ├── config.py             # FILE CẤU HÌNH
│   └── main.py               # FILE CHẠY CHÍNH
│
├── 📂 DATA/                   # THƯ MỤC DỮ LIỆU
│   ├── DATA_RISK_CLASSIFY.csv  # (File gốc)
│   ├── final_train_data.csv    # (File huấn luyện lưu tự động sau khi chạy code)
│   └── final_test_data.csv     # (File kiểm nghiệm lưu tự động sau khi chạy code)
│
├── 📂 RESULT/           # CHỨA MODEL ĐÃ TRAIN (.pkl) và BIỂU ĐỒ CÁC MÔ HÌNH
│   └── (Sẽ tự động tạo khi chạy code)
│
├── 📄 requirements.txt       # Danh sách thư viện cần thiết
├── 📄 README.md              # Hướng dẫn sử dụng
├── 📄 automl_run.log         # File logging lưu lại quá trình chạy
└── 📄 evaluation_report.txt  # File txt lưu lại chỉ số đánh giá mô hình
```

## CÁC BƯỚC CÀI ĐẶT:
1. Tải cả thư mục dự án PYTHON_KHDL_K23 về máy.
2. Mở Terminal/CMD tại thư mục dự án (VD: C:\Users\TenUsers\Downloads\PYTHON_KHDL_K23>)

3. Chạy lệnh cài đặt: pip install -r requirements.txt
   
4. Chạy lệnh chương trình chính: python CODE/main.py

## TÙY CHỈNH THAM SỐ TRUYỀN VÀO TRONG TERMINAL:

      Mặc định: tuning_method = "random_search", feature_method = "rfe", n_features = 15
          python CODE/main.py
    
Có thể lựa chọn phương pháp tuning, phương pháp lựa chọn đặc trưng, số đặc trưng cần giữ bằng cách gọi:

    Mặc định code đang để là 15, muốn giảm xuống 10 thì gõ:
        python CODE/main.py --n_features 10
        
    Mặc định là rfe, muốn đổi sang forward (chọn tiến) hoặc backward (chọn lùi):
        python CODE/main.py --feature_method forward
        
    Mặc định là random_search, muốn đổi sang grid_search hoặc default:
        python CODE/main.py --tuning grid_search hoặc
        python CODE/main.py --tuning default
## TÙY CHỈNH THAM SỐ CHO TIỀN XỬ LÍ DỮ LIỆU : config.py
### Tham số cho làm sạch (Cleaning)
Nhóm tham số này điều khiển Class `DataCleaner` với vai trò cốt lõi là **đảm bảo tính hợp lệ của dữ liệu thô và xử lý giá trị thiếu**.
| Tham số | Vai trò & Mục đích (Rút gọn) | Cách điều chỉnh (Ngắn gọn) |
| :--- | :--- | :--- |
| **`my_golden_specs`** | **Xác định giá trị CHUẨN** cho các cột phân loại (Category) để bắt lỗi nhập liệu. | Cập nhật danh sách giá trị chuẩn (ví dụ: `['male', 'female']`). |
| **`my_range_rules`** | **Giới hạn Min/Max hợp lý** cho các cột số (Numerical). | Thay đổi giới hạn số học (`(min, max)`) cho cột tương ứng. |
| **`imputation_strategy`** | **Chiến lược thay thế** cho các giá trị thiếu (NaN). | Chọn `'auto'`, `'ffill'`, `'bfill'`|
| **`cols_to_drop`** | Danh sách các cột **cần loại bỏ** (ví dụ: ID, cột rò rỉ). | Thêm hoặc xóa tên cột không cần thiết. |
| **`max_drop_ratio`** | **Ngưỡng an toàn** để xóa dòng (tối đa bao nhiêu % dòng bị lỗi có thể xóa). | Tăng/Giảm tỷ lệ (ví dụ: `0.05` = 5%) tùy theo độ quan trọng của dữ liệu. |
| **`fuzzy_threshold`** | **Ngưỡng sửa lỗi chính tả** tự động (độ tương đồng). | Điều chỉnh độ nghiêm ngặt khi sửa lỗi nhập liệu (thường là 80-95). |

### Tham số cho tranformation 
Nhóm tham số này điều khiển Class `DataTransformer` và `FeatureEngineer`, tập trung vào việc **Mã hóa, Xử lý Outlier và Chuẩn hóa Feature** cho mô hình.

| Tham số | Vai trò & Mục đích (Rút gọn) | Cách điều chỉnh (Ngắn gọn) |
| :--- | :--- | :--- |
| **`my_outlier_strategies`** | **Phương pháp điều chỉnh** cho các cột có phân phối lệch (Outlier). | Thay đổi loại chuyển đổi: `'log'`, `'capping'`, ` `None`. |
| **`my_ordinal_mappings`** | **Ánh xạ thứ tự** (Ranking) các cột Category thành số (Encoding). | Điều chỉnh lại thứ tự hoặc giá trị số của mapping. |
| **`nominal_columns`** | Danh sách các cột **không có thứ tự** (dùng One-Hot Encoding hoặc tương đương). | Thêm hoặc xóa các cột Nominal cần mã hóa. |
| **`scaling_strategy`** | **Thuật toán chuẩn hóa thang đo** (ví dụ: đưa về [0, 1]). | Chọn giữa `auto`,`'minmax'`, `'standard'`, hoặc `'robust'`. |
| **`ignore_cols`** | Các cột **KHÔNG** được áp dụng phép biến đổi (chủ yếu là cột Target). | Đảm bảo cột Target (`OUTCOME`) nằm trong danh sách này. |
## KẾT QUẢ CÀI ĐẶT:

Sau khi cài đặt:
- Các mô hình được lưu dưới dạng (.pkl) và các biểu đổ confusion matrix và ROC curve (.png) được lưu trong thư mục RESULT.
- Một file logging có tên automl_run.log được lưu trong thư mục chính.
- Một file txt lưu các chỉ số của từng mô hình có tên evaluation_report.txt được lưu trong thư mục chính.

## CÁC TÍNH NĂNG NỔI BẬT:

### 1. Tiền xử lý dữ liệu mạnh mẽ (preprocessing.py)

- Data Cleaner: Tự động sửa lỗi chính tả (Fuzzy Logic), xử lý ngoại lai (Outliers), chuẩn hóa dữ liệu bị thiếu (Missing Values).

- Feature Engineer: Tự động tạo đặc trưng mới (ví dụ: FAMILY_STABILITY).

- Data Transformer: Tự động mã hóa (One-Hot, Ordinal) và chuẩn hóa số liệu (Scaling).

### 2. Tự động lựa chọn đặc trưng (modeling.py)

- Hỗ trợ các phương pháp Wrapper: RFE, Forward Selection, Backward Selection.Tự động lọc ra top $K$ đặc trưng quan trọng nhất.

### 3. Đa dạng mô hình & Tinh chỉnh tham số (modeling.py)

- Hỗ trợ 6 thuật toán phổ biến: Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost, KNN.

- Tích hợp Grid Search và Random Search để tự động tìm bộ tham số tốt nhất.

### 4. Báo cáo & Lưu trữ (evaluation.py)
- Lưu Model: Xuất model ra file .pkl (dùng joblib) để tái sử dụng.

- Báo cáo: Tự động sinh file evaluation_report.txt so sánh hiệu suất các model.

- Biểu đồ: Tự động vẽ và lưu ảnh Confusion Matrix và ROC Curve.


   










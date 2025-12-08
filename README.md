## GIỚI THIỆU VỀ ĐỒ ÁN CỦA NHÓM:

Đây là hệ thống Machine Learning tự động (AutoML Pipeline) được thiết kế theo cấu trúc Modular, giúp tự động hóa quy trình từ xử lý dữ liệu, lựa chọn đặc trưng, tinh chỉnh tham số (Hyperparameter Tuning) đến huấn luyện và đánh giá nhiều mô hình khác nhau.

Mục tiêu chính: Dự đoán rủi ro (Risk Prediction) dựa trên dữ liệu hành vi và nhân khẩu học.

## CẤU TRÚC REPO:
```text
PYTHON_KHDL_K23/
│
├── 📁 modules/                # MÃ NGUỒN CHÍNH (SOURCE CODE)
│   ├── __init__.py           # Package init
│   ├── utils.py              # Công cụ: Logger, DataLoader, Splitter
│   ├── preprocessing.py      # Xử lý: Cleaner, Feature Engineer, Transformer
│   ├── modeling.py           # Model: Trainer, Tuner, Feature Selector
│   ├── evaluation.py         # Đánh giá: Evaluator, Reporter, Visualizer
│   └── pipeline.py           # Luồng chạy chính (AutoML)
│
├── 📁 saved_models/           # CHỨA MODEL ĐÃ TRAIN (.pkl)
│   ├── xgboost.pkl
│   ├── ...
│
├── 📄 config.py              # FILE CẤU HÌNH 
├── 📄 main.py                # FILE CHẠY CHÍNH
├── 📄 DATA_CLASSIFY.csv      # Dữ liệu huấn luyện gốc
├── 📄 transformed_train.csv  # Dữ liệu huấn luyện đã chuyển đổi
├── 📄 transformed_test.csv   # Dữ liệu kiểm nghiệm đã chuyển đổi
├── 📄 requirements.txt       # Danh sách thư viện cần thiết
└── 📄 README.md              # Hướng dẫn sử dụng
```

## CÁC BƯỚC CÀI ĐẶT:

1. Mở Terminal/CMD tại thư mục dự án.

2. Chạy lệnh cài đặt:
   pip install -r requirements.txt
   
3. Chạy lệnh chương trình chính:
   python main.py

## CÁC TÍNH NĂNG NỔI BẬT:

### 1. Tiền xử lý dữ liệu mạnh mẽ (preprocessing.py)

- Data Cleaner: Tự động sửa lỗi chính tả (Fuzzy Logic), xử lý ngoại lai (Outliers), chuẩn hóa dữ liệu bị thiếu (Missing Values).

- Feature Engineer: Tự động tạo đặc trưng mới (ví dụ: RISK_SCORE, FAMILY_STABILITY).

- Data Transformer: Tự động mã hóa (One-Hot, Ordinal) và chuẩn hóa số liệu (Scaling).

### 2. Tự động lựa chọn đặc trưng (modeling.py)

- Hỗ trợ các phương pháp Wrapper: RFE, Forward Selection, Backward Selection.Tự động lọc ra top $K$ đặc trưng quan trọng nhất.

### 3. Đa dạng mô hình & Tinh chỉnh tham số

- Hỗ trợ 6 thuật toán phổ biến: Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost, KNN.

- Tích hợp Grid Search và Random Search để tự động tìm bộ tham số tốt nhất.

### 4. Báo cáo & Lưu trữ (evaluation.py)
- Lưu Model: Xuất model ra file .pkl (dùng joblib) để tái sử dụng.

- Báo cáo: Tự động sinh file evaluation_report.txt so sánh hiệu suất các model.

- Biểu đồ: Tự động vẽ và lưu ảnh Confusion Matrix và ROC Curve.


   

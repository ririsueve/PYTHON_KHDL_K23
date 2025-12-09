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
│   ├── final_train_data.csv    # (File sạch sau khi chạy)
│   └── final_test_data.csv     # (File sạch sau khi chạy)
│
├── 📂 saved_models/           # CHỨA MODEL ĐÃ TRAIN (.pkl)
│   └── (Sẽ tự động tạo khi chạy code)
│
├── 📄 requirements.txt       # Danh sách thư viện cần thiết
└── 📄 README.md              # Hướng dẫn sử dụng
```

## CÁC BƯỚC CÀI ĐẶT:
1. Tải cả thư mục dự án PYTHON_KHDL_K23 về máy.
2. Mở Terminal/CMD tại thư mục dự án (VD: C:\Users\TenUsers\Downloads\PYTHON_KHDL_K23>)

3. Chạy lệnh cài đặt: pip install -r requirements.txt
   
4. Chạy lệnh chương trình chính: python CODE/main.py

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


   






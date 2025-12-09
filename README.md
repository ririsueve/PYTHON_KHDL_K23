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


   






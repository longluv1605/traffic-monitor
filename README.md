# ORBRO AI Developer Test - Option 1: Phân tích CCTV giao thông

## Tác giả

- Họ tên: Phạm Thành Long
- Email: [longtrong53@gmail.com](mailto:longtrong53@gmail.com)

## Tóm tắt

### Mô tả

Dự án này triển khai hệ thống phân tích video CCTV giao thông, đáp ứng yêu cầu của bài test ORBRO AI Developer (Option 1). Hệ thống được chia thành hai module:

- **`finetune.py`**: Fine-tune mô hình YOLOv8 trên dataset giao thông để phát hiện 6 lớp (`Bicycle`, `Bus`, `Car`, `Motorbike`, `Person`, `Truck`).

- **`traffic_monitor.py`**: Xử lý video CCTV, thực hiện phát hiện phương tiện, đếm số lượng từng lớp, theo dõi đa đối tượng, phân tích hành vi (xe đi ngược chiều), và hiển thị kết quả thời gian thực.

Code được viết bằng Python, tuân theo chuẩn **PEP8**, sử dụng YOLOv8 cho object detection, DeepSORT cho tracking, và tự động gán màu khác nhau cho từng lớp phương tiện để dễ phân biệt. Kết quả được lưu dưới dạng file CSV (hành vi) và TXT (log hiệu suất).

### Tính năng

#### Fine-tune (`finetune.py`)

- Fine-tune YOLOv8 trên dataset giao thông.
- Phát hiện 6 lớp: `Bicycle`, `Bus`, `Car`, `Motorbike`, `Person`, `Truck`.
- Đánh giá metrics: mAP@0.5, mAP@0.5:0.95, Precision, Recall.
- Lưu mô hình tại: `runs/train/vehicle_detection/weights/best.pt`.

#### Traffic Monitor (`traffic_monitor.py`)

- **Phát hiện phương tiện**: Sử dụng YOLOv8 để phát hiện và phân loại phương tiện.
- **Đếm số lượng từng lớp**: Thống kê số lượng riêng cho mỗi lớp (`Bus`, `Car`, v.v.).
- **Gán màu tự động**: Mỗi lớp phương tiện được gán một màu RGB khác nhau cho bounding box và text ID, tự động dựa trên số lượng class.
- **Theo dõi đa đối tượng**: Sử dụng DeepSORT để gán và duy trì ID duy nhất cho mỗi phương tiện.
- **Phân tích hành vi**: Phát hiện xe đi ngược chiều ("Wrong Way") dựa trên quỹ đạo di chuyển.
- **Hiển thị thời gian thực**: Overlay trên video bao gồm bounding box, track ID, FPS, và số lượng từng lớp.
- **Xuất kết quả**:
- `behaviors.csv`: Hành vi (`Track ID`, `Behavior`, `Frame`).
- `traffic_log.txt`: Log hiệu suất (Frame, FPS, Latency, số lượng từng lớp).
- Metrics trung bình: FPS, latency, số lượng từng lớp (in ra console).

## Cấu trúc thư mục

```plain
traffic_monitor/
├── datasets/
│   ├── train/
│   │   ├── images/         # Ảnh huấn luyện
│   │   └── labels/         # Nhãn YOLO
│   ├── valid/
│   │   ├── images/         # Ảnh xác thực
│   │   └── labels/
│   ├── test/
│   │   ├── images/         # Ảnh kiểm tra
│   │   └── labels/
│   └── data.yaml           # File cấu hình dataset
├── runs/
│   └── train/
│       └── vehicle_detection/
│           └── weights/
│               └── best.pt # Mô hình YOLOv8 đã fine-tune
├── videos/                 # Video CCTV giao thông
├── .env                    # File lưu trữ biến môi trường, tạo nếu muốn tải dataset từ Roboflow thông qua python
├── load_data.py            # Script tải dữ liệu từ roboflow
├── finetune.py             # Script fine-tune YOLOv8
├── traffic_monitor.py      # Script giám sát giao thông
├── behaviors.csv           # File CSV lưu hành vi
├── traffic_log.txt         # File log hiệu suất
├── config.yaml             # File thiết đặt cho project
├── environment.yaml        # File môi trường cho Anaconda
├── EVALUATION.md           # Tài liệu Trả lời Bài tập số 2
└── README.md               # Tài liệu Dự án

```

## Yêu cầu phần mềm

Đã cài đặt `Anaconda` trên máy tính.

## Cài đặt

1. **Tạo virtual environment**:

    ```bash
    conda env create -f environment.yaml
    ```

2. **Chạy virtual environment**:

    ```bash
    conda activate traffic-monitoring
    ```

## Chuẩn bị dữ liệu

1. **Dataset** (dùng trong `finetune.py`):

    Cách 1 (Tải thủ công):
    - Tạo thư mục `datasets/`
    - Tải dữ liệu từ [ROBOFLOW](https://universe.roboflow.com/fsmvu/street-view-gdogo/dataset/3) và copy vào thư mục datasets với cấu trúc như trên.

    Cách 2 (Chạy script): **`Cần tạo tài khoản trên Roboflow để lấy API KEY`**.

    ```bash
    python load_data.py
    ```

2. **Video** (dùng trong `traffic_monitor.py`):

    - Tạo thư mục `videos/`
    - Đặt file `*.mp4` trong thư mục.
    - Video có thể là file local, YouTube (tải về), hoặc luồng RTSP.
    - Cập nhật `video_path` trong `config.yaml`.

3. **Mô hình**:

    - Sau khi chạy `finetune.py`, mô hình được lưu tại `runs/train/vehicle_detection/weights/best.pt`.
    - Cập nhật `model_path` trong `config.yaml` để trỏ đến `best.pt`.

## Cách chạy

### 1. Fine-tune mô hình

- **File**: `finetune.py`
- **Mục đích**: Huấn luyện YOLOv8 và đánh giá trên tập validation.
- **Bước**:

1. Cập nhật `data_yaml` trong `config.yaml` (ví dụ: `dataset/data.yaml`).

2. Chạy:

    ```bash
    python finetune.py
    ```

- **Kết quả**:
  - Mô hình: `runs/train/vehicle_detection/weights/best.pt`.
  - Metrics: mAP@0.5, mAP@0.5:0.95, Precision, Recall (in ra console).

### 2. Chạy giám sát giao thông

- **File**: `traffic_monitor.py`
- **Mục đích**: Phân tích video CCTV, hiển thị và lưu kết quả.
- **Bước**:

1. Cập nhật `video_path` và `model_path` trong `config.yaml`.
2. Chạy:

    ```bash
    python traffic_monitor.py
    ```

- **Kết quả**:
  - **Video output**: Hiển thị thời gian thực với:
    - Bounding box và track ID (màu riêng cho từng lớp).
    - Số lượng từng lớp (`Bus: X`, `Car: Y`, v.v.).
    - FPS.
  - **behaviors.csv**: Hành vi (`Track ID`, `Behavior`, `Frame`), chỉ chứa "Wrong Way".
  - **traffic_log.txt**: Log mỗi frame (Frame, FPS, Latency, số lượng từng lớp).
  - **Metrics**: Average FPS, latency, số lượng trung bình mỗi lớp (console).

## Kết quả

### File đầu ra

1. **behaviors.csv**:
    - Cột: `Track ID`, `Behavior`, `Frame`.
    - Ví dụ:

        ```plain
        Track ID,Behavior,Frame
        1,Wrong Way,150
        2,Wrong Way,200
        ```

2. **traffic_log.txt**:
    - Cột: `Frame`, `FPS`, `Latency_ms`, và số lượng mỗi lớp (từ `model.names`).
    - Ví dụ (với 6 lớp):

        ```plain
        Frame,FPS,Latency_ms,Bus,Car,Motorcycle,Person,Truck
        0,15.23,65.72,1,3,2,0,1
        1,14.89,67.12,1,4,1,0,2
        ```

### Metrics

1. **Fine-tune** (in ra console trong `finetune.py`):
   - mAP@0.5: Độ chính xác trung bình (IoU=0.5).
   - mAP@0.5:0.95: Độ chính xác trung bình (IoU từ 0.5 đến 0.95).
   - Precision, Recall: Độ chính xác và độ bao phủ.
2. **Traffic Monitor** (in ra console trong `traffic_monitor.py`):
   - Average FPS: Tốc độ xử lý trung bình.
   - Average Latency: Thời gian xử lý mỗi frame (ms).
   - Average số lượng mỗi lớp: Số lượng trung bình của `Bus`, `Car`, v.v.
3. **Tracking và Behavior** (yêu cầu ground truth, hiện để trống):
   - MOTA (Multiple Object Tracking Accuracy).
   - Behavior Accuracy, FPR (False Positive Rate).

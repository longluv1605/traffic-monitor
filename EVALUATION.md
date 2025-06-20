# Bài 2: Tự đánh giá triển khai - ORBRO AI Developer Test

## Thí sinh

### Phạm Thành Long ([longtrong53@gmail.com](longtrong53@gmail.com))

Dưới đây là phần tự đánh giá pipeline phân tích CCTV giao thông (Bài 1), bao gồm các mục chưa thực hiện đầy đủ, lý do, và khả năng giải quyết, theo yêu cầu của bài test ORBRO AI Developer.

---

## 2-1: Những phần chưa thực hiện hoặc làm chưa đầy đủ

1. **Khả năng detecting và tracking còn chưa chính xác**:
   - Chưa cải thiện được độ chính xác của tác vụ detecting và tracking các đối tượng như `Person`,... Ngoài ra, vùng detect còn chưa được hoàn thiện.
2. **Chưa đánh giá đầy đủ tracking và behavior analysis**:
   - Hàm `evaluate_tracking_metrics` (MOTA) và `evaluate_behavior_metrics` (Accuracy, FPR) yêu cầu ground truth, nhưng hiện để trống do thiếu dữ liệu nhãn.
3. **Chưa xử lý vùng quan tâm (ROI)**:
   - Phát hiện xe đi ngược chiều ("Wrong Way") áp dụng cho toàn khung hình, không giới hạn trong vùng cụ thể (ví dụ: làn đường).
   - Ngoài ra, mới chỉ giả định rằng xe đi hướng lên (theo chiều màn hình) là ngược chiều, chưa áp dụng xử lý thực tế.
4. **Chưa tối ưu hiệu suất trên CPU**:
   - FPS trên CPU thấp (dưới 30 FPS với `IMGSZ=640`), trong khi pipeline ưu tiên GPU (70 FPS).
5. **Xử lý chồng lấn đối tượng còn sơ xài**:
   - Trong trường hợp phương tiện chồng lấn (occlusion), YOLOv8 và DeepSORT có thể mất tracking hoặc phát hiện sai.
6. **Chưa tối ưu hóa và kiến trúc hệ thống**

---

## 2-2: Lý do chưa thực hiện

1. **Khả năng detecting và tracking còn chưa chính xác**:
   - **Lý do**:
      - Dataset không đủ khái quát và đa dạng, phân bố của dataset có thể khác với phân bố của vật thể trong video.
      - Mô hình chưa đủ mạnh để có thể detect được hoàn hảo.
2. **Chưa đánh giá tracking và behavior**:
   - **Lý do**: Dataset không cung cấp ground truth cho tracking (track ID) hoặc hành vi ("Wrong Way"). Thiếu thời gian để tự tạo nhãn thủ công.
3. **Chưa xử lý ROI**:
   - **Lý do**: Yêu cầu bài test chỉ đề cập "Wrong Way" chung, không có thông tin về cấu trúc làn đường trong video mẫu. Thiếu thời gian để phân tích video.
4. **Chưa tối ưu trên CPU**:
   - **Lý do**: Ưu tiên độ chính xác với `IMGSZ=640` trên GPU. Tối ưu CPU cần thử nghiệm thêm (giảm `IMGSZ`, đổi mô hình), nhưng thời gian hạn chế.
5. **Chưa xử lý chồng lấn đối tượng**:
   - **Lý do**: Chồng lấn là vấn đề phức tạp, yêu cầu cải tiến mô hình (ví dụ: Non-Max Suppression nâng cao) hoặc thêm post-processing. Thiếu thời gian để triển khai.
6. **Chưa tối ưu hệ thống**:
   - **Lý do**: Cài đặt và thiết kế phức tạp, cần có nhiều thời gian để triển khai.

---

## 2-3: Phân loại vấn đề

### Có thể giải quyết

1. **Tăng cường độ chính xác của hệ thống**:
   - **Giải pháp**: Thay đổi mô hình nếu có phần cứng đủ mạnh, cải thiện các thuật toán xử lý ảnh.
   - **Khả thi**: Ảnh hưởng bởi hệ thống, có thể khả thi.
2. **Đánh giá tracking và behavior**:
   - **Giải pháp**: Tạo nhãn thủ công cho một số frame video hoặc dùng công cụ như CVAT để tạo ground truth. Sau đó, tính MOTA, Accuracy, FPR.
   - **Khả thi**: Phụ thuộc vào công cụ và thời gian tạo nhãn, nhưng khả thi.
3. **Xử lý ROI**:
   - **Giải pháp**: Vẽ ROI (hình chữ nhật) trong `traffic_monitor.py` và chỉ phân tích "Wrong Way" trong vùng này. Có thể dùng OpenCV để người dùng chọn ROI.
   - **Khả thi**: Đơn giản, chỉ cần thêm logic kiểm tra tọa độ.
4. **Tối ưu CPU**:
   - **Giải pháp**: Giảm `IMGSZ` (ví dụ: 416) trong `config.yaml` hoặc dùng mô hình nhẹ hơn. Tối ưu DeepSORT bằng cách giảm `nn_budget`.
   - **Khả thi**: Dễ thực hiện, chỉ cần điều chỉnh tham số.
5. **Xử lý chồng lấn đối tượng**:
   - **Giải pháp**: Cải thiện Non-Max Suppression trong YOLOv8 hoặc thêm post-processing để lọc detection sai. Tăng `max_age` trong DeepSORT để duy trì tracking qua occlusion.
   - **Khả thi**: Cần thử nghiệm, nhưng có thể đạt cải thiện đáng kể.
6. **Tối ưu hóa và kiến trúc hệ thống**:
   - **Giải pháp**: Cài đặt và thiết kế hệ thống sử dụng các công cụ đã đề cập, tối ưu tốc độ inference, đặc biệt sao cho phù hợp với các edge device.
   - **Khả thi**: Cần thử nghiệm.

### Không thể giải quyết

- Không có vấn đề nào hoàn toàn không thể giải quyết với thời gian và tài nguyên phù hợp. Tất cả các vấn đề trên đều có giải pháp kỹ thuật khả thi.

---

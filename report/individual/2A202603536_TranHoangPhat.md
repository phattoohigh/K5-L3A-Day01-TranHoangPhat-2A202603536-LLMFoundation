# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** 8462; thử thách #3.
- Dữ liệu sau bỏ trùng: 5.169 tin, gồm 4.516 ham và 653 spam. Train/val/test: 3.619/775/775; SUB: 40 tin cân bằng.
- Nguồn số liệu: notebook đã chạy và `lab1_report.json` được xuất từ Colab. Không chạy lại API khi chuẩn bị bộ nộp này.

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| [day01_lab.ipynb](day01_lab.ipynb) | Có code, câu trả lời và output từ Colab |
| [lab1_report.json](lab1_report.json) | 15/15 check đạt; đủ 10 dự đoán; SEED khớp |
| [hand_calc.jpg](hand_calc.jpg) | Ảnh viết tay có bước tính cho 5 bài |

Các check là kết quả đã lưu, không chứng minh đầu ra LLM thật đã hợp lệ. Một số dự đoán và thí nghiệm được ghi trong cùng giây nên JSON chỉ xác nhận chúng không có thời gian muộn hơn thí nghiệm, không chứng minh thứ tự ở độ phân giải nhỏ hơn giây.

## 2. Kết quả chính

| Phương pháp | F1 test | F1 SUB 40 | Train (s) | Tin/s cả lô | p95 (ms/tin) | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.877 | 0.824 | 9.44 | 12798.04 | 0.37 | 0.00000 |
| 2. BoW + MLP (deep learning) | 0.907 | 0.919 | 10.72 | 7427.85 | 6.71 | 0.00000 |
| 3a. Encoder (MiniLM) + kNN | 0.880 | 0.919 | 83.01 | 25.58 | 76.76 | 0.00000 |
| 3b. Encoder (MiniLM) + LogReg | 0.939 | 0.947 | 83.50 | 29.66 | 51.60 | 0.00000 |
| 3c. Encoder (MiniLM) + MLP head | 0.960 | 0.974 | 84.15 | 30.67 | 40.62 | 0.00000 |
| 4a. LLM nhỏ zero-shot | — (chỉ SUB) | 0.000 | 0.00 | 1.24 | 1195.29 | 0.00806 |
| 4b. LLM lớn zero-shot | — (chỉ SUB) | 0.000 | 0.00 | 1.09 | 1253.85 | 0.08057 |
| 4c. LLM nhỏ few-shot | — (chỉ SUB) | 0.000 | 0.00 | 1.08 | 1546.71 | 0.02072 |

Hàng 1–3 đánh giá toàn bộ test; chỉ F1 SUB được dùng để so với LLM. Chi phí là ước tính theo số từ và bảng giá trong template, không phải hóa đơn API. Thời gian chuẩn bị hàng embedding gồm encode train; kNN không tối ưu trọng số. Chi phí API bằng 0 của mô hình cục bộ chưa bao gồm hạ tầng.

Pareto trên SUB gồm BoW + LogReg, BoW + MLP và MiniLM + MLP head. MiniLM + MLP head vừa nhanh hơn vừa có F1 SUB cao hơn MiniLM + kNN. LLM có `invalid=40` trong mỗi lượt phân loại SUB, bị fallback thành ham nên F1 spam bằng 0; chưa thể dùng kết quả này để xếp hạng năng lực LLM. Nguyên nhân đầu ra không hợp lệ chưa được xác định vì chưa lưu phản hồi thô của các lượt phân loại.

## 3. Dự đoán và thực tế

1. **MLP so với LogReg:** dự đoán MLP kém hơn hoặc tương đương. Thực tế F1 test tăng từ 0,877 lên 0,907; hidden layer có thể học tổ hợp từ. Val loss thấp nhất ở epoch 5, nên vẫn có dấu hiệu overfitting khi train lâu. Chưa có nhiều lần chạy để kết luận ưu thế ổn định.
2. **Tìm kiếm embedding:** dự đoán hàng xóm gần nhất là ham. Thực tế cả 6 hàng xóm được in ra đều là spam; mẫu gần nhất có cosine 0,563. Encoder vẫn nhận ra ngữ nghĩa nhận thưởng/du lịch dù cách diễn đạt thay đổi.
3. **Stress test:** dự đoán BoW đạt 60–75%, embedding 75–85%, LLM 80–90% accuracy. Thực tế BoW đạt 52–58%, embedding 62–74%; LLM đạt 42% nhưng đang lỗi đầu ra. Tôi đã đánh giá quá cao khả năng tổng quát trên các bẫy viết tay. Dự đoán few-shot tăng chi phí 2–4 lần khớp tỷ lệ đo được khoảng 2,57 lần; dự đoán cải thiện F1 chưa kiểm chứng được vì toàn bộ phản hồi không hợp lệ.

## 4. Phân tích 50 case khó

| Phương pháp | F1 SUB thường | F1 50 case khó | Accuracy 50 case |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0,824 | 0,333 | 52% |
| BoW + MLP | 0,919 | 0,462 | 58% |
| Embedding + kNN | 0,919 | 0,772 | 74% |
| Embedding + LogReg | 0,947 | 0,640 | 64% |
| Embedding + MLP head | 0,974 | 0,612 | 62% |
| LLM nhỏ zero-shot | 0,000 | 0,000 | 42% |
| LLM nhỏ + prompt phòng thủ | — (không đo riêng trên SUB) | 0,000 | 42% |

Số liệu tập khó được đọc từ output notebook, không nằm trong `results` của JSON. BoW + LogReg tụt nhiều nhất: khoảng 49,1 điểm phần trăm F1. Test thường có cùng phân phối train và vẫn có thể chứa các mẫu gần trùng, nên chưa phản ánh đủ các kiểu đánh lừa.

BoW đạt 10–20% ở spam giọng tự nhiên và 17% ở né từ khóa vì dựa vào token trong vocab. Embedding + kNN đạt 40% ở spam tự nhiên nhưng chỉ 25% ở nhóm nói về spam: gần nghĩa chưa chắc cùng nhãn. Hai case cụ thể mà tất cả model đều sai trong output:

- **#12:** mở đầu lịch sự như làm quen, sau đó dẫn đến hồ sơ hẹn hò. Các dấu hiệu hội thoại có thể khiến mô hình bỏ sót mục đích quảng cáo.
- **#13:** nhắc nhận ảnh sự kiện như một thông báo thân thiện nhưng dẫn đến trang bán bản in không được yêu cầu. Ngữ cảnh giao dịch hợp lệ và quảng cáo không mong muốn dễ bị nhầm.

Hai phương án LLM 4a/4d đều đạt 42% tổng thể và khoảng 17% ở prompt injection. Kết quả này chưa chứng minh prompt phòng thủ hiệu quả hay thất bại do injection, vì lỗi đầu ra có thể gây dự đoán toàn ham. Ngoài prompt, có thể kiểm tra chéo bằng bộ phân loại độc lập. Chuẩn hóa chữ biến dạng có thể hỗ trợ BoW nhưng cũng có nguy cơ sửa nhầm tin hợp lệ. Bộ 50 case là phép thử độ bền, không đại diện đầy đủ spam ngoài đời.


## 5. Thử thách cá nhân

Đề #3: đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Cùng đầu vào BoW 2.000 chiều, seed, Adam, 15 epoch, learning rate 0,001 và batch size 64.

| Hidden | Tham số | F1 test | Train (s) |
| --- | ---: | ---: | ---: |
| 4 | 8.009 | 0,918 | 3,33 |
| 512 | 1.025.025 | 0,913 | 28,99 |

Mạng nhỏ train nhanh hơn khoảng 8,71 lần và có F1 cao hơn nhẹ 0,005 ở lần chạy này. Chọn mạng nhỏ khi cần tiết kiệm bộ nhớ và thời gian; chênh lệch một lần chạy chưa chứng minh ưu thế tổng quát. Số tham số được tính bằng `input_dim × hidden + hidden + hidden + 1`.


## 6. Phương án cho sản phẩm thật

BoW + LogReg (p95 0,37 ms), BoW + MLP (6,71 ms) và MiniLM + MLP head (40,62 ms) đạt điều kiện p95 dưới 50 ms ở cấu hình đo. MiniLM + MLP head có F1 test cao nhất 0,960, F1 SUB 0,974, nhưng thông lượng cả lô chỉ 30,67 tin/s, thấp hơn nhu cầu trung bình 115,74 tin/s. Vì vậy chưa thể nói một cấu hình hiện tại đáp ứng toàn bộ 10 triệu tin/ngày. BoW + MLP đạt khoảng 7.428 tin/s và F1 test 0,907, là baseline có bằng chứng đáp ứng cả thông lượng trung bình và độ trễ; vẫn cần kiểm thử tải đỉnh.

Có thể dùng BoW xử lý tin chắc chắn, chuyển tin lưỡng lự sang embedding hoặc bước kiểm tra bổ sung. Nếu dùng MiniLM + MLP làm tuyến chính, cần tối ưu hoặc tăng tài nguyên; tỷ lệ 115,74/30,67 ≈ 3,77 chỉ là ước lượng tải, không chứng minh 4 worker sẽ đủ. LLM chỉ nên xem xét sau khi sửa lỗi nhãn; nhánh này hiện vượt SLA, không nên mặc định chờ LLM trong đường xử lý 50 ms.

Theo giả định trong đề: 100 token input, 1 token output, giá 0,10/0,50 USD mỗi triệu token và tháng 30 ngày:

`10.000.000 × (100 × 0,10 + 1 × 0,50) / 1.000.000 × 30 = 3.150 USD/tháng`.

Đây là ước tính theo giả định cố định, khác phép ngoại suy từ chi phí số từ đo trong notebook; chưa gồm retry và hạ tầng. Nếu 1% tin vào LLM thì phần chi phí token tương ứng khoảng 31,50 USD/tháng, với cùng giả định.

## 7. AI log và điều đã học

Bảng dưới ghi lại hỗ trợ AI trong Phần 4: mục tiêu, đề xuất và kết quả kiểm chứng.

| Phần | Yêu cầu gửi AI | Hỗ trợ nhận được | Kiểm chứng / quyết định thực tế |
| --- | --- | --- | --- |
| 4.1 | Hoàn thiện hàm API theo chữ ký template, giải thích tham số và latency. | Đề xuất code gọi API, sử dụng `provider_kwargs`, trích xuất văn bản và đo thời gian. | Đã chạy thật và có latency/chi phí; thử lời chào trả văn bản, nhưng phân loại SUB báo `invalid=40`. Chưa xác định nguyên nhân. |
| 4.2–4.3 | Hỗ trợ tạo prompt zero-shot/few-shot và đọc nhãn nhiều định dạng. | Đề xuất code prompt và parser; giải thích lỗi nhận nhầm `not spam`. | Đã in prompt mẫu, chạy kiểm thử có sẵn và thí nghiệm SUB. Kiểm thử giả lập đạt nhưng đầu ra thật vẫn chưa đọc được nhãn. |
| 4.4 | Tổ chức batch, retry, lưu latency, đếm nhãn lỗi và ước tính chi phí. | Đề xuất code kết hợp các hàm template, giải thích cách ước tính token theo số từ. | Chạy SUB, cung cấp output để đối chiếu; chưa so với token usage thực tế hoặc hóa đơn API. |
| Cấu hình provider | Hỏi khả năng chuyển sang Gemini để tiết kiệm. | Hướng dẫn cấu hình provider, key và chạy lại định nghĩa hàm. | Sau đó chọn tiếp OpenAI và nạp tiền; kết quả nộp sử dụng OpenAI, không báo kết quả Gemini. |
| 4.A–4.B | Gợi ý giả thuyết và đối chiếu kết quả. | Gợi ý dự đoán, tính tỷ lệ chi phí, phân tích giới hạn của F1 bằng 0. | Dùng số liệu thật để đối chiếu: model lớn đắt 10 lần, few-shot khoảng 2,57 lần. Chưa kết luận model nào tốt hơn khi nhãn đều không hợp lệ. |

Tôi nhận ra cần phân biệt kiểm thử giả lập đạt với API thực tế trả nhãn hợp lệ; F1 bằng 0 do fallback không đủ để đánh giá năng lực LLM. Tôi cũng thấy p95 đạt chưa đồng nghĩa đủ thông lượng triển khai, và điểm test thường cao không bảo đảm độ bền trên case khó.

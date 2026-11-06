# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM


- **MSSV:** 2A202603425
- **Họ và tên:**  Nguyễn Tiến Cường
- **SEED** (in ở ô đầu notebook):9232

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.823      0.903        0.947   
2. BoW + MLP (deep learning)         0.989   0.938      0.963        1.000   
3a. Encoder (MiniLM) + kNN           0.829   0.958      0.889        0.950   
3b. Encoder (MiniLM) + LogReg        0.920   0.958      0.939        0.950   
3c. Encoder (MiniLM) + MLP head      0.938   0.938      0.938        0.976   
4a. LLM nhỏ zero-shot                1.000   0.900      0.947        0.947   
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000   
4c. LLM nhỏ few-shot                 1.000   0.900      0.947        0.947   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            3.089   2.630147e+04       0.064   
2. BoW + MLP (deep learning)         5.168   4.268940e+04       0.178   
3a. Encoder (MiniLM) + kNN          80.790   1.645800e+01     173.452   
3b. Encoder (MiniLM) + LogReg       82.759   3.618500e+01      43.610   
3c. Encoder (MiniLM) + MLP head     81.859   4.379300e+01      42.957   
4a. LLM nhỏ zero-shot                0.000   2.007000e+00     407.561   
4b. LLM lớn zero-shot                0.000   4.000000e+10       0.000   
4c. LLM nhỏ few-shot                 0.000   1.436000e+00     503.930   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.100         0.000  
2. BoW + MLP (deep learning)          0.293         0.000  
3a. Encoder (MiniLM) + kNN          346.676         0.000  
3b. Encoder (MiniLM) + LogReg        57.089         0.000  
3c. Encoder (MiniLM) + MLP head      62.489         0.000  
4a. LLM nhỏ zero-shot               974.059         0.030  
4b. LLM lớn zero-shot                 0.000         0.073  
4c. LLM nhỏ few-shot               2220.405         0.097  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.


## 3. Dự đoán và thực tế

1. Thí nghiệm 4.A — Hiệu năng của Model lớn (Gemini 3.8 Flash) so với Model nhỏ (Gemini 3.5 Flash-lite)
Tôi đã dự đoán: Model lớn hơn (Gemini 3.8 Flash) sẽ có F1-score vượt trội hoặc ít nhất là cao hơn model nhỏ nhờ khả năng hiểu sâu ngữ cảnh và lập luận tốt hơn.
Kết quả thực tế: Model nhỏ đạt F1 = 0.947 trong khi model lớn đạt F1 = 0.000.
Nguyên nhân sâu xa: Model lớn (Gemini 3.8 Flash) là dòng model suy luận (reasoning model) có cơ chế suy nghĩ ẩn trước khi đưa ra nhãn cuối cùng. Lời giải thích quá dài dòng của nó khiến hàm parse nhãn regex (parse_label) thất bại hoàn toàn và trả về None (bị coi là không hợp lệ và mặc định phân loại thành ham), dẫn đến F1-score sụp đổ hoàn toàn về 0.000.
2. Thí nghiệm 1.B — Ảnh hưởng của Learning Rate (lr = 50)
Tôi đã dự đoán: Khi learning rate cao (lr = 50), trọng số thay đổi quá mạnh khiến mô hình quá nhạy cảm với các từ khóa spam.
Kết quả thực tế: Đường loss dao động cực kỳ hỗn loạn, không thể giảm xuống và mô hình không thể hội tụ.
Nguyên nhân sâu xa: Đây là hiện tượng phân kỳ gradient (gradient divergence/overshooting). Bước cập nhật trọng số quá lớn khiến mô hình liên tục nhảy qua nhảy lại điểm tối ưu cục bộ, làm cho hàm loss bùng nổ hoặc dao động vô tận thay vì giảm dần và hội tụ.
3. Thí nghiệm Stress Test — Hiệu năng trên 50 case khó
Tôi đã dự đoán: Độ chính xác chung của tất cả các mô hình sẽ sụt giảm đồng đều xuống khoảng 60-70%.
Kết quả thực tế: Có sự phân hóa cực kỳ sâu sắc giữa các nhóm mô hình. BoW sụp đổ hoàn toàn với F1 chỉ đạt ~36-38%, trong khi nhóm LLM có prompt phòng thủ (4d) chống chịu rất tốt với F1 đạt 0.945.
Nguyên nhân sâu xa: BoW bị lỗi từ điển nghiêm trọng trước các tin nhắn né từ khóa (obfuscation) và các lỗi ngữ cảnh tinh vi. Ngược lại, LLM phòng thủ đạt điểm cực kỳ cao nhờ kỹ thuật cô lập dữ liệu thô trong thẻ <sms>. Kỹ thuật này giúp bảo vệ mô hình khỏi các cuộc tấn công prompt injection bằng cách hướng dẫn nó coi mọi nội dung bên trong thẻ chỉ là dữ liệu thuần túy, loại bỏ hoàn toàn bẫy chỉ thị giả mạo.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

Đầu tiên, em sẽ chụp toàn bộ ngữ cảnh vào cho AI rồi promt nó phân tích và giải thích trước, sau đó yêu cầu AI đưa ra hướng giải quyết, em sẽ lựa chọn phương án mình hiểu và khả thi nhất để trả lời câu hỏi được giao.

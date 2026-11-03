# Model card

## Mục đích và phạm vi

Phân loại một câu mạng xã hội tiếng Việt vào bảy nhãn UIT-VSMEC. Phục vụ đồ án NLP và demo học thuật. Không dùng kết quả để chẩn đoán tâm lý, đánh giá con người hoặc tự động áp dụng chế tài. Không hỗ trợ nhận diện lập trường hoặc cảm xúc đa nhãn.

## Mô hình và nguồn

- TF-IDF word 1–3 + character 3–5, LinearSVC C=1, balanced, fit train.
- TextCNN với Word2Vec Skip-gram 300 chiều, embedding học train và cập nhật khi phân loại.
- Fine-tune đầy đủ `vinai/phobert-base-v2`, dùng Pyvi tách từ.
- Fine-tune đầy đủ `uitnlp/visobert`, dùng văn bản chuẩn hóa và giữ emoji.

Checkpoint nằm trong `models/`. Tham khảo điều kiện của nguồn trước khi phân phối lại:

- https://huggingface.co/vinai/phobert-base-v2
- https://huggingface.co/uitnlp/visobert
- https://arxiv.org/abs/1911.09339

Bộ đồ án không thay thế giấy phép của dữ liệu hoặc mô hình nền và chưa công khai lên GitHub.

## Thực nghiệm

Train 5.548, validation 686, test 693. Seed 42. Chọn checkpoint theo validation Macro-F1. Không tuning trên test trong lần chạy này, nhưng test đã được xem trong phiên bản dự án trước. Audit có 1 câu chuẩn hóa chung giữa train và validation; không phát hiện giao train/test theo kiểm tra exact match sau chuẩn hóa.

| Mô hình | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| PhoBERT | 66,67% | 60,51% | 66,05% |
| ViSoBERT | 62,77% | 59,55% | 62,53% |
| TF-IDF + SVM | 57,58% | 56,03% | 57,53% |
| CNN + Word2Vec | 59,88% | 55,33% | 59,70% |

Nguồn: `results/model_comparison.csv`. Thông tin máy, phiên bản và mã băm dữ liệu: `results/run_manifest.json`. Kiểm tra metric từ predictions và thử cả bốn mô hình trong demo: `results/verification.json`.

## Giới hạn

Một seed, dữ liệu nhỏ và mất cân bằng, không có ngữ cảnh hội thoại. Pyvi khác bộ tách từ được tác giả PhoBERT khuyến nghị. Ngân sách fine-tuning Transformer bốn epoch. SVM dùng class weights nhưng neural không dùng nên chưa cô lập ảnh hưởng kiến trúc. Chưa đánh giá chuyển miền, chưa calibration hoặc kiểm định thống kê.

SVM trả decision score. Các mô hình neural trả softmax chưa hiệu chuẩn. Điểm cao không bảo đảm dự đoán đúng. Câu mỉa mai, slang, lỗi chính tả hoặc nhiều cảm xúc có thể gây nhầm.

## Tái lập và an toàn checkpoint

Giữ nguyên requirements, cấu hình, dữ liệu và seed để chạy lại. Sai khác phần cứng hoặc thư viện vẫn có thể ảnh hưởng kết quả. `RETRAIN=False` dùng checkpoint có sẵn, `True` sẽ cập nhật chúng. Các file pickle/joblib/PyTorch checkpoint chỉ nên nạp từ nguồn tin cậy. `SHA256SUMS.txt` giúp kiểm tra file có bị thay đổi sau khi đóng gói.

# Nhận diện cảm xúc trong văn bản mạng xã hội tiếng Việt

Đồ án môn NLP so sánh **TF-IDF + SVM, CNN + Word2Vec, PhoBERT và ViSoBERT**. Toàn bộ mã thực nghiệm nằm trong `DoAn_NhanDienCamXuc.ipynb`, phù hợp mở và chạy bằng Visual Studio Code.

## Lý do chọn đề tài

Bình luận mạng xã hội chứa thông tin về cảm xúc, nhưng thường có teencode, emoji, phủ định và cách viết không chuẩn. Đề tài giúp áp dụng nội dung tiền xử lý, biểu diễn văn bản, phân loại và đánh giá trong môn NLP vào dữ liệu tiếng Việt thực tế. Mục tiêu là hiểu hiệu quả và giới hạn của bốn phương pháp, không chỉ xây dựng giao diện dự đoán.

## Bộ dữ liệu

UIT-VSMEC gồm 6.927 câu được gán nhãn: 5.548 train, 686 validation và 693 test. Bảy nhãn là Anger, Disgust, Enjoyment, Fear, Other, Sadness, Surprise. Mỗi câu có một nhãn; Other không được tự động hiểu là neutral. Đây là emotion classification, không phải stance detection hoặc sentiment ba lớp.

Nguồn: [UIT NLP Datasets](https://nlp.uit.edu.vn/datasets), [bài giới thiệu UIT-VSMEC](https://arxiv.org/abs/1911.09339). Ba file XLSX gốc đã được đưa vào `data/raw`, tái sử dụng từ bộ dữ liệu của dự án trước. Lần chạy mới không tải một bộ dữ liệu khác. Mã băm SHA256 trong `results/run_manifest.json` giúp đối chiếu file.

Giữ nguyên split chuẩn và báo cáo audit trùng lặp ở `results/data_audit.json`. Không học vocabulary, IDF hoặc Word2Vec từ validation/test. Tài liệu chỉ báo cáo kết quả của nhóm, không so điểm với bài báo. Khi công khai lại dữ liệu, cần theo điều kiện của nguồn gốc; repository này không cấp một giấy phép mới cho UIT-VSMEC hoặc checkpoint nền.

## Cấu trúc bộ nộp

```text
DoAn_NhanDienCamXuc.ipynb     # Một notebook, đầy đủ mã và output
data/raw/                   # Ba file XLSX gốc
data/processed/             # CSV đã chuẩn hóa và tách từ
models/                     # Bốn checkpoint + Word2Vec
results/                    # Metric, dự đoán, biểu đồ và log cấu hình
demo/app.py                 # Streamlit, dùng hàm trong notebook
report/                     # Báo cáo Word, tài liệu hỏi đáp
slides/                     # PowerPoint 18 slide và kịch bản trình bày
requirements.txt            # Phiên bản thư viện của lần chạy
README.md
```

## Thiết lập môi trường và kiểm tra CPU GPU

Khuyến nghị Python 3.12, VS Code với extension **Python** và **Jupyter**. Mở đúng thư mục gốc đồ án bằng File > Open Folder, rồi mở terminal PowerShell.

```powershell
cd 'D:\7.UIT\12.NLP\DoAn_NhanDienCamXuc_TiengViet'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name nlp-emotion --display-name 'Python NLP Emotion'
```

Không cần activate môi trường nếu dùng đầy đủ đường dẫn như trên. Trong notebook chọn **Select Kernel > Python Environments > .venv** hoặc kernel **Python NLP Emotion**. Nếu không có lệnh `py`, dùng đường dẫn Python 3.12 đang cài trên máy để tạo venv.

Kiểm tra PyTorch và GPU trong terminal:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu `CUDA: False`, notebook vẫn chạy bằng CPU nhưng fine-tune Transformer có thể chậm. Để dùng GPU NVIDIA, chọn gói PyTorch tương thích driver theo [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/). Không mặc định CUDA khả dụng chỉ vì máy có GPU. Lần chạy bộ nộp này dùng RTX 3050 Ti Laptop 4 GB; thông tin thực tế được ghi trong `results/run_manifest.json`.

GPU 4 GB: giữ batch=4, accumulation=4; nếu bị CUDA out of memory, đóng ứng dụng đang chiếm VRAM, restart kernel hoặc giảm batch=2 và tăng accumulation=8. Thay đổi cấu hình phải huấn luyện lại và ghi nhận kết quả mới.

## Chạy notebook trong VS Code

1. Mở `DoAn_NhanDienCamXuc.ipynb`, chọn kernel đúng môi trường.
2. Giữ `RETRAIN = False` để dùng lại checkpoint và xem kết quả. Chạy **Run All**.
3. Muốn huấn luyện lại, đổi `RETRAIN = True`. Có thể sửa `MODELS_TO_RUN` thành `['svm']`, `['cnn_word2vec']`, `['phobert']` hoặc `['visobert']` để chỉ chạy mô hình tương ứng.
4. Huấn luyện lại cập nhật checkpoint và kết quả trong thư mục này. Sao lưu `models/` và `results/` nếu muốn giữ lần chạy cũ.
5. Khi hoàn tất, trả `RETRAIN = False` để tránh chạy lại ngoài ý muốn.

Lần fine-tune đầu cần Internet để tải `vinai/phobert-base-v2` và `uitnlp/visobert` nếu máy chưa có cache Hugging Face. Demo dùng checkpoint local đã có và không cần tải mô hình nền.

Notebook không gọi script train bên ngoài. File demo đọc các cell có tag `demo_safe`: import/cấu hình, TextCNN và hàm dự đoán. Những cell huấn luyện không được thực thi khi mở demo. Chỉ chạy notebook trong bộ đồ án tin cậy, vì notebook là mã Python có thể thực thi.

## Demo

```powershell
.\.venv\Scripts\python.exe -m streamlit run demo\app.py
```

Khi lần đầu Streamlit hỏi email, có thể để trống và nhấn Enter. Đợi terminal hiện **Local URL**, rồi mở `http://localhost:8501`. Giữ terminal chạy. Nếu cổng đã dùng, thêm `--server.port 8502` rồi mở URL tương ứng.

Chọn một mô hình hoặc bật thử cả bốn. Demo tải checkpoint, không huấn luyện. Có cell demo trong mục 12 notebook làm phương án dự phòng; sau restart kernel chỉ cần chạy cell import/cấu hình mục 1, định nghĩa TextCNN mục 6 và hàm/cell demo mục 12.

SVM trả decision score, không phải xác suất. Softmax của CNN/PhoBERT/ViSoBERT chưa được hiệu chuẩn. Không so trực tiếp những điểm này giữa mô hình.


B1: Mở terminal : 
    cd "D:\7.UIT\12.NLP\DoAn_NhanDienCamXuc_TiengViet"
B2: Run giao diện demo: 
    & "D:\2. PERSONALIZATION\8. PYTHON-REINSTALL\python.exe" -m streamlit run demo\app.py

## Thiết lập thực nghiệm

- SVM: TF-IDF word n-gram 1–3 và character n-gram 3–5, LinearSVC C=1, class_weight balanced.
- CNN: Word2Vec Skip-gram 300 chiều, chỉ học train; bộ lọc 3/4/5, 128 filter mỗi loại, dropout 0.5, tối đa 20 epoch, patience 4.
- PhoBERT: checkpoint **base-v2**, Pyvi tách từ trước tokenizer.
- ViSoBERT: văn bản chuẩn hóa, giữ emoji, không nối từ bằng Pyvi.
- Hai Transformer: tối đa 4 epoch, learning rate 2e-5, batch 4, accumulation 4, patience 2; cập nhật toàn bộ trọng số.
- Seed 42. Checkpoint tốt nhất theo validation Macro-F1; test chỉ đánh giá sau khi chọn. Mô hình demo mặc định chọn theo validation.

## Kết quả và giới hạn

| Mô hình | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| PhoBERT | 66,67% | 60,51% | 66,05% |
| ViSoBERT | 62,77% | 59,55% | 62,53% |
| TF-IDF + SVM | 57,58% | 56,03% | 57,53% |
| CNN + Word2Vec | 59,88% | 55,33% | 59,70% |

Bảng số liệu chính: `results/model_comparison.csv`. Điểm từng lớp: `results/per_label_f1.csv`. Mỗi thư mục mô hình trong results có metrics và predictions để tính lại kết quả. Báo cáo và slide được tạo từ lần chạy mới, không lấy điểm phiên bản trước.

Không cam kết điểm cao nhất cho một mô hình trước khi chạy. Một seed chưa đủ kết luận ưu thế thống kê. Dữ liệu nhỏ, lệch nhãn và có trùng lặp được audit; teencode, mỉa mai, thiếu ngữ cảnh có thể gây lỗi. Test đã được xem ở dự án trước nên không phải tập kiểm định mù mới.


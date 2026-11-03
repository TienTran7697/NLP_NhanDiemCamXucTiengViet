"""Giao diện UIT; dùng nguyên hàm dự đoán từ các cell demo_safe trong notebook."""
from pathlib import Path
import base64
import html
import json
import time
import nbformat
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / 'assets'
st.set_page_config(page_title='UIT · Nhận diện cảm xúc', page_icon='💬', layout='wide')
st.markdown((ASSETS/'style.css').read_text(encoding='utf-8'), unsafe_allow_html=True)

@st.cache_resource
def runtime():
    nb = nbformat.read(ROOT/'DoAn_NhanDienCamXuc.ipynb', as_version=4)
    namespace = {'PROJECT_ROOT': ROOT, '__name__': 'emotion_demo'}
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'demo_safe' in cell.metadata.get('tags', []):
            exec(compile(cell.source, f'notebook_cell_{index}', 'exec'), namespace)
    return namespace

names = {'svm': 'TF-IDF + SVM', 'cnn_word2vec': 'CNN + Word2Vec', 'phobert': 'PhoBERT', 'visobert': 'ViSoBERT'}
config_path = ROOT/'models/demo_config.json'
default = json.loads(config_path.read_text(encoding='utf-8'))['default_model'] if config_path.exists() else 'svm'
if default not in names:
    default = 'svm'
logo = base64.b64encode((ASSETS/'uit_logo.png').read_bytes()).decode('ascii')
st.markdown(f'''<section class="uit-hero">
<div class="uit-brand"><img src="data:image/png;base64,{logo}" alt="Trường Đại học Công nghệ Thông tin UIT"/>
<span>ĐỒ ÁN MÔN XỬ LÝ NGÔN NGỮ TỰ NHIÊN</span></div>
<h1>Nhận diện cảm xúc<br/>trong văn bản tiếng Việt</h1>
<p>Khám phá cách bốn mô hình đọc cảm xúc trong bình luận mạng xã hội.</p>
<div class="hero-meta">UIT-VSMEC <span>•</span> 7 cảm xúc <span>•</span> 4 mô hình <span>•</span> Checkpoint đã huấn luyện</div>
</section>''', unsafe_allow_html=True)

left, right = st.columns([1.1, 1], gap='large')
with left:
    st.subheader('Bình luận của bạn')
    st.caption('Nhập văn bản có dấu, emoji hoặc cách viết thường gặp trên mạng xã hội.')
    with st.form('predict'):
        text = st.text_area('Nội dung bình luận', value='Tôi rất vui vì cuối cùng cũng làm được rồi!', max_chars=2000, height=160)
        selected = st.selectbox('Mô hình', list(names), index=list(names).index(default), format_func=lambda k: names[k])
        compare = st.checkbox('So sánh cả bốn mô hình', value=False)
        submitted = st.form_submit_button('Phân tích cảm xúc', type='primary', use_container_width=True)
    st.caption(f'Mặc định: {names[default]} · Chọn theo Macro-F1 validation. Demo không huấn luyện lại.')
    with st.expander('Gợi ý câu để thử'):
        st.write('• Tôi rất vui vì cuối cùng cũng làm được rồi!')
        st.write('• Sao lại đối xử với tôi như vậy, bực mình thật!')
        st.write('• người ta có bạn bè nhìn vui thật')
        st.caption('Đây là các câu minh họa, không bảo đảm mô hình luôn dự đoán đúng.')

with right:
    st.subheader('Kết quả nhận diện')
    if not submitted:
        st.markdown('''<div class="empty-result"><div class="empty-symbol">💬</div>
        <h3>Câu này mang cảm xúc gì?</h3><p>Nhập bình luận rồi nhấn <strong>Phân tích cảm xúc</strong>.<br/>Kết quả sẽ xuất hiện tại đây.</p></div>''', unsafe_allow_html=True)
    elif not text.strip():
        st.warning('Vui lòng nhập một câu không rỗng.')
    else:
        try:
            with st.spinner('Đang tải checkpoint và dự đoán...'):
                ns = runtime()
                rows, details = [], []
                for key in names if compare else [selected]:
                    start = time.perf_counter()
                    table, kind = ns['predict_saved'](text, key)
                    rows.append({'Mô hình': names[key], 'Cảm xúc': table.iloc[0]['cảm xúc'], 'Thời gian (giây)': round(time.perf_counter()-start, 2)})
                    details.append((key, table, kind))
            colors = {'Anger': '#BA3C35', 'Disgust': '#7656A2', 'Enjoyment': '#087C61', 'Fear': '#8A641A', 'Other': '#526779', 'Sadness': '#2564AC', 'Surprise': '#AD4C12'}
            for key, table, kind in details:
                winner = table.iloc[0]
                color = colors.get(str(winner['label']), '#006EB0')
                st.markdown(f'''<div class="result-card" style="border-left-color:{color}">
                <span class="result-model">{html.escape(names[key])}</span>
                <div class="result-label" style="color:{color}">{html.escape(str(winner['cảm xúc']))}</div>
                <span class="result-sub">Nhãn dự đoán · {html.escape(str(winner['label']))}</span></div>''', unsafe_allow_html=True)
            if compare:
                st.dataframe(rows, hide_index=True, width='stretch')
            for key, table, kind in details:
                with st.expander(names[key]+' — điểm từng cảm xúc', expanded=not compare):
                    st.caption(kind)
                    st.bar_chart(table.set_index('cảm xúc')[['score']], color='#0077B6', horizontal=True)
                    st.dataframe(table[['cảm xúc', 'score']], hide_index=True, width='stretch')
            st.caption('Thời gian bao gồm tải checkpoint. SVM dùng decision score, không phải xác suất. Softmax của CNN/Transformer chưa hiệu chuẩn; không so trực tiếp điểm giữa các mô hình.')
        except FileNotFoundError:
            st.error('Chưa tìm thấy checkpoint hoặc file cần thiết. Kiểm tra thư mục models và notebook trong thư mục đồ án.')
        except Exception as exc:
            st.error(f'Không dự đoán được: {exc}')

st.markdown('''<div class="uit-footer"><strong>Bảy nhãn cảm xúc</strong><br/>
Tức giận · Ghê tởm · Vui thích · Sợ hãi · Khác · Buồn · Ngạc nhiên
<p>Mô hình có thể nhầm với câu mỉa mai hoặc thiếu ngữ cảnh. Kết quả là dự đoán, không phải kết luận chắc chắn về người viết.</p>
</div>''', unsafe_allow_html=True)

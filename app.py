import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from utils import load_models, predict_toxic

# Page config
st.set_page_config(
    page_title="Phát hiện Bình luận Toxic",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .toxic-box {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .normal-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Load models
@st.cache_resource
def load_model_cache():
    """Load models with caching"""
    try:
        vectorizer, model = load_models()
        return vectorizer, model, None
    except Exception as e:
        return None, None, str(e)

vectorizer, model, error = load_model_cache()

# Header
st.markdown('<div class="main-header">🛡️ Hệ thống Phát hiện Bình luận Toxic</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Phân tích nội dung tin nhắn và đánh giá mức độ tiêu cực</div>', unsafe_allow_html=True)

# Check if models loaded successfully
if error:
    st.error(f"❌ Lỗi khi tải model: {error}")
    st.info("Vui lòng đảm bảo các file model đã được đặt trong thư mục 'models/'")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    # Mode selection
    mode = st.radio(
        "Chế độ sử dụng:",
        ["Phân tích đơn", "Phân tích hàng loạt", "Lịch sử"]
    )
    
    st.markdown("---")
    
    # Information
    st.subheader("ℹ️ Thông tin")
    st.info("""
    **Model:** XGBoost Classifier
    
    **Loại Toxic:**
    - Ngôn từ thô tục
    - Xúc phạm, sỉ nhục
    - Kỳ thị, phân biệt đối xử
    - Đe dọa, quấy rối
    
    **Mức độ tin cậy:**
    - 🟢 Rất Cao: > 95%
    - 🔴 Cao: 85-95%
    - 🟠 Trung Bình: 70-85%
    - 🟡 Thấp: 50-70%
    - ⚪ Rất Thấp: < 50%
    """)
    
    st.markdown("---")
    st.caption("Phát triển bởi [Tên của bạn]")
    st.caption(f"Cập nhật: {datetime.now().strftime('%d/%m/%Y')}")

# Main content
if mode == "Phân tích đơn":
    st.header("📝 Phân tích Tin nhắn Đơn")
    
    # Input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "Nhập nội dung tin nhắn cần phân tích:",
            height=150,
            placeholder="Ví dụ: Chào bạn, hôm nay thế nào?"
        )
    
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔍 Phân tích", use_container_width=True)
        clear_btn = st.button("🗑️ Xóa", use_container_width=True)
    
    if clear_btn:
        st.rerun()
    
    if analyze_btn:
        if user_input:
            with st.spinner("Đang phân tích..."):
                # Predict
                result = predict_toxic(user_input, vectorizer, model)
                
                if 'error' in result and result['error']:
                    st.error(result['error'])
                else:
                    # Save to history
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'text': user_input[:100] + '...' if len(user_input) > 100 else user_input,
                        'is_toxic': result['is_toxic'],
                        'confidence': result['confidence']
                    })
                    
                    # Display results
                    st.markdown("---")
                    
                    # Main result
                    if result['is_toxic']:
                        st.markdown(f"""
                        <div class="toxic-box">
                            <h2>🔴 TOXIC - Nội dung Tiêu cực</h2>
                            <p style="font-size: 1.1rem;">Tin nhắn này có khả năng chứa nội dung tiêu cực, xúc phạm hoặc không phù hợp.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="normal-box">
                            <h2>🟢 NORMAL - Nội dung Bình thường</h2>
                            <p style="font-size: 1.1rem;">Tin nhắn này không có dấu hiệu nội dung tiêu cực.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Độ tin cậy",
                            f"{result['confidence']:.1%}",
                            delta=result['confidence_level']
                        )
                    
                    with col2:
                        st.metric(
                            "Xác suất Toxic",
                            f"{result['toxic_probability']:.1%}"
                        )
                    
                    with col3:
                        st.metric(
                            "Xác suất Normal",
                            f"{result['normal_probability']:.1%}"
                        )
                    
                    # Probability chart
                    st.subheader("📊 Phân bố xác suất")
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Normal', 'Toxic'],
                            y=[result['normal_probability'], result['toxic_probability']],
                            marker_color=['#4CAF50', '#F44336'],
                            text=[f"{result['normal_probability']:.1%}", 
                                  f"{result['toxic_probability']:.1%}"],
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title="Xác suất dự đoán",
                        yaxis_title="Xác suất",
                        yaxis=dict(range=[0, 1], tickformat='.0%'),
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Details expander
                    with st.expander("🔍 Chi tiết phân tích"):
                        st.write("**Văn bản gốc:**")
                        st.code(result['original_text'])
                        
                        st.write("**Văn bản đã xử lý:**")
                        st.code(result['cleaned_text'])
                        
                        st.write("**Thông số kỹ thuật:**")
                        st.json({
                            'Label': 'Toxic (1)' if result['is_toxic'] else 'Normal (0)',
                            'Confidence': f"{result['confidence']:.4f}",
                            'Confidence Level': result['confidence_level'],
                            'Toxic Probability': f"{result['toxic_probability']:.4f}",
                            'Normal Probability': f"{result['normal_probability']:.4f}"
                        })
        else:
            st.warning("⚠️ Vui lòng nhập nội dung tin nhắn!")

elif mode == "Phân tích hàng loạt":
    st.header("📊 Phân tích Hàng loạt")
    
    st.info("Upload file CSV với cột 'Content' hoặc 'text' chứa nội dung cần phân tích")
    
    uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.write("**Preview dữ liệu:**")
            st.dataframe(df.head())
            
            # Detect text column
            text_col = None
            for col in ['Content', 'content', 'text', 'Text', 'cleaned_content']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                st.error("❌ Không tìm thấy cột chứa nội dung. Vui lòng đảm bảo có cột 'Content' hoặc 'text'")
            else:
                st.success(f"✓ Đã phát hiện cột: '{text_col}'")
                
                if st.button("🚀 Bắt đầu phân tích", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    total = len(df)
                    
                    for idx, row in df.iterrows():
                        text = row[text_col]
                        result = predict_toxic(text, vectorizer, model)
                        
                        results.append({
                            'text': text[:100] + '...' if len(str(text)) > 100 else text,
                            'is_toxic': result['is_toxic'],
                            'label': result['label'],
                            'confidence': result['confidence'],
                            'toxic_probability': result['toxic_probability'],
                            'normal_probability': result['normal_probability']
                        })
                        
                        progress = (idx + 1) / total
                        progress_bar.progress(progress)
                        status_text.text(f"Đã xử lý: {idx + 1}/{total}")
                    
                    results_df = pd.DataFrame(results)
                    
                    # Statistics
                    st.markdown("---")
                    st.subheader("📈 Thống kê tổng quan")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Tổng số", len(results_df))
                    
                    with col2:
                        toxic_count = results_df['is_toxic'].sum()
                        st.metric("Toxic", f"{toxic_count} ({toxic_count/len(results_df)*100:.1f}%)")
                    
                    with col3:
                        normal_count = len(results_df) - toxic_count
                        st.metric("Normal", f"{normal_count} ({normal_count/len(results_df)*100:.1f}%)")
                    
                    with col4:
                        avg_conf = results_df['confidence'].mean()
                        st.metric("Độ tin cậy TB", f"{avg_conf:.1%}")
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=['Normal', 'Toxic'],
                            values=[normal_count, toxic_count],
                            marker_colors=['#4CAF50', '#F44336']
                        )])
                        fig.update_layout(title="Phân bố Toxic/Normal")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Confidence histogram
                        fig = go.Figure(data=[go.Histogram(
                            x=results_df['confidence'],
                            nbinsx=20,
                            marker_color='#1E88E5'
                        )])
                        fig.update_layout(
                            title="Phân bố độ tin cậy",
                            xaxis_title="Độ tin cậy",
                            yaxis_title="Số lượng"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("📋 Kết quả chi tiết")
                    
                    # Filter
                    filter_option = st.selectbox(
                        "Lọc kết quả:",
                        ["Tất cả", "Chỉ Toxic", "Chỉ Normal"]
                    )
                    
                    if filter_option == "Chỉ Toxic":
                        display_df = results_df[results_df['is_toxic'] == True]
                    elif filter_option == "Chỉ Normal":
                        display_df = results_df[results_df['is_toxic'] == False]
                    else:
                        display_df = results_df
                    
                    st.dataframe(
                        display_df.style.applymap(
                            lambda x: 'background-color: #FFEBEE' if x == True else 'background-color: #E8F5E9',
                            subset=['is_toxic']
                        ),
                        use_container_width=True
                    )
                    
                    # Download
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Tải xuống kết quả (CSV)",
                        csv,
                        "toxic_analysis_results.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý file: {e}")

else:  # History mode
    st.header("📜 Lịch sử Phân tích")
    
    if len(st.session_state.history) == 0:
        st.info("Chưa có lịch sử phân tích. Hãy thử phân tích một số tin nhắn!")
    else:
        # Statistics
        toxic_count = sum(1 for item in st.session_state.history if item['is_toxic'])
        normal_count = len(st.session_state.history) - toxic_count
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tổng số phân tích", len(st.session_state.history))
        
        with col2:
            st.metric("Toxic", f"{toxic_count} ({toxic_count/len(st.session_state.history)*100:.0f}%)")
        
        with col3:
            st.metric("Normal", f"{normal_count} ({normal_count/len(st.session_state.history)*100:.0f}%)")
        
        st.markdown("---")
        
        # History table
        history_df = pd.DataFrame(st.session_state.history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        st.dataframe(
            history_df.style.applymap(
                lambda x: 'background-color: #FFEBEE' if x == True else 'background-color: #E8F5E9',
                subset=['is_toxic']
            ),
            use_container_width=True
        )
        
        # Clear history
        if st.button("🗑️ Xóa lịch sử", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🛡️ Hệ thống Phát hiện Bình luận Toxic</p>
    <p>Phát triển với ❤️ sử dụng Streamlit & XGBoost</p>
</div>
""", unsafe_allow_html=True)
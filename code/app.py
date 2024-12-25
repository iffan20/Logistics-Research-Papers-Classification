import streamlit as st
import joblib
from docx import Document
import fitz  # For PDF files
import io
import zipfile
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_words, thai_stopwords
from pythainlp.util import dict_trie
from sklearn.preprocessing import LabelEncoder

# Load the model and its components
objects = joblib.load('../model/model.pkl')
tfidf = objects['tfidf']
model = objects['model']
label_encoder = objects['label_encoder']

# Sidebar for file upload
st.sidebar.header("File Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload files", type=["docx", "pdf"], accept_multiple_files=True
)

# Main Page Title
st.title("Logistics Research Paper Classification")
st.subheader("Do you want to know what category your research paper belongs to?")

with st.expander("How does the file upload and classification work?"):
    st.markdown(
        """
        This application allows you to upload multiple DOCX or PDF files of logistics research papers.
        
        **Steps:**
        1. Upload your files using the file uploader on the left sidebar.
        2. The application will extract text from your documents (either DOCX or PDF).
        3. The text is then preprocessed to remove unnecessary words and tokens.
        4. Based on the content, the application will predict the category that best fits your research paper using a trained machine learning model.
        5. You will see the class prediction displayed, along with the confidence level (probability) for each paper.
        6. The system will also provide a categorized ZIP file that you can download containing the processed papers.

        **Notes:**
        - The confidence score is displayed, and files with low-confidence predictions (≤ 40%) will be flagged for rechecking.
        - You can download the categorized files as a ZIP after the classification is completed.
        """
    )
    
# Function to extract text from DOCX files
def extract_text_from_docx(uploaded_file):
    try:
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

# Function to extract text from PDF files
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_text = []
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                pdf_text.append(page.get_text())
        return "\n".join(pdf_text).replace(' า', 'ำ').replace(' ำ', 'ำ')
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to preprocess text and predict class
def preprocess_and_predict(text):
    added_words = [
        'การนำเข้า', 'ฐานนิยม', 'คราฟท์', 'แนวทาง', 'ผู้ส่งมอบ', 'โซ่อุปทาน',
        'ปัจจัยรอง', 'การส่งมอบ', 'รถขนส่ง', 'นำไปใช้งาน', 'อย่างถูกต้อง', 'การขับรถ',
        'ที่เกี่ยวข้อง', 'ในการปฏิบัติงาน', 'พนักงานขับรถ', 'สิ่งสำคัญ', 'ขั้นตอน', 'ที่ชัดเจน',
        'การไหล', 'ยอดขาย', 'การจัดทำ', 'คราฟท์เบียร์', 'ฝึกสหกิจ', 'อย่างก้าวกระโดด',
        'การจัดซื้อจัดหา', 'กระบวนการ', 'แบบประเมิน', 'เก็บข้อมูล', 'อย่างชัดเจน',
        'การดำเนินการ', 'การส่งเสริม', 'ถังดับเพลิง', 'แนวทาง'
    ]
    
    custom_words = set(thai_words()).union(added_words)
    custom_trie = dict_trie(custom_words)
    
    tokens = word_tokenize(text, custom_dict=custom_trie, engine="newmm")
    text = " ".join([token for token in tokens if token not in thai_stopwords()])
    text_tfidf = tfidf.transform([text])
    
    prediction = model.predict(text_tfidf.toarray(), verbose=0)
    all_classes = label_encoder.inverse_transform(range(prediction.shape[1]))
    sorted_class_probs = sorted(
        zip(all_classes, prediction[0]), key=lambda x: x[1], reverse=True
    )
    
    filtered_class_probs = [cat for cat in sorted_class_probs if cat[1] * 100 > 40]
    if not filtered_class_probs:
        filtered_class_probs = [sorted_class_probs[0]]

    return filtered_class_probs

# Process uploaded files and display predictions
if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
    st.subheader('Results')
    # Create an in-memory ZIP file
    zip_buffer = io.BytesIO()
    files_to_recheck = []

    # Initialize progress bar and count display
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    progress_text = st.empty()  # Placeholder for progress text

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        with st.expander("Details of Prediction Results"):
            for idx, uploaded_file in enumerate(uploaded_files):
                st.markdown(
                    f"""
                    <div style="padding:10px; border-radius:5px; color:white; text-align:left; font-size:16px;">
                        Processing file: {uploaded_file.name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Extract text based on file type
                if uploaded_file.name.endswith(".docx"):
                    text = extract_text_from_docx(uploaded_file)
                elif uploaded_file.name.endswith(".pdf"):
                    text = extract_text_from_pdf(uploaded_file)
                else:
                    st.warning("Unsupported file format")
                    continue

                # Predict category
                filtered_class_probs = preprocess_and_predict(text)
                for cat in filtered_class_probs:
                    # Save file into the correct folder within the ZIP
                    folder_path = f"{cat[0]}/"
                    file_name = os.path.basename(uploaded_file.name)
                    zipf.writestr(f"{folder_path}{file_name}", uploaded_file.getvalue())

                    # Display prediction
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(
                            f"""
                            <div style="padding:10px; border-radius:5px; color:white; text-align:left; font-weight:bold; font-size:14px;">
                                Category: {cat[0]}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col2:
                        probability = cat[1] * 100
                        color = "#55a630" if probability > 60 else "#ff9f1c" if probability > 40 else "#f94144"
                        st.markdown(
                            f"""
                            <div style="padding:5px; border-radius:5px; background-color:{color}; color:white; text-align:center; font-size:14px;">
                                Probability: {probability:.2f}%
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Add file to recheck list if any probability is ≤40%
                if any(cat[1] * 100 <= 40 for cat in filtered_class_probs):
                    files_to_recheck.append(uploaded_file.name)

                st.markdown('---')

                # Update progress bar and progress text
                progress_bar.progress((idx + 1) / total_files)
                #progress_text.text(f"{idx + 1}/{total_files}")


    
    # Show a finish message
    st.success(f"All {total_files} files have been processed successfully!")
    
    # Display recheck message if any files require it
    if files_to_recheck:
        st.markdown(
            f"""
            <div style="padding:10px; border-radius:5px; background-color:#f9c6c6; color:black; text-align:left; font-size:16px; font-weight:bold;">
            Files with low-confidence predictions (≤40%) and need rechecking:
                <ul>
                    {''.join([f"<li>{file}</li>" for file in files_to_recheck])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    

    # Allow users to download the categorized ZIP file
    zip_buffer.seek(0)
    st.text('')
    st.download_button(
        label="Download Categorized Files as ZIP",
        data=zip_buffer,
        file_name="categorized_files.zip",
        mime="application/zip",
    )
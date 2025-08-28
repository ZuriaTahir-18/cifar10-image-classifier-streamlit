import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# ----------------- Load model -----------------
model = load_model(
    r"C:\Users\GIFT\Downloads\Image classification\Image classification\model_cnn_cifar10.h5",
    compile=False
)

# CIFAR-10 classes
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ----------------- Page + State -----------------
st.set_page_config(page_title="CIFAR-10 Image Classification", layout="wide")

for k in ("classified", "pred_idx", "confidence"):
    if k not in st.session_state:
        st.session_state[k] = None

# ----------------- CSS -----------------
st.markdown("""
    <style>
    div[data-testid="stFileUploader"] button,
    div[data-testid="stFileUploader"] div[role="button"] {
      background-color: #16a34a !important; 
      color: #ffffff !important;
      border-radius: 8px !important;
      padding: 6px 12px !important;
      border: none !important;
    }

    div[data-testid="stButton"] > button {
      background: linear-gradient(90deg, #2563eb, #1e40af) !important;
      color: #ffffff !important;
      border-radius: 8px !important;
      padding: 8px 18px !important;
      border: none !important;
      font-weight: 600 !important;
    }

    .stApp {
      background: linear-gradient(135deg, #2575fc, #ECFDF5);
    }

    header, header > div, [data-testid="stToolbar"], [data-testid="stDecoration"],
    #MainMenu {
        display: none !important;
    }

    .top-bar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 78px;
        display:flex;
        align-items:center;
        justify-content:center;
        background: linear-gradient(90deg,#6a11cb,#2575fc);
        color:white;
        z-index: 9999;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .top-bar h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
    }
    main .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    .about-box {
        background: white;
        padding: 28px;
        border-radius: 12px;
        box-shadow: 0px 6px 14px rgba(0,0,0,0.08);
        width: 100%;
    }

    .result-box {
        font-size: 18px; font-weight: 700; text-align: center;
        background: white; padding: 14px; border-radius: 12px;
        box-shadow: 0px 6px 12px rgba(0,0,0,0.10);
        margin-top: 10px;
    }

    .stImage > img {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Top bar -----------------
st.markdown("<div class='top-bar'><h1>Image Classification App</h1></div>", unsafe_allow_html=True)

# ----------------- Main wrapper -----------------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Two columns
col1, col2 = st.columns([1.5, 1], gap="large")

# ---------- Left: About + Results ----------
with col1:
    st.markdown(
        "<div class='about-box'>"
        "<h3>About CIFAR-10 Dataset</h3>"
        "<p>This app uses a Convolutional Neural Network (CNN) trained on the <b>CIFAR-10 dataset</b>. "
        "The dataset contains <b>60,000 color images</b> (32×32 pixels, RGB) across <b>10 classes</b>: "
        "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.</p>"
        "<p>CIFAR-10 is one of the most widely used benchmark datasets in computer vision for testing small image classification models.</p>"
        "</div>",
        unsafe_allow_html=True
    )

    if st.session_state.classified and st.session_state.pred_idx is not None:
        pred_name = class_names[st.session_state.pred_idx]
        conf = st.session_state.confidence

        if conf < 70:
            st.error("⚠️ The uploaded image likely does not belong to CIFAR-10 classes.")
        else:
            st.markdown(
                f"<div class='result-box'> The Uploaded Image is classified as: "
                f"<span style='color:#16a34a'>{pred_name}</span><br>"
                f"🎯 Confidence Score: {conf:.2f}%</div>",
                unsafe_allow_html=True
            )

# ---------- Right: Upload + Classify ----------
with col2:
    uploaded_file = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"], key="uploader")

    if uploaded_file is not None:
        # ✅ Reset results when a new file is uploaded
        if uploaded_file.name != st.session_state.get("last_filename", None):
            st.session_state.last_filename = uploaded_file.name
            st.session_state.classified = False
            st.session_state.pred_idx = None
            st.session_state.confidence = None

        img_placeholder = st.empty()
        original_image = Image.open(uploaded_file).convert("RGB")
        img_placeholder.image(original_image, width=420, caption="Uploaded Image")

        if not st.session_state.classified:
            if st.button("Classify Image", key="classify", help="Run model on uploaded image"):
                try:
                    image_resized = original_image.resize((32, 32))
                    img_array = np.array(image_resized).astype("float32") / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    predictions = model.predict(img_array)
                    class_index = int(np.argmax(predictions[0]))
                    confidence = float(np.max(predictions[0]) * 100.0)

                    st.session_state.pred_idx = class_index
                    st.session_state.confidence = confidence
                    st.session_state.classified = True
                except Exception as e:
                    st.error(f"❌ Error during classification: {e}")


st.markdown("</div>", unsafe_allow_html=True)

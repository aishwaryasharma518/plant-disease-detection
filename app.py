import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f4fff4;
}
.title-box {
    background: linear-gradient(135deg, #1b5e20, #43a047);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
}
.title-box h1 {
    font-size: 42px;
    margin-bottom: 5px;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.12);
    margin-bottom: 20px;
}
.result-card {
    background: linear-gradient(135deg, #e8f5e9, #ffffff);
    padding: 25px;
    border-radius: 18px;
    border-left: 8px solid #2e7d32;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.12);
}
.note-box {
    background-color: #fffde7;
    padding: 15px;
    border-radius: 12px;
    border-left: 6px solid #fbc02d;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🌿 Plant Disease Detector")
st.sidebar.info("""
Developed using:

• TensorFlow  
• CNN  
• Streamlit  
• PlantVillage Dataset
""")
st.sidebar.success("Model Accuracy: 90.5%")
st.sidebar.success("Classes: 10")
st.sidebar.markdown("---")
st.sidebar.write("Developed by **Aishwarya Sharma**")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/plant_disease_model.h5")

model = load_model()

# Load class names
with open("model/class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Display names
display_names = {
    "Tomato___healthy": "Healthy Tomato Leaf",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus"
}

# Disease information
disease_info = {
    "Tomato___healthy":
        "The tomato leaf appears healthy and disease free.",

    "Tomato___Early_blight":
        "Early Blight is a fungal disease that causes brown spots on older leaves. Remove infected leaves and use suitable fungicides.",

    "Tomato___Late_blight":
        "Late Blight spreads quickly in humid conditions and causes dark patches on leaves. Remove infected parts and apply preventive fungicides.",

    "Tomato___Leaf_Mold":
        "Leaf Mold usually appears in humid environments. Improve air circulation and avoid overhead watering.",

    "Tomato___Bacterial_spot":
        "Bacterial Spot causes small dark lesions on leaves. Use disease-free seeds and avoid wetting the leaves.",

    "Tomato___Septoria_leaf_spot":
        "Septoria Leaf Spot causes circular spots with dark borders. Remove infected leaves and keep the plant area clean.",

    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Spider mites cause tiny yellow or white spots on leaves. Use water spray or suitable miticide treatment.",

    "Tomato___Target_Spot":
        "Target Spot causes circular brown lesions. Remove infected leaves and improve plant spacing.",

    "Tomato___Tomato_mosaic_virus":
        "Tomato Mosaic Virus causes mottled leaves and poor growth. Remove infected plants and disinfect tools.",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Yellow Leaf Curl Virus causes curled yellow leaves and stunted growth. Control whiteflies and remove infected plants."
}

# Header
st.markdown("""
<div class="title-box">
    <h1>🌿 Plant Disease Detection</h1>
    <p>Detect tomato leaf diseases using Deep Learning and CNN</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 🤖 AI Powered Plant Health Monitoring
Upload or capture a tomato leaf image and instantly identify diseases using Deep Learning.
""")

# Image input
left, right = st.columns([1, 1])
image = None

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Select Leaf Image")

    option = st.radio(
        "Choose input method:",
        ["📁 Upload Leaf Image", "📷 Use Camera"]
    )

    if option == "📁 Upload Leaf Image":
        uploaded_file = st.file_uploader(
            "Upload a tomato leaf image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

    else:
        camera_image = st.camera_input("Take a photo of tomato leaf")

        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🖼️ Image Preview")

    if image is not None:
        st.image(image, caption="Selected Leaf Image", use_container_width=True)
    else:
        st.info("Upload or capture a tomato leaf image to preview it here.")

    st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if image is not None:
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]
    display_class = display_names.get(predicted_class, predicted_class)

    confidence = float(np.max(prediction) * 100)
    progress_value = min(max(confidence / 100, 0.0), 1.0)

    st.markdown("---")
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Detected Disease", display_class)

    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")

    st.write("Prediction Confidence")
    st.progress(progress_value)

    st.info(
        disease_info.get(
            predicted_class,
            "Information not available for this disease."
        )
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="note-box">
        <b>📌 Note:</b> This result is generated by a CNN model trained on tomato leaf disease images.
        For real agricultural use, expert verification is recommended.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="note-box">
        🌱 Please upload a clear tomato leaf image or use the camera to start prediction.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Developed by Aishwarya Sharma | Plant Disease Detection Using Deep Learning")
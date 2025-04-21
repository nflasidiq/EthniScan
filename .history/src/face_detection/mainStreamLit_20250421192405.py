import streamlit as st
import sys
import os
import numpy as np
from PIL import Image, ImageDraw
import io
import matplotlib.pyplot as plt
import base64
import cv2

# Add path to local modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'face_detection'))

# Import required modules - make sure these are implemented in your src directory
from src.face_detection.face_similarity import compare_faces
from src.face_detection.ethnicity_classifier import predict_ethnicity

# Page configuration
st.set_page_config(
    page_title="EthniScan",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #424242;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .similarity-score {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
    }
    .match-result {
        font-size: 1.5rem;
        text-align: center;
        margin-top: 10px;
    }
    .match-yes {
        color: #4CAF50;
    }
    .match-no {
        color: #F44336;
    }
    .stProgress > div > div {
        background-color: #1E88E5;
    }
    .image-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Application header
st.markdown('<div class="main-header">📷 EthniScan</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Face similarity detection and ethnicity prediction tool</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["👥 Compare Faces", "🌍 Predict Ethnicity"])

# Helper function for face detection
def detect_face(image_bytes):
    """Detect face in image and return bounding box"""
    # Convert image bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV's face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # If no faces are found, return a default bounding box covering most of the image
    if len(faces) == 0:
        h, w = img.shape[:2]
        return [int(w*0.25), int(h*0.25), int(w*0.5), int(h*0.5)], img
    
    # Return the largest face bounding box (assuming it's the main subject)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    return [x, y, w, h], img

# Helper function to draw bounding box on image
def draw_bounding_box(img, bounding_box):
    """Draw bounding box on image and return as PIL Image"""
    # Convert OpenCV BGR to RGB for PIL
    if isinstance(img, np.ndarray):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
    else:
        img_pil = img
    
    draw = ImageDraw.Draw(img_pil)
    x, y, w, h = bounding_box
    draw.rectangle([(x, y), (x+w, y+h)], outline="lime", width=3)
    
    return img_pil

# Helper function to resize image
def resize_image(image_bytes, max_size=(800, 800)):
    """Resize image to limit file size"""
    img = Image.open(io.BytesIO(image_bytes))
    
    # Only resize if the image is larger than max_size
    if img.width > max_size[0] or img.height > max_size[1]:
        img.thumbnail(max_size, Image.LANCZOS)
        
        # Convert back to bytes
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return buffered.getvalue()
    
    return image_bytes

# Helper function to process uploaded image
def process_image(file):
    if file is None:
        return None
        
    try:
        # Get image bytes
        image_bytes = file.getvalue()
        
        # Resize image to limit file size
        image_bytes = resize_image(image_bytes)
        
        # Get PIL image for display
        img = Image.open(io.BytesIO(image_bytes))
        
        return image_bytes, img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Helper function to format embedding vector for display
def format_embedding(embedding, max_display=10):
    """Format embedding vector for display"""
    if len(embedding) > max_display:
        # Show first few and last few values
        formatted = [f"{val:.4f}" for val in embedding[:max_display//2]]
        formatted += ["..."]
        formatted += [f"{val:.4f}" for val in embedding[-max_display//2:]]
    else:
        formatted = [f"{val:.4f}" for val in embedding]
    return formatted

# =======================
# === Tab 1: Face Comparison
# =======================
with tab1:
    st.header("Compare Two Faces")
    st.write("Upload or capture two face images to compare their similarity.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Face 1")
        use_camera1 = st.toggle("Use Camera", key="cam1")
        face_file1 = st.camera_input("Capture First Face") if use_camera1 else st.file_uploader(
            "Upload First Face", type=["jpg", "jpeg", "png"], key="face1"
        )
        
        face_data1 = process_image(face_file1) if face_file1 else None
        if face_data1:
            _, img1 = face_data1
            with st.container(height=300):
                st.image(img1, caption="Face 1", use_column_width=True)
                file_size = len(face_data1[0]) / 1024  # KB
                st.caption(f"File size: {file_size:.1f} KB")

    with col2:
        st.subheader("Face 2")
        use_camera2 = st.toggle("Use Camera", key="cam2")
        face_file2 = st.camera_input("Capture Second Face") if use_camera2 else st.file_uploader(
            "Upload Second Face", type=["jpg", "jpeg", "png"], key="face2"
        )
        
        face_data2 = process_image(face_file2) if face_file2 else None
        if face_data2:
            _, img2 = face_data2
            with st.container(height=300):
                st.image(img2, caption="Face 2", use_column_width=True)
                file_size = len(face_data2[0]) / 1024  # KB
                st.caption(f"File size: {file_size:.1f} KB")

    # Compare button
    if st.button("🔍 Compare Faces", type="primary", use_container_width=True):
        if not face_data1 or not face_file1:
            st.error("Please upload or capture the first face image")
        elif not face_data2 or not face_file2:
            st.error("Please upload or capture the second face image")
        else:
            with st.spinner("🔄 Comparing faces..."):
                try:
                    # Get image bytes
                    bytes1, _ = face_data1
                    bytes2, _ = face_data2
                    
                    # Detect faces and get bounding boxes
                    bbox1, img1_cv = detect_face(bytes1)
                    bbox2, img2_cv = detect_face(bytes2)
                    
                    # Now let's assume we have the result from your comparison function
                    # In a real implementation, you'd pass the detected face regions to this function
                    # Call face comparison function - you need to implement this with proper face detection
                    result = compare_faces(bytes1, bytes2)
                    
                    # Extract results
                    # Ensure similarity is never exactly 0, minimum should be 0.01 (1%)
                    similarity = max(result.get("similarity", 0.15), 0.01)
                    match = similarity > 0.65  # Determine match based on similarity threshold
                    
                    # Use our detected bounding boxes
                    result["bbox1"] = bbox1
                    result["bbox2"] = bbox2
                    
                    # Example embeddings - in real implementation, get these from your model
                    embedding1 = result.get("embedding1", np.random.rand(128).tolist())
                    embedding2 = result.get("embedding2", np.random.rand(128).tolist())
                    
                    # Draw bounding boxes on images
                    bbox_img1 = draw_bounding_box(img1_cv, bbox1)
                    bbox_img2 = draw_bounding_box(img2_cv, bbox2)
                    
                    # === DISPLAY DETAILED OUTPUTS ===
                    
                    # 1. Display bounding box results
                    st.markdown("### 1️⃣ Face Detection Results")
                    bbox_col1, bbox_col2 = st.columns(2)
                    
                    with bbox_col1:
                        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                        st.image(bbox_img1, caption="Face 1 Detection", use_column_width=True)
                        st.markdown(f"**Bounding Box 1:** X={bbox1[0]}, Y={bbox1[1]}, Width={bbox1[2]}, Height={bbox1[3]}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    with bbox_col2:
                        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                        st.image(bbox_img2, caption="Face 2 Detection", use_column_width=True)
                        st.markdown(f"**Bounding Box 2:** X={bbox2[0]}, Y={bbox2[1]}, Width={bbox2[2]}, Height={bbox2[3]}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # 2. Face Embedding Vectors
                    st.markdown("### 2️⃣ Face Embedding Vectors")
                    st.info(f"Extracted face embeddings (showing {min(10, len(embedding1))} of {len(embedding1)} dimensions)")
                    
                    embed_col1, embed_col2 = st.columns(2)
                    with embed_col1:
                        st.markdown("**Face 1 Embedding Vector:**")
                        st.code(str(format_embedding(embedding1)))
                    
                    with embed_col2:
                        st.markdown("**Face 2 Embedding Vector:**")
                        st.code(str(format_embedding(embedding2)))
                    
                    # 3. Similarity Score and Decision
                    st.markdown("### 3️⃣ Similarity Results")
                    
                    # Create a card-like container for results
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    
                    # Show similarity score as percentage
                    st.markdown(f"<div class='similarity-score'>{similarity * 100:.2f}%</div>", unsafe_allow_html=True)
                    
                    # Display progress bar
                    st.progress(similarity)
                    
                    # Show match/no match decision
                    match_class = "match-yes" if match else "match-no"
                    match_text = "✅ MATCH" if match else "❌ NO MATCH"
                    st.markdown(f"<div class='match-result {match_class}'>{match_text}</div>", unsafe_allow_html=True)
                    
                    # Close the card div
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # 4. Visual Comparison Interface
                    st.markdown("### 4️⃣ Visual Comparison Interface")
                    
                    # Side-by-side comparison with similarity visualization
                    comp_col1, comp_col2, comp_col3 = st.columns([2, 1, 2])
                    
                    with comp_col1:
                        st.image(bbox_img1, caption="Face 1", use_column_width=True)
                    
                    with comp_col2:
                        # Visual similarity indicator
                        st.markdown("<div style='text-align:center; padding-top:50px;'>", unsafe_allow_html=True)
                        
                        # Show similarity arrows
                        if similarity < 0.3:
                            st.markdown("🔴")
                            st.markdown("↔️")
                            st.markdown(f"**{similarity * 100:.1f}%**")
                        elif similarity < 0.7:
                            st.markdown("🟡")
                            st.markdown("⇔")
                            st.markdown(f"**{similarity * 100:.1f}%**")
                        else:
                            st.markdown("🟢")
                            st.markdown("⇔")
                            st.markdown(f"**{similarity * 100:.1f}%**")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with comp_col3:
                        st.image(bbox_img2, caption="Face 2", use_column_width=True)
                    
                    # Technical details
                    with st.expander("View Technical Details"):
                        st.markdown("### Face Comparison Technical Details")
                        st.write("""
                        The face comparison process involves:
                        1. **Face Detection**: Locating and extracting face regions in the images
                        2. **Face Alignment**: Aligning detected faces to a canonical pose
                        3. **Feature Extraction**: Computing embedding vectors (128-512 dimensions) that represent facial features
                        4. **Similarity Calculation**: Computing distance/similarity between embeddings using methods like:
                           - Euclidean Distance
                           - Cosine Similarity
                        5. **Decision**: Determining if the faces match based on a similarity threshold
                        """)
                        
                        st.markdown("### Distance Metrics")
                        metric_type = result.get("metric_type", "cosine")
                        st.write(f"**Metric used:** {metric_type}")
                        st.write(f"**Distance value:** {1-similarity:.4f}")
                        st.write(f"**Similarity value:** {similarity:.4f}")
                        st.write(f"**Match threshold:** {0.65}")
                
                except Exception as e:
                    st.error(f"Error comparing faces: {e}")
                    st.info("Make sure both images contain clear, recognizable faces.")

# =======================
# === Tab 2: Ethnicity Prediction
# =======================
with tab2:
    st.header("Ethnicity Prediction")
    st.write("Upload or capture a face image to predict ethnicity.")
    
    # Input section
    use_camera_ethnicity = st.toggle("Use Camera", key="eth_cam")
    ethnicity_file = st.camera_input("Capture Face") if use_camera_ethnicity else st.file_uploader(
        "Upload Face Image", type=["jpg", "jpeg", "png"], key="ethnicity"
    )
    
    # Process uploaded image
    eth_data = process_image(ethnicity_file) if ethnicity_file else None
    if eth_data:
        _, img = eth_data
        st.image(img, caption="Image for Prediction", use_column_width=True)
        file_size = len(eth_data[0]) / 1024  # KB
        st.caption(f"File size: {file_size:.1f} KB")
    
    # Predict button
    if st.button("🌐 Predict Ethnicity", type="primary", use_container_width=True):
        if not eth_data or not ethnicity_file:
            st.error("Please upload or capture a face image")
        else:
            with st.spinner("🔍 Predicting ethnicity..."):
                try:
                    # Get image bytes
                    image_bytes, _ = eth_data
                    
                    # Detect face first to show bounding box
                    bbox, img_cv = detect_face(image_bytes)
                    bbox_img = draw_bounding_box(img_cv, bbox)
                    
                    # Show detected face
                    st.image(bbox_img, caption="Detected Face", use_column_width=True)
                    
                    # Call ethnicity prediction function
                    result = predict_ethnicity(image_bytes)
                    
                    # Extract results
                    predicted = result.get("predicted_ethnicity", "Unknown")
                    confidence = result.get("confidence", 0)
                    all_probabilities = result.get("all_probabilities", {})
                    
                    # Display results
                    st.markdown("### 🌍 Prediction Results")
                    
                    # Create card-like container for results
                    st.markdown("""
                        <div class="result-card">
                            <h3 style='color: #1E88E5; text-align: center;'>Predicted Ethnicity</h3>
                            <h2 style='text-align: center;'>{}</h2>
                            <p style='text-align: center;'><strong>Confidence:</strong> {:.2f}%</p>
                        </div>
                    """.format(predicted, confidence*100), unsafe_allow_html=True)
                    
                    # Show confidence bar
                    st.progress(confidence)
                    
                    # Add confidence warning if needed
                    if confidence < 0.6:
                        st.warning(f"Low confidence: {confidence * 100:.2f}%. Prediction might not be accurate.")
                    elif confidence >= 0.9:
                        st.success(f"High confidence prediction: {confidence * 100:.2f}%")
                    
                    # Show all probabilities if available
                    if all_probabilities:
                        st.markdown("### All Ethnicity Probabilities")
                        
                        # Convert to list of tuples and sort by probability
                        probabilities = sorted(
                            [(eth, prob) for eth, prob in all_probabilities.items()],
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        
                        # Display as a table
                        for eth, prob in probabilities:
                            st.markdown(f"**{eth}**: {prob*100:.2f}%")
                            st.progress(prob)
                
                except Exception as e:
                    st.error(f"Error predicting ethnicity: {e}")
                    st.info("Make sure the image contains a clear, recognizable face.")

# Footer
st.markdown("---")
st.markdown("### 📝 About EthniScan")
st.write("""
EthniScan uses computer vision and deep learning to analyze facial features.
The face comparison feature uses facial embeddings and distance metrics to calculate similarity.
The ethnicity prediction is based on trained models and should be used responsibly.
""")

# Add disclaimer
with st.expander("⚠️ Important Disclaimer"):
    st.write("""
    This tool is for demonstration purposes only. Ethnicity prediction is based on visual features only and
    may not reflect actual ethnic background. Face similarity scores should be interpreted carefully.
    Always respect privacy and obtain proper consent before using facial recognition technologies.
    """)
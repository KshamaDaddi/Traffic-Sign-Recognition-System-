
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from data_utils import GTSRB_CLASS_NAMES
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Traffic Sign Recognition System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_traffic_sign_model(model_path):
    """Load the trained model (cached)"""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(32, 32)):
    """Preprocess image for model prediction"""
    # Resize image
    image_resized = cv2.resize(np.array(image), target_size)

    # Normalize
    image_normalized = image_resized.astype('float32') / 255.0

    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)

    return image_batch

def predict_traffic_sign(model, image):
    """Predict traffic sign class"""
    # Preprocess image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image, verbose=0)

    # Get class and confidence
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    # Get class name
    class_name = GTSRB_CLASS_NAMES.get(predicted_class, f"Unknown (Class {predicted_class})")

    return {
        'class_id': predicted_class,
        'class_name': class_name,
        'confidence': confidence,
        'all_probabilities': prediction[0]
    }

def create_prediction_chart(probabilities, top_n=10):
    """Create a bar chart of top predictions"""
    # Get top N predictions
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_probs = probabilities[top_indices]
    top_classes = [GTSRB_CLASS_NAMES[i] for i in top_indices]

    # Create bar chart
    fig = px.bar(
        x=top_probs * 100,
        y=top_classes,
        orientation='h',
        title=f'Top {top_n} Predictions',
        labels={'x': 'Confidence (%)', 'y': 'Traffic Sign Class'}
    )

    fig.update_layout(height=400)

    return fig

def main():
    # Title and header
    st.title("🚦 Traffic Sign Recognition System")
    st.markdown("Upload an image of a traffic sign to get real-time predictions using deep learning!")

    # Sidebar
    st.sidebar.header("Configuration")

    # Model selection
    model_options = {
        "Custom CNN": "outputs/final_model_custom.h5",
        "InceptionV3": "outputs/final_model_InceptionV3.h5",
        "VGG16": "outputs/final_model_VGG16.h5",
        "ResNet50": "outputs/final_model_ResNet50.h5"
    }

    selected_model = st.sidebar.selectbox("Select Model Architecture", list(model_options.keys()))
    model_path = model_options[selected_model]

    # Display options
    show_top_predictions = st.sidebar.checkbox("Show Top Predictions Chart", value=True)
    top_n_predictions = st.sidebar.slider("Number of Top Predictions", 5, 20, 10)

    # Load model
    if os.path.exists(model_path):
        model = load_traffic_sign_model(model_path)
        if model is not None:
            st.sidebar.success(f"✅ {selected_model} model loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load model")
            return
    else:
        st.sidebar.error(f"❌ Model file not found: {model_path}")
        st.info("Please train a model first using: `python train.py --architecture custom`")
        return

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Traffic Sign Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a clear image of a traffic sign for recognition"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Image info
            st.write(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Image mode:** {image.mode}")

    with col2:
        st.header("Prediction Results")

        if uploaded_file is not None:
            # Make prediction
            with st.spinner("Analyzing image..."):
                try:
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Predict
                    result = predict_traffic_sign(model, image)

                    # Display results
                    st.success("✅ Analysis Complete!")

                    # Main prediction
                    st.metric(
                        label="Predicted Traffic Sign",
                        value=result['class_name'],
                        delta=f"{result['confidence']:.1f}% confidence"
                    )

                    # Confidence indicator
                    if result['confidence'] > 80:
                        st.success(f"🎯 High confidence prediction ({result['confidence']:.1f}%)")
                    elif result['confidence'] > 60:
                        st.warning(f"⚠️ Moderate confidence prediction ({result['confidence']:.1f}%)")
                    else:
                        st.error(f"❌ Low confidence prediction ({result['confidence']:.1f}%)")

                    # Additional info
                    st.write(f"**Class ID:** {result['class_id']}")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    return
        else:
            st.info("👆 Upload an image to see predictions here")

    # Prediction chart
    if uploaded_file is not None and show_top_predictions:
        st.header("Detailed Prediction Analysis")

        try:
            # Create and display chart
            fig = create_prediction_chart(result['all_probabilities'], top_n_predictions)
            st.plotly_chart(fig, use_container_width=True)

            # Raw probabilities (expandable)
            with st.expander("View All Class Probabilities"):
                prob_df = []
                for class_id, prob in enumerate(result['all_probabilities']):
                    prob_df.append({
                        'Class ID': class_id,
                        'Class Name': GTSRB_CLASS_NAMES[class_id],
                        'Probability (%)': prob * 100
                    })

                st.dataframe(prob_df)

        except Exception as e:
            st.error(f"Error creating prediction chart: {e}")

    # Information section
    st.header("About This System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "**🎯 Accuracy**\n"
            "State-of-the-art CNN models\n"
            "trained on GTSRB dataset\n"
            "achieving >95% accuracy"
        )

    with col2:
        st.info(
            "**⚡ Speed**\n"
            "Real-time inference\n"
            "optimized for edge deployment\n"
            "<100ms processing time"
        )

    with col3:
        st.info(
            "**🌍 Coverage**\n"
            "43 traffic sign classes\n"
            "European traffic signs\n"
            "Extensible to other regions"
        )

    # Sample images section
    st.header("Try Sample Images")
    st.markdown("Don't have a traffic sign image? Try these sample images:")

    # You would need to add sample images to a 'samples' folder
    sample_folder = "samples"
    if os.path.exists(sample_folder):
        sample_images = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if sample_images:
            selected_sample = st.selectbox("Select a sample image", ["None"] + sample_images)

            if selected_sample != "None":
                sample_path = os.path.join(sample_folder, selected_sample)
                sample_image = Image.open(sample_path)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(sample_image, caption=f"Sample: {selected_sample}", use_column_width=True)

                with col2:
                    if st.button("Analyze Sample Image"):
                        result = predict_traffic_sign(model, sample_image)
                        st.metric(
                            label="Prediction",
                            value=result['class_name'],
                            delta=f"{result['confidence']:.1f}% confidence"
                        )

    # Footer
    st.markdown("---")
    st.markdown(
        "**Traffic Sign Recognition System** | "
        "Built with TensorFlow & Streamlit | "
        "[GitHub Repository](https://github.com/yourusername/traffic-sign-recognition)"
    )

if __name__ == "__main__":
    main()

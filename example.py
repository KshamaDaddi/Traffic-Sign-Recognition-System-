#!/usr/bin/env python3
"""
Simple example script demonstrating traffic sign recognition
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def run_simple_example():
    """Run a simple example of the traffic sign recognition system"""
    print("Traffic Sign Recognition - Simple Example")
    print("="*50)

    # Check if model exists
    model_path = "outputs/final_model_custom.h5"
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        print("Please train a model first:")
        print("  python train.py --architecture custom --epochs 20")
        return

    try:
        from traffic_sign_model import TrafficSignCNN
        from data_utils import GTSRB_CLASS_NAMES
        from tensorflow.keras.models import load_model

        print("✅ Loading trained model...")
        model = load_model(model_path)

        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output classes: {model.output_shape[1]}")

        # Create a synthetic test image (random noise - just for demo)
        print("\n🔍 Testing with synthetic image...")
        test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        test_image_normalized = test_image.astype('float32') / 255.0
        test_batch = np.expand_dims(test_image_normalized, axis=0)

        # Make prediction
        prediction = model.predict(test_batch, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100

        class_name = GTSRB_CLASS_NAMES.get(predicted_class, f"Unknown (Class {predicted_class})")

        print(f"\n📊 Prediction Results:")
        print(f"   Predicted class: {predicted_class}")
        print(f"   Class name: {class_name}")
        print(f"   Confidence: {confidence:.1f}%")

        # Show top 5 predictions
        top_5_indices = np.argsort(prediction[0])[-5:][::-1]
        print(f"\n🔝 Top 5 Predictions:")
        for i, idx in enumerate(top_5_indices, 1):
            class_name = GTSRB_CLASS_NAMES[idx]
            conf = prediction[0][idx] * 100
            print(f"   {i}. {class_name}: {conf:.1f}%")

        # Display the test image
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(test_image)
        plt.title('Test Image\n(Random Synthetic Data)')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.bar(range(len(top_5_indices)), [prediction[0][i] * 100 for i in top_5_indices])
        plt.title('Top 5 Prediction Confidence')
        plt.xlabel('Rank')
        plt.ylabel('Confidence (%)')

        plt.tight_layout()
        plt.savefig('example_prediction.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n✅ Example completed successfully!")
        print(f"   Prediction visualization saved as 'example_prediction.png'")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error running example: {e}")

def show_dataset_info():
    """Show information about the dataset"""
    print("\n📊 GTSRB Dataset Information:")
    print("-" * 30)
    print("Classes: 43 traffic sign categories")
    print("Training images: ~39,000")
    print("Test images: ~12,600") 
    print("Image sizes: Variable (15x15 to 250x250)")
    print("Categories:")
    print("  • Speed limit signs (8 classes)")
    print("  • Warning signs (11 classes)")
    print("  • Prohibition signs (16 classes)")
    print("  • Mandatory signs (5 classes)")
    print("  • Other signs (3 classes)")

def show_next_steps():
    """Show recommended next steps"""
    print("\n🚀 Next Steps:")
    print("-" * 20)

    if not os.path.exists("data"):
        print("1. 📥 Download dataset:")
        print("   python train.py --download_data")
    else:
        print("✅ Dataset directory exists")

    if not os.path.exists("outputs/final_model_custom.h5"):
        print("2. 🧠 Train your first model:")
        print("   python train.py --architecture custom --epochs 20")
    else:
        print("✅ Trained model found")

    print("3. 📈 Evaluate model performance:")
    print("   python evaluate.py --model_path outputs/final_model_custom.h5")

    print("4. 🎥 Try real-time detection:")
    print("   python realtime_detection.py --model_path outputs/final_model_custom.h5")

    print("5. 🌐 Launch web application:")
    print("   streamlit run streamlit_app.py")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--info':
        show_dataset_info()
        show_next_steps()
    else:
        run_simple_example()
        show_next_steps()

if __name__ == "__main__":
    main()

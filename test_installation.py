
#!/usr/bin/env python3
"""
Simple test script to verify installation and basic functionality
"""

import sys
import os
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'tensorflow',
        'keras', 
        'numpy',
        'cv2',
        'PIL',
        'sklearn',
        'matplotlib',
        'pandas',
        'tqdm'
    ]

    print("Testing package imports...")
    failed_imports = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)

    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True

def test_gpu():
    """Test GPU availability"""
    try:
        import tensorflow as tf
        print(f"\nTensorFlow version: {tf.__version__}")

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
        else:
            print("⚠ No GPU detected - will use CPU")

        return True
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    required_dirs = ['data', 'outputs', 'models', 'evaluation_results']

    print("\nChecking directories...")
    missing_dirs = []

    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (missing)")
            missing_dirs.append(dir_name)

    if missing_dirs:
        print(f"\nCreating missing directories...")
        for dir_name in missing_dirs:
            os.makedirs(dir_name, exist_ok=True)
            print(f"Created {dir_name}/")

    return True

def test_model_creation():
    """Test basic model creation"""
    try:
        print("\nTesting model creation...")
        from traffic_sign_model import TrafficSignCNN

        # Test custom CNN
        model = TrafficSignCNN(num_classes=43, architecture='custom')
        model.build_model()
        print("✓ Custom CNN model created successfully")

        # Test model compilation
        model.compile_model()
        print("✓ Model compiled successfully")

        return True

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def main():
    print("Traffic Sign Recognition System - Installation Test")
    print("=" * 60)

    tests = [
        ("Package Imports", test_imports),
        ("GPU Detection", test_gpu), 
        ("Directory Structure", test_directories),
        ("Model Creation", test_model_creation)
    ]

    all_passed = True

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Download data: python train.py --download_data")
        print("2. Train model: python train.py --architecture custom --epochs 50")
        print("3. Test model: python evaluate.py --model_path outputs/final_model_custom.h5")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())

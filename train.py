
import os
import numpy as np
import matplotlib.pyplot as plt
from traffic_sign_model import TrafficSignCNN
from data_utils import TrafficSignDataProcessor, GTSRB_CLASS_NAMES
import argparse
import json
from datetime import datetime

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def save_training_config(config, filepath='training_config.json'):
    """Save training configuration"""
    config['timestamp'] = datetime.now().isoformat()

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Training configuration saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Train Traffic Sign Recognition Model')
    parser.add_argument('--architecture', type=str, default='custom',
                       choices=['custom', 'InceptionV3', 'VGG16', 'ResNet50'],
                       help='CNN architecture to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--data_path', type=str, default='data/',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/',
                       help='Output directory for models and results')
    parser.add_argument('--download_data', action='store_true',
                       help='Download GTSRB dataset')
    parser.add_argument('--augment_data', action='store_true', default=True,
                       help='Apply data augmentation')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize data processor
    print("Initializing data processor...")
    processor = TrafficSignDataProcessor(data_path=args.data_path)

    # Download data if requested
    if args.download_data:
        processor.download_gtsrb_dataset()

    # Load and preprocess data
    try:
        print("Loading preprocessed data...")
        X_train, y_train, X_val, y_val, X_test, y_test = processor.load_preprocessed_data()
    except FileNotFoundError:
        print("Preprocessed data not found. Loading and preprocessing raw data...")
        X_train_raw, y_train_raw, X_test_raw, y_test_raw = processor.load_gtsrb_data()

        # Preprocess data
        X_train, y_train, X_val, y_val, X_test, y_test = processor.preprocess_data(
            X_train_raw, y_train_raw, X_test_raw, y_test_raw,
            augment=args.augment_data
        )

        # Save preprocessed data
        processor.save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Visualize sample data
    processor.visualize_samples(X_train[:1000], np.argmax(y_train[:1000], axis=1), 
                               class_names=list(GTSRB_CLASS_NAMES.values()))

    # Initialize model
    print(f"Initializing {args.architecture} model...")
    input_shape = X_train.shape[1:]
    if args.architecture == 'InceptionV3':
        input_shape = (75, 75, 3)  # InceptionV3 requires minimum 75x75
        # Resize data for InceptionV3
        import cv2
        X_train_resized = np.array([cv2.resize(img, (75, 75)) for img in X_train])
        X_val_resized = np.array([cv2.resize(img, (75, 75)) for img in X_val])
        X_test_resized = np.array([cv2.resize(img, (75, 75)) for img in X_test])
        X_train, X_val, X_test = X_train_resized, X_val_resized, X_test_resized

    model = TrafficSignCNN(
        num_classes=43,
        input_shape=input_shape,
        architecture=args.architecture
    )

    # Build and compile model
    model.compile_model(learning_rate=args.learning_rate)

    print("Model architecture:")
    model.model.summary()

    # Train model
    print("Starting training...")
    checkpoint_path = os.path.join(args.output_dir, f'best_model_{args.architecture}.h5')
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Save training history plot
    history_plot_path = os.path.join(args.output_dir, f'training_history_{args.architecture}.png')
    plot_training_history(history, history_plot_path)

    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate(X_test, y_test)

    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])

    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, f'confusion_matrix_{args.architecture}.png')
    model.plot_confusion_matrix(
        results['confusion_matrix'], 
        class_names=list(GTSRB_CLASS_NAMES.values()),
        figsize=(15, 12)
    )
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')

    # Save final model
    final_model_path = os.path.join(args.output_dir, f'final_model_{args.architecture}.h5')
    model.model.save(final_model_path)

    # Save training configuration and results
    config = {
        'architecture': args.architecture,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'augment_data': args.augment_data,
        'test_accuracy': float(results['test_accuracy']),
        'test_loss': float(results['test_loss']),
        'model_path': final_model_path,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test)
    }

    config_path = os.path.join(args.output_dir, f'training_config_{args.architecture}.json')
    save_training_config(config, config_path)

    print(f"\nTraining completed successfully!")
    print(f"Model saved to: {final_model_path}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

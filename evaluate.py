
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_utils import GTSRB_CLASS_NAMES
import argparse
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class TrafficSignEvaluator:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.class_names = GTSRB_CLASS_NAMES

    def predict_image(self, image_path, target_size=(32, 32)):
        """Predict single image"""
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        original_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = self.model.predict(img, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        return {
            'predicted_class': int(predicted_class),
            'class_name': self.class_names[predicted_class],
            'confidence': float(confidence),
            'original_image': original_img,
            'prediction_probabilities': prediction[0]
        }

    def predict_batch(self, image_paths, target_size=(32, 32)):
        """Predict batch of images"""
        results = []

        for image_path in image_paths:
            try:
                result = self.predict_image(image_path, target_size)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        return results

    def visualize_predictions(self, results, num_samples=10, figsize=(15, 10)):
        """Visualize prediction results"""
        num_samples = min(num_samples, len(results))

        cols = 5
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_samples > 1 else [axes]

        for i in range(num_samples):
            result = results[i]
            img = cv2.cvtColor(result['original_image'], cv2.COLOR_BGR2RGB)

            axes[i].imshow(img)
            axes[i].set_title(f"{result['class_name']}\nConfidence: {result['confidence']:.1f}%", 
                            fontsize=8)
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        return fig

    def evaluate_test_set(self, X_test, y_test):
        """Evaluate model on test set"""
        print("Evaluating model on test set...")

        # Get predictions
        y_pred = self.model.predict(X_test, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        # Classification report
        report = classification_report(y_true_classes, y_pred_classes, 
                                     target_names=list(self.class_names.values()))

        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred_classes,
            'y_true': y_true_classes
        }

    def plot_confusion_matrix(self, cm, save_path=None, figsize=(12, 10)):
        """Plot confusion matrix"""
        plt.figure(figsize=figsize)

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=list(self.class_names.values()),
                   yticklabels=list(self.class_names.values()))

        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def analyze_misclassifications(self, X_test, y_test, num_examples=10):
        """Analyze misclassified examples"""
        # Get predictions
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Find misclassified samples
        misclassified_idx = np.where(y_pred_classes != y_true_classes)[0]

        if len(misclassified_idx) == 0:
            print("No misclassifications found!")
            return

        print(f"Found {len(misclassified_idx)} misclassified samples")

        # Visualize some misclassifications
        num_examples = min(num_examples, len(misclassified_idx))

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i in range(num_examples):
            idx = misclassified_idx[i]
            img = X_test[idx]

            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

            true_class = y_true_classes[idx]
            pred_class = y_pred_classes[idx]
            confidence = np.max(y_pred[idx]) * 100

            axes[i].imshow(img)
            axes[i].set_title(f"True: {self.class_names[true_class][:15]}...\n"
                            f"Pred: {self.class_names[pred_class][:15]}...\n"
                            f"Conf: {confidence:.1f}%", fontsize=8)
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        return fig

def main():
    parser = argparse.ArgumentParser(description='Evaluate Traffic Sign Recognition Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test_images', type=str, nargs='+',
                       help='Path(s) to test images')
    parser.add_argument('--test_folder', type=str,
                       help='Folder containing test images')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/',
                       help='Output directory for results')
    parser.add_argument('--target_size', type=int, nargs=2, default=[32, 32],
                       help='Target image size (width height)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluator
    print(f"Loading model from {args.model_path}")
    evaluator = TrafficSignEvaluator(args.model_path)

    target_size = tuple(args.target_size)

    # Handle different input types
    if args.test_images:
        # Predict specific images
        print(f"Predicting {len(args.test_images)} images...")
        results = evaluator.predict_batch(args.test_images, target_size)

        # Visualize results
        fig = evaluator.visualize_predictions(results)
        fig.savefig(os.path.join(args.output_dir, 'prediction_results.png'), 
                   dpi=300, bbox_inches='tight')

        # Save results to JSON
        results_json = []
        for result in results:
            result_copy = result.copy()
            # Remove non-serializable items
            result_copy.pop('original_image', None)
            result_copy['prediction_probabilities'] = result_copy['prediction_probabilities'].tolist()
            results_json.append(result_copy)

        with open(os.path.join(args.output_dir, 'prediction_results.json'), 'w') as f:
            json.dump(results_json, f, indent=2)

    elif args.test_folder:
        # Predict all images in folder
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []

        for ext in image_extensions:
            image_paths.extend([os.path.join(args.test_folder, f) 
                               for f in os.listdir(args.test_folder) 
                               if f.lower().endswith(ext)])

        if not image_paths:
            print(f"No images found in {args.test_folder}")
            return

        print(f"Predicting {len(image_paths)} images from folder...")
        results = evaluator.predict_batch(image_paths, target_size)

        # Visualize results
        fig = evaluator.visualize_predictions(results, num_samples=20, figsize=(20, 12))
        fig.savefig(os.path.join(args.output_dir, 'folder_prediction_results.png'), 
                   dpi=300, bbox_inches='tight')

    else:
        print("Please provide either --test_images or --test_folder")
        return

    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Model comparison script for different architectures
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def load_training_results(output_dir='outputs/'):
    """Load training results from JSON files"""
    results = {}

    config_files = Path(output_dir).glob('training_config_*.json')

    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            architecture = config.get('architecture', 'unknown')
            results[architecture] = config

        except Exception as e:
            print(f"Error loading {config_file}: {e}")

    return results

def create_comparison_report(results):
    """Create detailed comparison report"""
    if not results:
        print("No training results found in outputs/ directory.")
        print("Train some models first using: python train.py --architecture <arch>")
        return

    print("Model Performance Comparison")
    print("=" * 60)

    # Create comparison table
    comparison_data = []

    for arch, config in results.items():
        comparison_data.append({
            'Architecture': arch,
            'Test Accuracy': f"{config.get('test_accuracy', 0):.4f}",
            'Test Loss': f"{config.get('test_loss', 0):.4f}",
            'Training Samples': config.get('training_samples', 'N/A'),
            'Epochs': config.get('epochs', 'N/A'),
            'Batch Size': config.get('batch_size', 'N/A'),
            'Learning Rate': config.get('learning_rate', 'N/A')
        })

    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))

    # Save comparison to CSV
    df.to_csv('outputs/model_comparison.csv', index=False)
    print(f"\nComparison saved to outputs/model_comparison.csv")

    # Create accuracy comparison plot
    if len(results) > 1:
        architectures = list(results.keys())
        accuracies = [results[arch]['test_accuracy'] for arch in architectures]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(architectures, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.title('Model Architecture Comparison - Test Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Architecture', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)

        # Add horizontal line at 0.95 (target accuracy)
        plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target (95%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig('outputs/accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nAccuracy comparison plot saved to outputs/accuracy_comparison.png")

def find_best_model(results):
    """Find the best performing model"""
    if not results:
        return None

    best_arch = max(results.keys(), key=lambda k: results[k].get('test_accuracy', 0))
    best_accuracy = results[best_arch]['test_accuracy']

    print(f"\n🏆 Best Model: {best_arch}")
    print(f"   Accuracy: {best_accuracy:.4f}")
    print(f"   Model Path: {results[best_arch].get('model_path', 'N/A')}")

    return best_arch, results[best_arch]

def recommend_next_steps(results):
    """Provide recommendations based on results"""
    print("\n📋 Recommendations:")
    print("-" * 30)

    if not results:
        print("• Train your first model: python train.py --architecture custom")
        return

    best_arch, best_config = find_best_model(results)
    best_accuracy = best_config['test_accuracy']

    if best_accuracy < 0.90:
        print("• Accuracy is low. Consider:")
        print("  - Increasing training epochs")
        print("  - Adding data augmentation")
        print("  - Trying transfer learning (InceptionV3)")

    elif best_accuracy < 0.95:
        print("• Good accuracy! To improve further:")
        print("  - Fine-tune hyperparameters")
        print("  - Try ensemble methods")
        print("  - Increase model complexity")

    else:
        print("• Excellent accuracy! Ready for deployment:")
        print(f"  - Use model: {best_config.get('model_path', 'N/A')}")
        print("  - Test real-time detection")
        print("  - Deploy web application")

    # Check if we have multiple architectures
    if len(results) == 1:
        print("\n• Train additional architectures for comparison:")
        untrained = ['custom', 'InceptionV3', 'VGG16', 'ResNet50']
        current = list(results.keys())
        suggestions = [arch for arch in untrained if arch not in current]

        for arch in suggestions[:2]:  # Suggest 2 more
            print(f"  python train.py --architecture {arch}")

def main():
    print("Loading training results...")
    results = load_training_results()

    create_comparison_report(results)
    recommend_next_steps(results)

if __name__ == "__main__":
    main()

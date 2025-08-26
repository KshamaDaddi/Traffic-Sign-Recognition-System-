
#!/usr/bin/env python3
"""
Display information about the GTSRB dataset and class distribution
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os
from data_utils import GTSRB_CLASS_NAMES, TrafficSignDataProcessor

def analyze_dataset():
    """Analyze the GTSRB dataset"""
    print("GTSRB Dataset Analysis")
    print("=" * 50)

    # Check if preprocessed data exists
    if os.path.exists('preprocessed_data.pkl'):
        print("Loading preprocessed data...")
        processor = TrafficSignDataProcessor()
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = processor.load_preprocessed_data()

            # Convert one-hot back to class indices
            if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                y_train_classes = np.argmax(y_train, axis=1)
                y_val_classes = np.argmax(y_val, axis=1)
                y_test_classes = np.argmax(y_test, axis=1)
            else:
                y_train_classes = y_train
                y_val_classes = y_val  
                y_test_classes = y_test

            analyze_data_distribution(X_train, y_train_classes, X_val, y_val_classes, X_test, y_test_classes)

        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            print("Run 'python train.py --download_data' first")

    else:
        print("No preprocessed data found.")
        print("To analyze dataset:")
        print("1. python train.py --download_data")
        print("2. python dataset_info.py")

def analyze_data_distribution(X_train, y_train, X_val, y_val, X_test, y_test):
    """Analyze class distribution and data statistics"""

    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Total samples: {len(X_train) + len(X_val) + len(X_test):,}")
    print(f"Image shape: {X_train.shape[1:]}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    # Class distribution analysis
    print("\nClass Distribution Analysis:")
    print("-" * 30)

    train_counts = Counter(y_train)
    val_counts = Counter(y_val)
    test_counts = Counter(y_test)

    # Create distribution dataframe
    distribution_data = []

    for class_id in range(43):
        class_name = GTSRB_CLASS_NAMES.get(class_id, f"Class {class_id}")

        distribution_data.append({
            'Class ID': class_id,
            'Class Name': class_name,
            'Train': train_counts.get(class_id, 0),
            'Val': val_counts.get(class_id, 0),
            'Test': test_counts.get(class_id, 0),
            'Total': train_counts.get(class_id, 0) + val_counts.get(class_id, 0) + test_counts.get(class_id, 0)
        })

    df = pd.DataFrame(distribution_data)

    # Display statistics
    print(f"Most common class: {df.loc[df['Total'].idxmax(), 'Class Name']} ({df['Total'].max():,} samples)")
    print(f"Least common class: {df.loc[df['Total'].idxmin(), 'Class Name']} ({df['Total'].min():,} samples)")
    print(f"Average samples per class: {df['Total'].mean():.1f}")
    print(f"Standard deviation: {df['Total'].std():.1f}")

    # Check for class imbalance
    imbalance_ratio = df['Total'].max() / df['Total'].min()
    print(f"Class imbalance ratio: {imbalance_ratio:.1f}:1")

    if imbalance_ratio > 10:
        print("⚠️  High class imbalance detected!")
        print("   Consider using class weights or SMOTE for balancing")
    elif imbalance_ratio > 3:
        print("⚠️  Moderate class imbalance detected")
        print("   Monitor performance on minority classes")
    else:
        print("✓ Relatively balanced dataset")

    # Save distribution to CSV
    df.to_csv('outputs/class_distribution.csv', index=False)
    print("\nClass distribution saved to outputs/class_distribution.csv")

    # Create visualizations
    create_distribution_plots(df)

    # Show top/bottom classes
    print("\nTop 5 most common classes:")
    top_5 = df.nlargest(5, 'Total')[['Class Name', 'Total']]
    print(top_5.to_string(index=False))

    print("\nTop 5 least common classes:")
    bottom_5 = df.nsmallest(5, 'Total')[['Class Name', 'Total']]
    print(bottom_5.to_string(index=False))

def create_distribution_plots(df):
    """Create visualization plots for class distribution"""

    # Class distribution histogram
    plt.figure(figsize=(15, 10))

    # Plot 1: Overall distribution
    plt.subplot(2, 2, 1)
    plt.bar(range(43), df['Total'], color='skyblue', alpha=0.7)
    plt.xlabel('Class ID')
    plt.ylabel('Number of Samples')
    plt.title('Sample Distribution by Class')
    plt.grid(axis='y', alpha=0.3)

    # Plot 2: Train/Val/Test split
    plt.subplot(2, 2, 2)
    x = np.arange(43)
    width = 0.25

    plt.bar(x - width, df['Train'], width, label='Train', alpha=0.7)
    plt.bar(x, df['Val'], width, label='Validation', alpha=0.7)
    plt.bar(x + width, df['Test'], width, label='Test', alpha=0.7)

    plt.xlabel('Class ID')
    plt.ylabel('Number of Samples')
    plt.title('Train/Validation/Test Split by Class')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # Plot 3: Distribution histogram
    plt.subplot(2, 2, 3)
    plt.hist(df['Total'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Samples per Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Samples per Class')
    plt.grid(axis='y', alpha=0.3)

    # Plot 4: Cumulative distribution
    plt.subplot(2, 2, 4)
    sorted_counts = df['Total'].sort_values()
    cumulative = np.cumsum(sorted_counts) / sorted_counts.sum() * 100

    plt.plot(range(43), cumulative, marker='o', markersize=4)
    plt.xlabel('Class Rank (sorted by sample count)')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Sample Distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/class_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Distribution plots saved to outputs/class_distribution_analysis.png")

def display_class_info():
    """Display detailed information about each traffic sign class"""
    print("\nGTSRB Traffic Sign Classes:")
    print("=" * 80)

    # Group classes by type
    speed_limits = {k: v for k, v in GTSRB_CLASS_NAMES.items() if 'Speed limit' in v}
    prohibitions = {k: v for k, v in GTSRB_CLASS_NAMES.items() if 'No ' in v or 'prohibited' in v}
    warnings = {k: v for k, v in GTSRB_CLASS_NAMES.items() if any(word in v.lower() for word in ['dangerous', 'caution', 'slippery', 'bumpy', 'crossing'])}
    mandatory = {k: v for k, v in GTSRB_CLASS_NAMES.items() if any(word in v.lower() for word in ['ahead', 'right', 'left', 'keep', 'roundabout', 'mandatory'])}

    categories = [
        ("Speed Limit Signs", speed_limits),
        ("Prohibition Signs", prohibitions), 
        ("Warning Signs", warnings),
        ("Mandatory Signs", mandatory)
    ]

    for category_name, signs in categories:
        if signs:
            print(f"\n{category_name} ({len(signs)} classes):")
            for class_id, name in sorted(signs.items()):
                print(f"  {class_id:2d}: {name}")

    # Show remaining classes
    categorized_ids = set()
    for _, signs in categories:
        categorized_ids.update(signs.keys())

    remaining = {k: v for k, v in GTSRB_CLASS_NAMES.items() if k not in categorized_ids}

    if remaining:
        print(f"\nOther Signs ({len(remaining)} classes):")
        for class_id, name in sorted(remaining.items()):
            print(f"  {class_id:2d}: {name}")

def main():
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)

    display_class_info()
    analyze_dataset()

if __name__ == "__main__":
    main()

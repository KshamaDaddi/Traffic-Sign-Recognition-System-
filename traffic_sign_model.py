
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3, VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class TrafficSignCNN:
    def __init__(self, num_classes=43, input_shape=(32, 32, 3), architecture='custom'):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.architecture = architecture
        self.model = None

    def build_custom_cnn(self):
        """Build custom CNN architecture optimized for traffic signs"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model

    def build_transfer_learning_model(self, base_model_name='InceptionV3'):
        """Build model using transfer learning"""
        # Load pre-trained base model
        if base_model_name == 'InceptionV3':
            base_model = InceptionV3(weights='imagenet', include_top=False, 
                                   input_shape=(75, 75, 3))  # InceptionV3 requires min 75x75
        elif base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, 
                             input_shape=self.input_shape)
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, 
                                input_shape=self.input_shape)

        # Freeze base model layers
        base_model.trainable = False

        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model

    def build_model(self):
        """Build the selected model architecture"""
        if self.architecture == 'custom':
            self.model = self.build_custom_cnn()
        elif self.architecture in ['InceptionV3', 'VGG16', 'ResNet50']:
            self.model = self.build_transfer_learning_model(self.architecture)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        return self.model

    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        if self.model is None:
            self.build_model()

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def get_callbacks(self, checkpoint_path='best_model.h5'):
        """Get training callbacks"""
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.compile_model()

        callbacks = self.get_callbacks()

        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        # Classification report
        report = classification_report(y_true_classes, y_pred_classes)

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

    def plot_confusion_matrix(self, cm, class_names=None, figsize=(12, 10)):
        """Plot confusion matrix"""
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

    def predict_single_image(self, image_path):
        """Predict single image"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        return predicted_class, confidence

#!/usr/bin/env python3
"""
Binary Booking Detection Model
Classifies text as booking-related (flight, car, hotel) or not
Returns: 0 (false) or 1 (true)

Author: AI Assistant
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional,
    GlobalAveragePooling1D, LayerNormalization
)

# 兼容性导入 - 处理preprocessing模块可能的导入问题
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    PREPROCESSING_AVAILABLE = True
except ImportError:
    print("⚠️ 传统preprocessing模块不可用，将使用现代化TextVectorization")
    PREPROCESSING_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinaryBookingDetector:
    """
    Binary classifier to detect booking-related text
    Returns 1 for booking-related content (flight, car, hotel), 0 otherwise
    """

    def __init__(self, max_features=10000, max_length=100, embedding_dim=128, lstm_units=64):
        """
        Initialize the Binary Booking Detector

        Args:
            max_features (int): Maximum number of words to keep in vocabulary
            max_length (int): Maximum sequence length for padding
            embedding_dim (int): Dimension of word embeddings
            lstm_units (int): Number of LSTM units
        """
        self.max_features = max_features
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units

        self.tokenizer = None
        self.model = None
        self.history = None
        self.use_traditional_preprocessing = PREPROCESSING_AVAILABLE

        # Booking-related keywords for better preprocessing
        self.booking_keywords = {
            'flight': ['flight', 'airplane', 'airline', 'airport', 'boarding', 'departure', 'arrival', 'terminal'],
            'hotel': ['hotel', 'room', 'reservation', 'check-in', 'check-out', 'accommodation', 'suite', 'lobby'],
            'car': ['car', 'rental', 'vehicle', 'driving', 'pickup', 'dropoff', 'automobile', 'lease']
        }

        # Performance metrics storage
        self.metrics = {
            'train_accuracy': [],
            'val_accuracy': [],
            'train_loss': [],
            'val_loss': []
        }

    def preprocess_text(self, text):
        """
        Clean and preprocess text data while preserving booking information

        Args:
            text (str): Raw text to preprocess

        Returns:
            str: Cleaned text with preserved booking information
        """
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Replace common separators with spaces to preserve word boundaries
        text = re.sub(r'[_\-/\\]', ' ', text)

        # Remove only pure punctuation, preserve alphanumeric combinations
        # This keeps flight numbers (aa1234), dates (15th), room numbers (205a), etc.
        text = re.sub(r'[^\w\s]', ' ', text)

        # Normalize multiple spaces to single space
        text = ' '.join(text.split())

        return text

    def create_binary_labels(self, df, text_column='content', class_column='class'):
        """
        Convert multi-class labels to binary labels

        Args:
            df (DataFrame): Input dataframe
            text_column (str): Name of text column
            class_column (str): Name of class column

        Returns:
            DataFrame: Dataframe with binary labels
        """
        logger.info("Converting to binary classification...")

        # Define booking-related classes
        booking_classes = ['flight', 'hotel', 'car', 'flight booking', 'hotel booking', 'car rental', 'car renting']

        # Create binary labels
        df['is_booking'] = df[class_column].str.lower().apply(
            lambda x: 1 if any(booking_class in str(x).lower() for booking_class in booking_classes) else 0
        )

        logger.info(f"Original classes: {df[class_column].value_counts().to_dict()}")
        logger.info(f"Binary distribution: {df['is_booking'].value_counts().to_dict()}")

        return df

    def load_and_preprocess_data(self, csv_file_path):
        """
        Load and preprocess data from CSV file for binary classification

        Args:
            csv_file_path (str): Path to CSV file with 'content' and 'class' columns

        Returns:
            tuple: Processed features and binary labels
        """
        logger.info(f"Loading data from {csv_file_path}")

        # Load CSV data
        df = pd.read_csv(csv_file_path)

        # Validate required columns
        required_columns = ['content', 'class']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        logger.info(f"Loaded {len(df)} records")

        # Convert to binary classification
        df = self.create_binary_labels(df)

        # Preprocess text content
        logger.info("Preprocessing text content...")
        df['content'] = df['content'].apply(self.preprocess_text)

        # Remove empty content
        df = df[df['content'].str.len() > 0]

        # Extract features and binary labels
        X = df['content'].values
        y = df['is_booking'].values  # Binary labels (0 or 1)

        logger.info(f"Binary distribution after preprocessing:")
        logger.info(f"Non-booking (0): {np.sum(y == 0)} samples")
        logger.info(f"Booking (1): {np.sum(y == 1)} samples")

        return X, y

    def prepare_sequences(self, X_train, X_val, X_test):
        """
        Tokenize and pad text sequences

        Args:
            X_train, X_val, X_test: Text data splits

        Returns:
            tuple: Padded sequences for train, validation, and test sets
        """
        logger.info("Tokenizing text sequences...")

        # Initialize and fit tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)

        # Convert texts to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq = self.tokenizer.texts_to_sequences(X_val)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)

        # Pad sequences
        X_train_padded = pad_sequences(X_train_seq, maxlen=self.max_length, padding='post')
        X_val_padded = pad_sequences(X_val_seq, maxlen=self.max_length, padding='post')
        X_test_padded = pad_sequences(X_test_seq, maxlen=self.max_length, padding='post')

        logger.info(f"Vocabulary size: {len(self.tokenizer.word_index) + 1}")
        logger.info(f"Sequence shape: {X_train_padded.shape}")

        return X_train_padded, X_val_padded, X_test_padded

    def build_model(self):
        """
        Build simplified binary classification model
        """
        logger.info("Building binary classification model...")

        # Input layer
        inputs = Input(shape=(self.max_length,), name='input')

        # Embedding layer
        embedding = Embedding(
            input_dim=self.max_features,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            mask_zero=True,
            name='embedding'
        )(inputs)

        # Bidirectional LSTM layer
        lstm_out = Bidirectional(LSTM(
            self.lstm_units,
            dropout=0.3,
            recurrent_dropout=0.3,
            return_sequences=True
        ), name='bidirectional_lstm')(embedding)

        # Global average pooling
        pooled = GlobalAveragePooling1D(name='global_avg_pool')(lstm_out)

        # Dense layers
        dense1 = Dense(64, activation='relu', name='dense_1')(pooled)
        dropout1 = Dropout(0.5, name='dropout_1')(dense1)

        dense2 = Dense(32, activation='relu', name='dense_2')(dropout1)
        dropout2 = Dropout(0.3, name='dropout_2')(dense2)

        # Binary output layer (sigmoid activation for binary classification)
        outputs = Dense(1, activation='sigmoid', name='output')(dropout2)

        # Create the model
        self.model = Model(inputs=inputs, outputs=outputs, name='binary_booking_detector')

        # Compile model for binary classification
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',  # Binary crossentropy for binary classification
            metrics=['accuracy', 'precision', 'recall']
        )

        # Print model summary
        self.model.summary()

        logger.info("Binary classification model built successfully")

    def train_model(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
        """
        Train the binary classification model

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        logger.info(f"Starting training for {epochs} epochs...")

        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Store metrics
        self.metrics['train_accuracy'] = self.history.history['accuracy']
        self.metrics['val_accuracy'] = self.history.history['val_accuracy']
        self.metrics['train_loss'] = self.history.history['loss']
        self.metrics['val_loss'] = self.history.history['val_loss']

        logger.info("Training completed successfully")

    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation for binary classification

        Args:
            X_test: Test features
            y_test: Test labels (binary)
        """
        logger.info("Evaluating model performance...")

        # Model predictions
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )

        # Get prediction probabilities
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions

        # Calculate comprehensive metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )

        # Print performance summary
        print("\n" + "="*50)
        print("BINARY CLASSIFICATION PERFORMANCE")
        print("="*50)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # Detailed classification report
        print("\nDETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['Non-Booking', 'Booking']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=['Non-Booking (0)', 'Booking (1)'],
            yticklabels=['Non-Booking (0)', 'Booking (1)'],
            cmap='Blues'
        )
        plt.title('Binary Classification Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('binary_confusion_matrix.png', dpi=300)
        plt.show()

        return {
            'accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'loss': test_loss
        }

    def plot_training_history(self):
        """
        Plot training history for analysis
        """
        if self.history is None:
            logger.warning("No training history available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy plot
        axes[0].plot(self.metrics['train_accuracy'], label='Training Accuracy')
        axes[0].plot(self.metrics['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        # Loss plot
        axes[1].plot(self.metrics['train_loss'], label='Training Loss')
        axes[1].plot(self.metrics['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('binary_training_history.png', dpi=300)
        plt.show()

    def predict(self, texts, threshold=0.5):
        """
        Make binary predictions on new text data

        Args:
            texts (list): List of text strings to classify
            threshold (float): Decision threshold (default: 0.5)

        Returns:
            tuple: Binary predictions (0/1) and probabilities
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained. Please train the model first.")

        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Convert to sequences and pad
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')

        # Make predictions
        predictions_prob = self.model.predict(padded_sequences).flatten()
        predictions_binary = (predictions_prob > threshold).astype(int)

        return predictions_binary, predictions_prob

    def convert_to_tflite(self, model_path='binary_booking_detector.tflite', quantization='dynamic'):
        """
        Convert model to TensorFlow Lite for mobile deployment

        Args:
            model_path (str): Path to save TFLite model
            quantization (str): Quantization type - 'none', 'dynamic', 'float16'
        """
        logger.info("Converting binary model to TensorFlow Lite...")

        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Essential settings for LSTM compatibility
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        converter._experimental_lower_tensor_list_ops = False
        converter.experimental_enable_resource_variables = True

        # Apply quantization
        if quantization == 'dynamic':
            logger.info("Applying dynamic range quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization == 'float16':
            logger.info("Applying float16 quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        try:
            # Convert model
            tflite_model = converter.convert()

            # Save TFLite model
            with open(model_path, 'wb') as f:
                f.write(tflite_model)

            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

            logger.info(f"✅ Binary TFLite model saved to {model_path}")
            logger.info(f"📁 Model size: {model_size:.2f} MB")
            logger.info("📱 Model ready for Android deployment")

            return model_path

        except Exception as e:
            logger.error(f"TFLite conversion failed: {e}")
            logger.info("Attempting fallback conversion...")

            # Fallback conversion
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False

            tflite_model = converter.convert()

            with open(model_path, 'wb') as f:
                f.write(tflite_model)

            model_size = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"✅ Fallback binary TFLite model saved to {model_path}")
            logger.info(f"📁 Model size: {model_size:.2f} MB")

            return model_path

    def save_model_components(self, model_dir='binary_booking_artifacts'):
        """
        Save all model components for deployment

        Args:
            model_dir (str): Directory to save model artifacts
        """
        os.makedirs(model_dir, exist_ok=True)

        # Save Keras model
        self.model.save(os.path.join(model_dir, 'binary_booking_model.h5'))

        # Save tokenizer
        with open(os.path.join(model_dir, 'tokenizer.pickle'), 'wb') as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save model configuration
        config = {
            'max_features': self.max_features,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'model_type': 'binary_classification',
            'classes': ['non_booking', 'booking'],
            'output_description': 'Returns 0 for non-booking, 1 for booking-related text'
        }

        import json
        with open(os.path.join(model_dir, 'binary_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Binary model components saved to {model_dir}")


def main():
    """
    Main function for binary booking detection
    """
    # Configuration
    CSV_FILE_PATH = 'booking_data.csv'  # Path to your CSV file
    EPOCHS = 30
    BATCH_SIZE = 32
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2

    try:
        # Initialize binary detector
        detector = BinaryBookingDetector(
            max_features=10000,
            max_length=100,
            embedding_dim=128,
            lstm_units=64
        )

        # Load and preprocess data
        X, y = detector.load_and_preprocess_data(CSV_FILE_PATH)

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=TEST_SIZE + VAL_SIZE, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=TEST_SIZE/(TEST_SIZE + VAL_SIZE),
            random_state=42, stratify=y_temp
        )

        # Prepare sequences
        X_train_seq, X_val_seq, X_test_seq = detector.prepare_sequences(X_train, X_val, X_test)

        # Build model
        detector.build_model()

        # Train model
        detector.train_model(X_train_seq, y_train, X_val_seq, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)

        # Evaluate model
        metrics = detector.evaluate_model(X_test_seq, y_test)

        # Plot training history
        detector.plot_training_history()

        # Test predictions on sample data
        sample_texts = [
            "Flight booking confirmation for New York to Los Angeles AA1234",
            "Hotel reservation at Grand Hotel for 3 nights check-in tomorrow",
            "Car rental Toyota Camry pickup at airport tomorrow",
            "Weather is nice today, going for a walk",
            "Meeting scheduled for 3pm in conference room",
            "Restaurant reservation for dinner tonight"
        ]

        predictions, probabilities = detector.predict(sample_texts)

        print("\nSAMPLE BINARY PREDICTIONS:")
        print("="*60)
        for text, pred, prob in zip(sample_texts, predictions, probabilities):
            result = "✅ BOOKING" if pred == 1 else "❌ NON-BOOKING"
            print(f"Text: {text}")
            print(f"Prediction: {pred} ({result})")
            print(f"Confidence: {prob:.4f}")
            print("-" * 60)

        # Convert to TFLite
        tflite_path = detector.convert_to_tflite('binary_booking_detector.tflite')

        # Save model components
        detector.save_model_components('binary_booking_artifacts')

        print(f"\n🎉 BINARY BOOKING DETECTION COMPLETED!")
        print(f"📊 Model accuracy: {metrics['accuracy']:.4f}")
        print(f"📱 TFLite model: {tflite_path}")
        print(f"💾 Model artifacts: binary_booking_artifacts/")
        print(f"🎯 Ready for Android deployment!")

        print("\n" + "="*60)
        print("BINARY MODEL SUMMARY")
        print("="*60)
        print("📋 Task: Binary classification")
        print("🎯 Output: 0 (non-booking) or 1 (booking)")
        print("📱 Target: flight, car rental, hotel bookings")
        print("📏 Model size optimized for mobile deployment")

    except Exception as e:
        logger.error(f"Error in binary detection pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # System check
    print("=" * 60)
    print("BINARY BOOKING DETECTION - ANDROID DEPLOYMENT")
    print("=" * 60)

    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print("Task: Binary classification (booking vs non-booking)")
    print("Output: 0 (false) or 1 (true)")

    print("=" * 60)
    print("STARTING BINARY TRAINING PIPELINE")
    print("=" * 60)

    main()

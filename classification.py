#!/usr/bin/env python3
"""
Travel Text Binary Classification Model
Classifies text as booking-related (1) or non-booking-related (0)
Returns: 0 (non-booking) or 1 (booking)

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
    GlobalAveragePooling1D, GlobalMaxPooling1D, LayerNormalization,
    Conv1D, MaxPooling1D, Concatenate, BatchNormalization
)

# Compatibility imports - handle preprocessing module import issues
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    PREPROCESSING_AVAILABLE = True
except ImportError:
    print("⚠️ Legacy preprocessing module unavailable, will use modern TextVectorization")
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

class TravelTextClassifier:
    """
    Travel Text Binary Classifier
    Detects if text is booking-related (flights, hotels, cars) or not
    Returns 1 for booking-related, 0 for non-booking
    Optimized for Android deployment with lightweight CNN architecture
    """

    def __init__(self, max_features=8000, max_length=80, embedding_dim=64):
        """
        Initialize Travel Text Classifier with mobile-optimized parameters

        Args:
            max_features (int): Maximum number of words in vocabulary (reduced for mobile)
            max_length (int): Maximum sequence length (reduced for speed)
            embedding_dim (int): Word embedding dimension (reduced for efficiency)
        """
        self.max_features = max_features
        self.max_length = max_length
        self.embedding_dim = embedding_dim

        self.tokenizer = None
        self.model = None
        self.history = None
        self.use_traditional_preprocessing = PREPROCESSING_AVAILABLE

        # Booking-related keywords for better preprocessing
        self.booking_keywords = {
            'flight': ['flight', 'airplane', 'airline', 'airport', 'boarding', 'departure', 'arrival', 'terminal', 'pnr', 'seat', 'passenger'],
            'hotel': ['hotel', 'room', 'reservation', 'check-in', 'check-out', 'accommodation', 'suite', 'lobby', 'guest', 'booking'],
            'car': ['car', 'rental', 'vehicle', 'driving', 'pickup', 'dropoff', 'automobile', 'lease', 'enterprise', 'avis', 'renter'],
            'travel': ['booking', 'confirmation', 'travel', 'trip', 'vacation', 'total', 'fare', 'ref', 'nights']
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
        text = re.sub(r'[_\-/\\|:,]', ' ', text)

        # Keep alphanumeric characters and spaces, remove other punctuation
        # This preserves flight numbers (aa1234), dates (15th), room numbers (205a), etc.
        text = re.sub(r'[^\w\s]', ' ', text)

        # Normalize multiple spaces to single space
        text = ' '.join(text.split())

        return text

    def load_and_preprocess_data(self, unrelated_csv_path, travel_txt_path, target_samples_per_class=10000):
        """
        Load and preprocess data from two different files for binary classification
        Balances the dataset by sampling equal numbers from each class

        Args:
            unrelated_csv_path (str): Path to CSV file with non-booking data
            travel_txt_path (str): Path to TXT file with booking confirmation data
            target_samples_per_class (int): Target number of samples per class (default: 10000)

        Returns:
            tuple: Processed features and binary labels (balanced)
        """
        logger.info(f"Loading and balancing data from files:")
        logger.info(f"Non-booking data: {unrelated_csv_path}")
        logger.info(f"Booking data: {travel_txt_path}")
        logger.info(f"Target samples per class: {target_samples_per_class}")

        # Load non-booking data (label 0)
        logger.info("Loading non-booking data...")
        unrelated_df = pd.read_csv(unrelated_csv_path)
        logger.info(f"CSV file columns: {unrelated_df.columns.tolist()}")

        # Check CSV file column names, use actual column names
        if 'contect' in unrelated_df.columns:
            text_column = 'contect'
        elif 'content' in unrelated_df.columns:
            text_column = 'content'
        else:
            # If expected columns not found, use first column
            text_column = unrelated_df.columns[0]
            logger.warning(f"Using first column as text column: {text_column}")

        # Create non-booking DataFrame
        unrelated_data = pd.DataFrame({
            'text': unrelated_df[text_column],
            'label': 0  # Non-booking label is 0
        })

        logger.info(f"Original non-booking records: {len(unrelated_data)}")

        # Load booking data (label 1)
        logger.info("Loading booking data...")
        with open(travel_txt_path, 'r', encoding='utf-8') as f:
            travel_texts = f.readlines()

        # Create booking DataFrame
        travel_data = pd.DataFrame({
            'text': [text.strip() for text in travel_texts if text.strip()],
            'label': 1  # Booking label is 1
        })

        logger.info(f"Original booking records: {len(travel_data)}")

        # Preprocess text content for both datasets
        logger.info("Preprocessing text content...")
        unrelated_data['text'] = unrelated_data['text'].apply(self.preprocess_text)
        travel_data['text'] = travel_data['text'].apply(self.preprocess_text)

        # Remove empty content
        unrelated_data = unrelated_data[unrelated_data['text'].str.len() > 0]
        travel_data = travel_data[travel_data['text'].str.len() > 0]

        logger.info(f"After preprocessing:")
        logger.info(f"Non-booking records: {len(unrelated_data)}")
        logger.info(f"Booking records: {len(travel_data)}")

        # Balance the dataset by sampling
        logger.info(f"Balancing dataset to {target_samples_per_class} samples per class...")

        # Sample non-booking data
        if len(unrelated_data) > target_samples_per_class:
            unrelated_balanced = unrelated_data.sample(n=target_samples_per_class, random_state=42)
            logger.info(f"✅ Downsampled non-booking data from {len(unrelated_data)} to {len(unrelated_balanced)}")
        else:
            unrelated_balanced = unrelated_data
            logger.warning(f"⚠️ Non-booking data has only {len(unrelated_data)} samples, less than target {target_samples_per_class}")

        # Sample booking data
        if len(travel_data) > target_samples_per_class:
            travel_balanced = travel_data.sample(n=target_samples_per_class, random_state=42)
            logger.info(f"✅ Downsampled booking data from {len(travel_data)} to {len(travel_balanced)}")
        else:
            travel_balanced = travel_data
            logger.warning(f"⚠️ Booking data has only {len(travel_data)} samples, less than target {target_samples_per_class}")

        # Combine balanced datasets
        combined_df = pd.concat([unrelated_balanced, travel_balanced], ignore_index=True)

        # Shuffle the combined dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"✅ Balanced dataset created with {len(combined_df)} total records")

        # Extract features and binary labels
        X = combined_df['text'].values
        y = combined_df['label'].values

        logger.info(f"Final balanced data distribution:")
        logger.info(f"Non-booking (0): {np.sum(y == 0)} samples")
        logger.info(f"Booking (1): {np.sum(y == 1)} samples")
        logger.info(f"Balance ratio: {np.sum(y == 1) / np.sum(y == 0):.2f}")

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
        Build lightweight CNN model optimized for Android deployment
        Uses 1D CNN instead of LSTM for better speed and smaller model size
        """
        logger.info("Building lightweight CNN binary classification model for Android...")

        # Input layer
        inputs = Input(shape=(self.max_length,), name='input')

        # Embedding layer (smaller dimension for efficiency)
        embedding = Embedding(
            input_dim=self.max_features,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            mask_zero=False,  # Disable masking for CNN to improve speed
            name='embedding'
        )(inputs)

        # Multi-scale CNN layers for different n-gram features
        # This captures patterns at different scales efficiently
        conv_layers = []

        # 3-gram features
        conv1 = Conv1D(64, 3, activation='relu', padding='same', name='conv1d_3gram')(embedding)
        conv1 = BatchNormalization(name='bn_3gram')(conv1)
        pool1 = GlobalMaxPooling1D(name='global_max_pool_3gram')(conv1)
        conv_layers.append(pool1)

        # 4-gram features
        conv2 = Conv1D(64, 4, activation='relu', padding='same', name='conv1d_4gram')(embedding)
        conv2 = BatchNormalization(name='bn_4gram')(conv2)
        pool2 = GlobalMaxPooling1D(name='global_max_pool_4gram')(conv2)
        conv_layers.append(pool2)

        # 5-gram features
        conv3 = Conv1D(64, 5, activation='relu', padding='same', name='conv1d_5gram')(embedding)
        conv3 = BatchNormalization(name='bn_5gram')(conv3)
        pool3 = GlobalMaxPooling1D(name='global_max_pool_5gram')(conv3)
        conv_layers.append(pool3)

        # Concatenate all CNN features
        concatenated = Concatenate(name='concatenate_features')(conv_layers)

        # Dense layers (smaller for mobile efficiency)
        dense1 = Dense(32, activation='relu', name='dense_1')(concatenated)
        dropout1 = Dropout(0.3, name='dropout_1')(dense1)  # Reduced dropout for smaller model

        dense2 = Dense(16, activation='relu', name='dense_2')(dropout1)
        dropout2 = Dropout(0.2, name='dropout_2')(dense2)

        # Binary output layer
        outputs = Dense(1, activation='sigmoid', name='output')(dropout2)

        # Create the model
        self.model = Model(inputs=inputs, outputs=outputs, name='travel_cnn_classifier')

        # Compile model for binary classification
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        # Print model summary
        self.model.summary()

        # Calculate model size
        model_size = self.model.count_params()
        logger.info(f"✅ Lightweight CNN model built successfully")
        logger.info(f"📊 Total parameters: {model_size:,}")
        logger.info(f"🚀 Optimized for Android deployment")

    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
        """
        Train the binary classification model with mobile-optimized settings

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Number of training epochs (reduced for faster training)
            batch_size (int): Batch size (increased for efficiency)
        """
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}...")

        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,  # Reduced patience for faster convergence
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,  # Reduced patience
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
        Comprehensive binary classification model evaluation

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
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        # Calculate comprehensive metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )

        # Print performance summary
        print("\n" + "="*60)
        print("TRAVEL TEXT BINARY CLASSIFICATION PERFORMANCE")
        print("="*60)
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
        plt.title('Travel Text Binary Classification Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('travel_confusion_matrix.png', dpi=300)
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
        plt.savefig('travel_training_history.png', dpi=300)
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

    def convert_to_tflite(self, model_path='travel_classifier.tflite', quantization='dynamic'):
        """
        Convert model to TensorFlow Lite for Android deployment
        Optimized for mobile with aggressive quantization

        Args:
            model_path (str): Path to save TFLite model
            quantization (str): Quantization type - 'none', 'dynamic', 'float16'
        """
        logger.info("Converting travel classification model to TensorFlow Lite...")

        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # CNN is more compatible with TFLite than LSTM
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

        # Apply aggressive quantization for mobile
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

            logger.info(f"✅ Travel CNN TFLite model saved to {model_path}")
            logger.info(f"📁 Model size: {model_size:.2f} MB")
            logger.info("📱 Model ready for Android deployment")
            logger.info("🚀 CNN architecture provides faster inference than LSTM")

            return model_path

        except Exception as e:
            logger.error(f"TFLite conversion failed: {e}")
            raise

    def save_model_components(self, model_dir='travel_classifier_artifacts'):
        """
        Save all model components for deployment

        Args:
            model_dir (str): Directory to save model artifacts
        """
        os.makedirs(model_dir, exist_ok=True)

        # Save Keras model
        self.model.save(os.path.join(model_dir, 'travel_classifier_model.h5'))

        # Save tokenizer
        with open(os.path.join(model_dir, 'tokenizer.pickle'), 'wb') as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save model configuration
        config = {
            'max_features': self.max_features,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
            'model_type': 'binary_classification_cnn',
            'architecture': 'Multi-scale CNN with Global Max Pooling',
            'classes': ['non_booking', 'booking'],
            'output_description': 'Returns 0 for non-booking, 1 for booking-related text',
            'optimized_for': 'Android deployment',
            'features': [
                'Lightweight CNN architecture',
                'Multi-scale n-gram features (3,4,5)',
                'Global max pooling for efficiency',
                'Reduced parameters for mobile',
                'Fast inference speed'
            ]
        }

        import json
        with open(os.path.join(model_dir, 'travel_config.json'), 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Travel classification model components saved to {model_dir}")


def main():
    """
    Main function for travel text binary classification
    """
    # Configuration - optimized for Android deployment
    UNRELATED_CSV_PATH = '/Users/jinze/agent_classification/unrelated_dataset/unrelated_text_part_1.csv'
    TRAVEL_TXT_PATH = '/Users/jinze/agent_classification/travel_confirmations.txt'
    EPOCHS = 20  # Reduced for faster training
    BATCH_SIZE = 64  # Increased for efficiency
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2

    # Data balancing configuration
    TARGET_SAMPLES_PER_CLASS = 10000  # Balance dataset: 10k booking + 10k non-booking

    try:
        # Initialize travel text classifier with mobile-optimized parameters
        classifier = TravelTextClassifier(
            max_features=8000,  # Reduced vocabulary for smaller model
            max_length=80,      # Shorter sequences for faster processing
            embedding_dim=64    # Smaller embeddings for efficiency
        )

        # Load and preprocess data with balanced sampling
        X, y = classifier.load_and_preprocess_data(UNRELATED_CSV_PATH, TRAVEL_TXT_PATH, TARGET_SAMPLES_PER_CLASS)

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=TEST_SIZE + VAL_SIZE, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=TEST_SIZE/(TEST_SIZE + VAL_SIZE),
            random_state=42, stratify=y_temp
        )

        # Prepare sequences
        X_train_seq, X_val_seq, X_test_seq = classifier.prepare_sequences(X_train, X_val, X_test)

        # Build model
        classifier.build_model()

        # Train model
        classifier.train_model(X_train_seq, y_train, X_val_seq, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)

        # Evaluate model
        metrics = classifier.evaluate_model(X_test_seq, y_test)

        # Plot training history
        classifier.plot_training_history()

        # Test predictions on sample data
        sample_texts = [
            "Flight booking confirmation from New York to Los Angeles AA1234",
            "Hotel reservation at Grand Hotel for 3 nights check-in tomorrow",
            "Car rental Toyota Camry pickup at airport tomorrow",
            "Nice weather today, going for a walk",
            "Meeting scheduled for 3pm in conference room",
            "Restaurant reservation for dinner tonight",
            "FLIGHT PNR: XY789Z | PASSENGER: Emily Chen | DELTA DL2234",
            "BOOKING REF: HTL456789 | GUEST: Michael Thompson | MARRIOTT",
            "CAR RENTAL #CR89012 | RENTER: David Wilson | ENTERPRISE",
            "The Bible teaches that God is perfectly righteous"
        ]

        predictions, probabilities = classifier.predict(sample_texts)

        print("\nTRAVEL TEXT BINARY CLASSIFICATION PREDICTIONS:")
        print("="*80)
        for text, pred, prob in zip(sample_texts, predictions, probabilities):
            result = "✅ BOOKING" if pred == 1 else "❌ NON-BOOKING"
            print(f"Text: {text}")
            print(f"Prediction: {pred} ({result})")
            print(f"Confidence: {prob:.4f}")
            print("-" * 80)

        # Convert to TFLite with aggressive optimization
        tflite_path = classifier.convert_to_tflite('travel_classifier.tflite', quantization='dynamic')

        # Save model components
        classifier.save_model_components('travel_classifier_artifacts')

        print(f"\n🎉 TRAVEL TEXT BINARY CLASSIFICATION COMPLETED!")
        print(f"📊 Model accuracy: {metrics['accuracy']:.4f}")
        print(f"📱 TFLite model: {tflite_path}")
        print(f"💾 Model artifacts: travel_classifier_artifacts/")
        print(f"🎯 Ready for Android deployment!")

        print("\n" + "="*80)
        print("OPTIMIZED MOBILE MODEL SUMMARY")
        print("="*80)
        print("📋 Task: Binary classification (booking vs non-booking)")
        print("🏗️ Architecture: Lightweight Multi-scale CNN")
        print("🎯 Output: 0 (non-booking) or 1 (booking)")
        print("📱 Target: Flight, car rental, hotel bookings")
        print("⚖️ Data: Balanced dataset (10k samples per class)")
        print("⚡ Features: Fast inference, small model size, high accuracy")
        print("🚀 Optimizations: Reduced params, CNN vs LSTM, efficient pooling, balanced training")

    except Exception as e:
        logger.error(f"Error in travel classification pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # System check
    print("=" * 80)
    print("TRAVEL TEXT BINARY CLASSIFIER - ANDROID OPTIMIZED")
    print("=" * 80)

    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print("Task: Binary classification (booking vs non-booking)")
    print("Output: 0 (non-booking) or 1 (booking)")
    print("Architecture: Lightweight CNN for mobile deployment")

    print("=" * 80)
    print("STARTING MOBILE-OPTIMIZED TRAINING PIPELINE")
    print("=" * 80)

    main()

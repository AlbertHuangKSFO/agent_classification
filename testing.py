#!/usr/bin/env python3
"""
Travel Text Binary Classifier Testing Script
Tests the trained model for booking vs non-booking text classification

Author: AI Assistant
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import json
import time
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TravelClassifierTester:
    """
    Test suite for the trained travel text binary classifier
    """

    def __init__(self, model_dir='travel_classifier_artifacts'):
        """
        Initialize the tester with trained model artifacts

        Args:
            model_dir (str): Directory containing model artifacts
        """
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.config = None
        self.max_length = None

        # Load model components
        self.load_model_components()

    def load_model_components(self):
        """
        Load all necessary model components for testing
        """
        try:
            logger.info(f"Loading model components from {self.model_dir}")

            # Load model configuration
            config_path = os.path.join(self.model_dir, 'travel_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                self.max_length = self.config.get('max_length', 80)
                logger.info(f"✅ Configuration loaded: max_length={self.max_length}")
            else:
                logger.warning("Config file not found, using default max_length=80")
                self.max_length = 80

            # Load trained model
            model_path = os.path.join(self.model_dir, 'travel_classifier_model.h5')
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                logger.info("✅ Model loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load tokenizer
            tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pickle')
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                logger.info("✅ Tokenizer loaded successfully")
            else:
                raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

            logger.info("🎉 All model components loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            raise

    def preprocess_text(self, text):
        """
        Preprocess text using the same method as training

        Args:
            text (str): Raw text to preprocess

        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text is None:
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Replace common separators with spaces
        text = re.sub(r'[_\-/\\|:,]', ' ', text)

        # Keep alphanumeric characters and spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Normalize multiple spaces to single space
        text = ' '.join(text.split())

        return text

    def predict_single(self, text, threshold=0.5):
        """
        Predict a single text sample

        Args:
            text (str): Text to classify
            threshold (float): Decision threshold

        Returns:
            dict: Prediction results
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please check model artifacts.")

        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Convert to sequence and pad
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post')

        # Make prediction
        probability = self.model.predict(padded_sequence, verbose=0)[0][0]
        prediction = 1 if probability > threshold else 0

        return {
            'original_text': text,
            'processed_text': processed_text,
            'prediction': prediction,
            'probability': float(probability),
            'confidence': float(probability if prediction == 1 else 1 - probability),
            'label': 'BOOKING' if prediction == 1 else 'NON-BOOKING'
        }

    def predict_batch(self, texts, threshold=0.5):
        """
        Predict multiple text samples

        Args:
            texts (list): List of texts to classify
            threshold (float): Decision threshold

        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            result = self.predict_single(text, threshold)
            results.append(result)
        return results

    def test_sample_data(self):
        """
        Test the model with predefined sample data
        """
        logger.info("Testing model with sample data...")

        # Sample test cases
        test_cases = [
            # Booking examples (should predict 1)
            "Your Flight Receipt - ALbert H 10MAY25 Delta Air Lines DeltaAirLines@t.delta.com Saturday, May 10 2025 at 12:30 To: abcdgsdd@outlook.comPassenger InfoName: ALBERT H kyMiles #*******779FLIGHT    SEATDELTA 637    30F DELTA 667    40A isit delta.com or download the Fly Delta app to view, select or change your seat.f you purchased a Delta Comfort+™ seat or a Trip Extra, please visit My Trips to access a receipt of your purchase.Wed, 07MAY    DEPART    ARRIVDELTA 637 Main (L)    SAN FRANCISCO 10:45PM    NYC-KENNEDY 07:30AM **Thu 08MAYSat, 10MAY    DEPART    ARRIVEELTA 667 Main (Q)    NYC-KENNEDY 07:40PM    SAN FRANCISCO 11:29PM"
        ]

        # Expected labels (1 for booking, 0 for non-booking)
        expected_labels = [1]

        # Make predictions
        results = self.predict_batch(test_cases)

        # Display results
        print("\n" + "="*100)
        print("SAMPLE DATA TEST RESULTS")
        print("="*100)

        correct_predictions = 0
        for i, (result, expected) in enumerate(zip(results, expected_labels)):
            is_correct = result['prediction'] == expected
            correct_predictions += is_correct

            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"\nTest {i+1}: {status}")
            print(f"Text: {result['original_text'][:80]}...")
            print(f"Expected: {expected} ({'BOOKING' if expected == 1 else 'NON-BOOKING'})")
            print(f"Predicted: {result['prediction']} ({result['label']})")
            print(f"Confidence: {result['confidence']:.4f}")
            print("-" * 100)

        accuracy = correct_predictions / len(test_cases)
        print(f"\n📊 SAMPLE TEST ACCURACY: {accuracy:.4f} ({correct_predictions}/{len(test_cases)})")

        return results, expected_labels

    def test_from_file(self, test_file_path, expected_labels=None):
        """
        Test the model with data from a file

        Args:
            test_file_path (str): Path to test file (CSV or TXT)
            expected_labels (list): Optional list of expected labels

        Returns:
            list: Prediction results
        """
        logger.info(f"Testing model with data from: {test_file_path}")

        # Load test data
        if test_file_path.endswith('.csv'):
            df = pd.read_csv(test_file_path)
            # Assume first column is text
            texts = df.iloc[:, 0].tolist()
        elif test_file_path.endswith('.txt'):
            with open(test_file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
        else:
            raise ValueError("Unsupported file format. Use CSV or TXT files.")

        # Make predictions
        results = self.predict_batch(texts)

        # Calculate accuracy if labels provided
        if expected_labels:
            correct = sum(1 for r, e in zip(results, expected_labels) if r['prediction'] == e)
            accuracy = correct / len(results)
            print(f"📊 File Test Accuracy: {accuracy:.4f} ({correct}/{len(results)})")

        return results

    def interactive_test(self):
        """
        Interactive testing mode - user can input text manually
        """
        print("\n" + "="*60)
        print("INTERACTIVE TESTING MODE")
        print("="*60)
        print("Enter text to classify (type 'quit' to exit):")

        while True:
            try:
                user_input = input("\n📝 Enter text: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Exiting interactive mode...")
                    break

                if not user_input:
                    continue

                # Make prediction
                result = self.predict_single(user_input)

                # Display result
                print(f"\n🔍 PREDICTION RESULT:")
                print(f"📊 Classification: {result['label']}")
                print(f"🎯 Prediction: {result['prediction']}")
                print(f"📈 Probability: {result['probability']:.4f}")
                print(f"💯 Confidence: {result['confidence']:.4f}")

            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def benchmark_performance(self, num_samples=1000):
        """
        Benchmark model inference performance

        Args:
            num_samples (int): Number of samples to test
        """
        logger.info(f"Benchmarking model performance with {num_samples} samples...")

        # Generate sample texts
        sample_texts = [
            "Flight booking confirmation AA1234 from LAX to JFK",
            "Hotel reservation at Marriott downtown for 3 nights",
            "Car rental Enterprise pickup tomorrow at airport",
            "Weather is nice today for outdoor activities",
            "Meeting scheduled for next week in conference room"
        ] * (num_samples // 5 + 1)

        sample_texts = sample_texts[:num_samples]

        # Measure inference time
        start_time = time.time()
        results = self.predict_batch(sample_texts)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_sample = total_time / num_samples
        samples_per_second = num_samples / total_time

        print(f"\n⚡ PERFORMANCE BENCHMARK:")
        print(f"📊 Total samples: {num_samples}")
        print(f"⏱️ Total time: {total_time:.4f} seconds")
        print(f"📈 Average time per sample: {avg_time_per_sample*1000:.2f} ms")
        print(f"🚀 Samples per second: {samples_per_second:.2f}")

        return {
            'total_time': total_time,
            'avg_time_per_sample': avg_time_per_sample,
            'samples_per_second': samples_per_second
        }

    def analyze_model_info(self):
        """
        Display detailed model information
        """
        print("\n" + "="*80)
        print("MODEL INFORMATION")
        print("="*80)

        if self.config:
            print("📋 Configuration:")
            for key, value in self.config.items():
                print(f"   {key}: {value}")

        if self.model:
            print(f"\n🏗️ Model Architecture:")
            self.model.summary()

            print(f"\n📊 Model Statistics:")
            total_params = self.model.count_params()
            print(f"   Total parameters: {total_params:,}")

            # Estimate model size
            param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
            print(f"   Estimated size: {param_size_mb:.2f} MB")

        if self.tokenizer:
            vocab_size = len(self.tokenizer.word_index) + 1
            print(f"\n📝 Tokenizer Info:")
            print(f"   Vocabulary size: {vocab_size:,}")
            print(f"   Max sequence length: {self.max_length}")


def main():
    """
    Main testing function with multiple testing options
    """
    print("=" * 80)
    print("TRAVEL TEXT BINARY CLASSIFIER - TESTING SUITE")
    print("=" * 80)

    try:
        # Initialize tester
        tester = TravelClassifierTester()

        # Display model information
        tester.analyze_model_info()

        # Test with sample data
        tester.test_sample_data()

        # Benchmark performance
        tester.benchmark_performance(num_samples=100)

        # Interactive testing prompt
        while True:
            print("\n" + "="*60)
            print("TESTING OPTIONS:")
            print("1. Interactive testing (manual text input)")
            print("2. Test with sample data again")
            print("3. Benchmark performance")
            print("4. Test from file")
            print("5. Exit")

            choice = input("\nSelect option (1-5): ").strip()

            if choice == '1':
                tester.interactive_test()
            elif choice == '2':
                tester.test_sample_data()
            elif choice == '3':
                num_samples = input("Enter number of samples (default 1000): ").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else 1000
                tester.benchmark_performance(num_samples)
            elif choice == '4':
                file_path = input("Enter test file path: ").strip()
                if os.path.exists(file_path):
                    tester.test_from_file(file_path)
                else:
                    print("File not found!")
            elif choice == '5':
                print("Exiting testing suite...")
                break
            else:
                print("Invalid option. Please try again.")

    except Exception as e:
        logger.error(f"Error in testing suite: {e}")
        raise


if __name__ == "__main__":
    # Quick test function for single predictions
    def quick_test(text):
        """Quick test function for single text prediction"""
        tester = TravelClassifierTester()
        result = tester.predict_single(text)

        print(f"\n📝 Text: {text}")
        print(f"🔍 Result: {result['label']}")
        print(f"📊 Confidence: {result['confidence']:.4f}")

        return result

    # Run main testing suite
    if len(sys.argv) > 1:
        # Command line text prediction
        test_text = " ".join(sys.argv[1:])
        quick_test(test_text)
    else:
        # Full testing suite
        main()

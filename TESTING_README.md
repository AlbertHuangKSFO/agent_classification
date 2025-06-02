# Travel Text Classifier Testing Suite

## 📋 Overview

The `testing.py` script provides a comprehensive testing suite for the trained travel text binary classifier. It allows you to test the model's performance, benchmark inference speed, and interactively classify new text samples.

## 🚀 Quick Start

### 1. Basic Usage

```bash
# Run the full testing suite
python testing.py

# Quick test a single text from command line
python testing.py "Flight booking confirmation AA1234 from LAX to JFK"
```

### 2. Requirements

- Trained model artifacts in `travel_classifier_artifacts/` directory
- Required files:
  - `travel_classifier_model.h5` (trained Keras model)
  - `tokenizer.pickle` (fitted tokenizer)
  - `travel_config.json` (model configuration)

## 🔧 Features

### 1. **Model Information Display**

- Shows model architecture and parameters
- Displays configuration settings
- Reports vocabulary size and model statistics

### 2. **Sample Data Testing**

- Tests with predefined booking and non-booking examples
- Calculates accuracy on known test cases
- Provides detailed prediction results

### 3. **Interactive Testing**

- Manual text input for real-time testing
- Shows prediction, probability, and confidence
- Type 'quit' to exit interactive mode

### 4. **File-based Testing**

- Test with CSV or TXT files
- Supports batch prediction
- Optional accuracy calculation with known labels

### 5. **Performance Benchmarking**

- Measures inference speed
- Reports samples per second
- Useful for mobile deployment planning

## 📊 Testing Options Menu

When you run `python testing.py`, you'll see:

```
TESTING OPTIONS:
1. Interactive testing (manual text input)
2. Test with sample data again
3. Benchmark performance
4. Test from file
5. Exit
```

## 🎯 Example Usage

### Command Line Testing

```bash
# Test a booking text
python testing.py "FLIGHT PNR: XY123 | PASSENGER: John Doe | DELTA DL456"

# Test a non-booking text
python testing.py "The weather is nice today for a walk"
```

### File Testing

```bash
# Place your test data in a file
echo "Flight booking AA123 LAX to JFK" > my_test.txt
echo "Meeting at 3pm conference room" >> my_test.txt

# Run testing.py and select option 4, then enter: my_test.txt
python testing.py
```

### Interactive Testing

```bash
python testing.py
# Select option 1
# Enter text samples one by one
# Type 'quit' when done
```

## 📈 Sample Output

```
🔍 PREDICTION RESULT:
📊 Classification: BOOKING
🎯 Prediction: 1
📈 Probability: 0.8942
💯 Confidence: 0.8942
```

## 🧪 Test File Format

### TXT Format

```
Flight booking confirmation AA1234
Hotel reservation at Marriott downtown
Weather is nice today
Meeting at 3pm conference room
```

### CSV Format

```
text,label
"Flight booking AA123",1
"Weather report sunny",0
"Hotel check-in today",1
```

## ⚡ Performance Benchmarking

The benchmark feature tests inference speed:

```
⚡ PERFORMANCE BENCHMARK:
📊 Total samples: 1000
⏱️ Total time: 2.1234 seconds
📈 Average time per sample: 2.12 ms
🚀 Samples per second: 470.71
```

## 🔍 Sample Test Cases

The script includes predefined test cases:

**Booking Examples (should predict 1):**

- Flight confirmations with PNR codes
- Hotel reservations with check-in dates
- Car rental confirmations
- Booking references and confirmations

**Non-Booking Examples (should predict 0):**

- Weather reports
- Meeting schedules
- General text content
- Technical documentation

## 📱 Mobile Deployment Testing

Use the benchmark feature to ensure your model meets mobile performance requirements:

```python
# Test with different sample sizes
tester.benchmark_performance(num_samples=100)   # Quick test
tester.benchmark_performance(num_samples=1000)  # Standard test
tester.benchmark_performance(num_samples=5000)  # Stress test
```

## 🛠️ Troubleshooting

### Model Not Found

```
Error: Model file not found: travel_classifier_artifacts/travel_classifier_model.h5
```

**Solution:** Run `classification.py` first to train and save the model.

### Tokenizer Not Found

```
Error: Tokenizer file not found: travel_classifier_artifacts/tokenizer.pickle
```

**Solution:** Ensure the complete training pipeline completed successfully.

### Low Accuracy

If sample test accuracy is low:

1. Check if model was trained on balanced data
2. Verify preprocessing is consistent
3. Consider retraining with more data

## 📚 Class Methods

### `TravelClassifierTester`

- `predict_single(text, threshold=0.5)` - Predict single text
- `predict_batch(texts, threshold=0.5)` - Predict multiple texts
- `test_sample_data()` - Test with predefined samples
- `test_from_file(file_path, labels=None)` - Test from file
- `interactive_test()` - Interactive testing mode
- `benchmark_performance(num_samples=1000)` - Performance benchmark
- `analyze_model_info()` - Display model information

## 🎯 Best Practices

1. **Test Regularly:** Run tests after each model update
2. **Use Diverse Samples:** Include edge cases in your test data
3. **Monitor Performance:** Benchmark on target hardware
4. **Validate Preprocessing:** Ensure consistent text cleaning
5. **Check Confidence:** Low confidence predictions may need review

## 🔗 Integration

You can also import and use the tester in your own scripts:

```python
from testing import TravelClassifierTester

# Initialize tester
tester = TravelClassifierTester()

# Make a prediction
result = tester.predict_single("Flight booking AA123")
print(f"Prediction: {result['label']} (confidence: {result['confidence']:.3f})")
```

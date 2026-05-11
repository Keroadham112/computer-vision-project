# Class Label Reversal Debugging Guide

## The Problem

When using `ImageDataGenerator.flow_from_directory()` with TensorFlow:

1. **Classes are sorted alphabetically** by folder name
2. The binary sigmoid output may be interpreted incorrectly
3. The mapping between model output and class labels can be reversed

## How ImageDataGenerator.flow_from_directory() Works

Example directory structure:
```
train/
  fractured/      ← "fractured" comes first alphabetically
    img1.jpg
    img2.jpg
  not fractured/  ← "not fractured" comes second
    img1.jpg
```

**Class Indices:**
```python
class_indices = {
  'fractured': 0,      # Alphabetically first
  'not fractured': 1   # Alphabetically second
}
```

## Sigmoid Output Interpretation

For a **binary classification model with sigmoid output**:

```python
output_value = 0.92  # Model prediction (0-1)

# Standard interpretation:
if output_value > 0.5:
    predicted_class = 1  # "not fractured"
else:
    predicted_class = 0  # "fractured"
```

## Possible Scenarios

### Scenario 1: Standard Order (CORRECT)
- Folder: `fractured/` → Class 0
- Folder: `not fractured/` → Class 1
- Sigmoid > 0.5 → Class 1 → "Not Fractured" ✓

### Scenario 2: Reversed Order (NEEDS FIX)
- Folder: `not fractured/` → Class 0 (comes first alphabetically)
- Folder: `fractured/` → Class 1
- Sigmoid > 0.5 → Class 1 → "Fractured" ✓

## How to Determine Your Case

### Step 1: Check Directory Names

```python
import os
train_dir = 'archive (1)/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train'
classes = sorted(os.listdir(train_dir))
for idx, cls in enumerate(classes):
    print(f"Class {idx}: {cls}")
```

### Step 2: Check Model Output

```python
import tensorflow as tf
model = tf.keras.models.load_model('fracture_model.h5')

# Create dummy input
import numpy as np
x = np.random.rand(1, 224, 224, 3).astype('float32')
pred = model.predict(x)
print(f"Output shape: {pred.shape}")
print(f"Output value: {pred[0]}")
```

### Step 3: Run Inspection Script

```bash
python backend/inspect_model.py
```

## Common Mistakes

### ❌ Mistake 1: Assuming Standard Order

If folders are not alphabetically as expected, your mapping will be wrong:

```python
# Wrong assumption:
if pred > 0.5:
    label = "Fractured"  # But class 1 might actually be "not fractured"!
```

### ❌ Mistake 2: Reversed Sigmoid Logic

If the model outputs probability for class 0:

```python
# This might be wrong:
if pred > 0.5:
    label = "Fractured"

# Should be:
if pred < 0.5:  # Inverted!
    label = "Fractured"
```

### ❌ Mistake 3: Confidence Mismatch

Using the wrong confidence formula for the predicted class.

## Solution: Universal Approach

**Ask these questions about YOUR model:**

1. **What are the folder names in alphabetical order?**
   - Use: `sorted(os.listdir('train/'))`

2. **Which folder corresponds to which class index?**
   - Class 0 = first folder (alphabetically)
   - Class 1 = second folder (alphabetically)

3. **What does sigmoid output represent?**
   - Typically: probability of the POSITIVE class (class 1)

4. **Is "Fractured" class 0 or class 1 in your training?**
   - If folders are named `fractured/` and `not fractured/`
   - Then "fractured" = class 0, "not fractured" = class 1

## Corrected Backend Logic

### Case A: "fractured" comes BEFORE "not fractured" (standard)

```python
# Class 0 = "fractured", Class 1 = "not fractured"
# Sigmoid output > 0.5 → Class 1 → "Not Fractured"

if prob > 0.5:
    label = "Not Fractured"
    confidence = round(prob * 100, 1)
else:
    label = "Fractured"
    confidence = round((1 - prob) * 100, 1)
```

### Case B: "not fractured" comes BEFORE "fractured" (reversed)

```python
# Class 0 = "not fractured", Class 1 = "fractured"
# Sigmoid output > 0.5 → Class 1 → "Fractured"

if prob > 0.5:
    label = "Fractured"
    confidence = round(prob * 100, 1)
else:
    label = "Not Fractured"
    confidence = round((1 - prob) * 100, 1)
```

### Case C: Using Softmax (Two Outputs)

```python
# Model outputs [prob_class0, prob_class1]
class_probs = model.predict(x)[0]  # [prob_class0, prob_class1]

predicted_class = np.argmax(class_probs)
max_confidence = round(class_probs[predicted_class] * 100, 1)

# Map class index to label:
# If "fractured" is class 0: [0→"Fractured", 1→"Not Fractured"]
# If "not fractured" is class 0: [0→"Not Fractured", 1→"Fractured"]

label = class_labels[predicted_class]
```

## Steps to Fix

1. **Run the inspection script:**
   ```bash
   python backend/inspect_model.py
   ```

2. **Check the output for:**
   - Class 0 folder name
   - Class 1 folder name
   - Model output shape

3. **Update `backend/main.py` with correct logic**

4. **Test with curl:**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -F "file=@test_image.jpg"
   ```

5. **Verify result matches visual inspection of the image**

## Quick Fix Template

Replace the prediction logic in `backend/main.py` with:

```python
# Get raw prediction
preds = model.predict(x, verbose=0)

if preds.shape[-1] == 1:
    # Sigmoid output
    prob = float(preds[0][0])
    
    # CHOOSE ONE BASED ON YOUR INSPECTION:
    
    # Option A: Standard order
    if prob > 0.5:
        label = 'Not Fractured'
        confidence = round(prob * 100, 1)
    else:
        label = 'Fractured'
        confidence = round((1 - prob) * 100, 1)
    
    # Option B: Reversed order
    # if prob > 0.5:
    #     label = 'Fractured'
    #     confidence = round(prob * 100, 1)
    # else:
    #     label = 'Not Fractured'
    #     confidence = round((1 - prob) * 100, 1)

else:
    # Softmax output with 2 classes
    class_probs = preds[0]  # [prob_class0, prob_class1]
    predicted_class = int(np.argmax(class_probs))
    confidence = round(class_probs[predicted_class] * 100, 1)
    
    # Map to labels (adjust based on your class order)
    labels = ['Fractured', 'Not Fractured']  # Or reversed
    label = labels[predicted_class]
```

---

**Next:** Run `python backend/inspect_model.py` and share the output to determine your exact case!

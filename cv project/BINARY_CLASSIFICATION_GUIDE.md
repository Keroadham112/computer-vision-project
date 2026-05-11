# Binary Classification Label Reversal — Complete Debugging Guide

## TL;DR — Quick Fix Steps

1. **Run inspection script:**
   ```bash
   python backend/inspect_model.py
   ```

2. **Check the output** for which class folder comes first alphabetically

3. **Choose the right backend:**
   - Option A: if `fractured` folder comes BEFORE `not fractured`
   - Option B: if `not fractured` folder comes BEFORE `fractured`

4. **Copy the correct code from `backend/main_alternatives.py`**

5. **Restart backend and test**

---

## Understanding the Problem

### How Binary Classification Works

Your training setup:
```
train/
  fractured/           # Images of X-rays with fractures
    img1.jpg
    ...
  not fractured/       # Images of X-rays without fractures
    img2.jpg
    ...
```

### ImageDataGenerator.flow_from_directory() Class Mapping

**Key Point:** Classes are assigned indices in **alphabetical order** of folder names:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    'train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

print(train_generator.class_indices)
# Output depends on folder names!
```

### The Alphabetical Sorting Issue

**Scenario A: "fractured" comes first alphabetically**
```
'fractured' < 'not fractured'  ✓

class_indices = {
    'fractured': 0,      # ← Class 0
    'not fractured': 1   # ← Class 1
}
```

**Scenario B: "not fractured" comes first alphabetically**
```
'not fractured' < 'fractured'  ✓

# Wait, that's wrong! Let me check again...
# Actually comparing first characters: 'n' vs 'f'
# 'f' < 'n' in ASCII, so "fractured" still comes first!
```

Actually, let me verify this:
```python
>>> sorted(['fractured', 'not fractured'])
['fractured', 'not fractured']

>>> sorted(['not_fractured', 'fractured'])
['fractured', 'not_fractured']

>>> sorted(['not fractured', 'fractured'])
['fractured', 'not fractured']
```

**So "fractured" ALWAYS comes first alphabetically!**

Unless your folder is named something like:
- `NOT_fractured/` or
- `normal/` (instead of "not fractured")

---

## The Sigmoid Output Interpretation

### Binary Classification with Sigmoid

For a model trained with:
- `class_mode='binary'` 
- Final layer: `Dense(1, activation='sigmoid')`

The output is a **single probability value** (0-1):

```python
model.predict(image) → [[0.92]]  # Single value in array

prob = 0.92
```

### What Does This Probability Mean?

In TensorFlow's binary classification:
- **Prob > 0.5** → Predicted as **Class 1**
- **Prob < 0.5** → Predicted as **Class 0**

So if:
- **Class 0 = "fractured"** and **Class 1 = "not fractured"**
- **Prob = 0.92** (> 0.5)
- **Prediction = Class 1 = "not fractured"** ✓

But the original request's code was:
```python
if prediction > 0.5:
    result = "Fractured"  # ❌ Wrong! This is Class 1, not Class 0
    confidence = prediction * 100
```

This is **backwards**!

---

## Correct Implementation

### Option A: Standard Setup (LIKELY YOUR CASE)

**Class mapping:**
```python
class_indices = {
    'fractured': 0,
    'not fractured': 1
}
```

**Correct backend logic:**
```python
prob = model.predict(image)[0][0]  # e.g., 0.92

if prob > 0.5:
    label = 'Not Fractured'      # ✓ Sigmoid > 0.5 → Class 1
    confidence = prob * 100       # 92%
else:
    label = 'Fractured'           # ✓ Sigmoid < 0.5 → Class 0
    confidence = (1 - prob) * 100 # 92% (if prob was 0.08)
```

### Option B: Reversed Folder Names

**Only if your folders are named differently**, like:
```
train/
  normal/         # Not fractured
  broken/         # Fractured
```

Then sorting might give different results. Check with:
```python
import os
print(sorted(os.listdir('train/')))
```

---

## Confidence Calculation

### For Sigmoid Binary Classification

The confidence should represent **how certain the model is about its prediction**:

```python
prob = 0.92  # Raw sigmoid output

# If prob > 0.5 (predicting Class 1):
label = "Class 1"
confidence = prob * 100  # 92% confident

# If prob < 0.5 (predicting Class 0):
label = "Class 0"
confidence = (1 - prob) * 100  # If prob=0.1, then 90% confident
```

**Why?** Because:
- Prob 0.92 means "92% Class 1, 8% Class 0" → 92% confident it's Class 1
- Prob 0.1 means "10% Class 1, 90% Class 0" → 90% confident it's Class 0

---

## Step-by-Step Debugging

### Step 1: Verify Your Training Data Structure

```bash
cd "D:\computer vision project"
dir "archive (1)/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train/"
```

You should see folders like:
```
fractured/
not fractured/
```

Check the exact names (case, spacing, punctuation).

### Step 2: Check Alphabetical Order

```python
import os
classes = sorted(os.listdir('train/'))
for i, cls in enumerate(classes):
    print(f"Class {i}: {cls}")

# Example output:
# Class 0: fractured
# Class 1: not fractured
```

### Step 3: Load and Inspect Model

```bash
python backend/inspect_model.py
```

This will show:
- Model input/output shapes
- Whether it's sigmoid or softmax
- Inferred class mapping

### Step 4: Test With a Known Image

Manually test with an image you **know** is fractured:

```bash
python backend/test_api.py "path/to/known_fractured_xray.jpg"
```

The result should say "Fractured" with high confidence.

### Step 5: Compare With Visual Inspection

If the result doesn't match your visual inspection:
- The class labels are reversed
- You need to use the other backend option

---

## Common Mistakes & Fixes

### ❌ Mistake 1: Using the Original Logic

```python
# Original (WRONG):
if prediction > 0.5:
    result = "Fractured"
```

✅ **Fix:**
```python
if prediction > 0.5:
    result = "Not Fractured"  # Class 1
else:
    result = "Fractured"      # Class 0
```

### ❌ Mistake 2: Confidence Always Based on Raw Prob

```python
# Wrong - doesn't match predicted label:
if prob > 0.5:
    label = "Not Fractured"
    confidence = prob * 100  # ✓ OK
else:
    label = "Fractured"
    confidence = prob * 100  # ❌ Wrong! prob is low but confidence high
```

✅ **Fix:**
```python
if prob > 0.5:
    label = "Not Fractured"
    confidence = prob * 100
else:
    label = "Fractured"
    confidence = (1 - prob) * 100  # ✓ Inverted
```

### ❌ Mistake 3: Forgetting About Softmax

```python
# If your model outputs [prob_class0, prob_class1]:
# ❌ Won't work:
if preds[0][0] > 0.5:
    label = "Fractured"

# ✅ Should be:
if preds[0][1] > preds[0][0]:
    label = "Fractured"
    confidence = preds[0][1] * 100
```

---

## Implementation Checklist

- [ ] Run `python backend/inspect_model.py`
- [ ] Note the class order output
- [ ] Check model output shape (1 or 2 dimensions)
- [ ] Read the recommendation in the inspection output
- [ ] Choose correct option (A or B) from `backend/main_alternatives.py`
- [ ] Copy correct code to `backend/main.py`
- [ ] Restart backend: `Ctrl+C` then `uvicorn backend.main:app --reload`
- [ ] Test with curl: `curl -X POST "http://localhost:8000/predict" -F "file=@test.jpg"`
- [ ] Verify result matches visual inspection
- [ ] Test in frontend browser

---

## Frontend Display Logic

Once backend returns correct `prediction` and `confidence`:

```typescript
// Home.tsx
const handlePredict = async (file: File) => {
    const res = await predictImage(form)  // {prediction: "Fractured", confidence: 92.4}
    setResult(res)
}

// StatusCard.tsx
const StatusCard = ({ prediction, confidence }) => {
    const isFractured = prediction === 'Fractured'
    
    return (
        <div className={isFractured ? 'red-gradient' : 'green-gradient'}>
            <h2>{prediction}</h2>
            <p>{confidence}%</p>
        </div>
    )
}
```

This should work automatically once backend is fixed.

---

## Still Having Issues?

1. **Run the inspection script** and share the output
2. **Check if model file exists**: `ls fracture_model.h5`
3. **Verify training folder names** match exactly
4. **Test with known images** (you manually inspected)
5. **Check backend logs** for errors

---

## References

- TensorFlow Binary Classification: https://www.tensorflow.org/guide/keras/working_with_data
- ImageDataGenerator: https://tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
- Sigmoid vs Softmax: https://en.wikipedia.org/wiki/Softmax_function

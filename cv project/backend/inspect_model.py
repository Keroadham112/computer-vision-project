#!/usr/bin/env python
"""
Debugging script to inspect TensorFlow model structure and determine
the correct class label mapping.
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
import os


def inspect_model(model_path='fracture_model.h5'):
    """Inspect model architecture and input/output shapes"""
    print("=" * 60)
    print("TensorFlow Model Inspector")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("\nLooking for model files in current directory...")
        for f in os.listdir('.'):
            if f.endswith('.h5'):
                print(f"  Found: {f}")
        return
    
    print(f"\n✓ Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print("\n--- Model Summary ---")
    model.summary()
    
    print("\n--- Input Shape ---")
    print(f"  Expected input: {model.input_shape}")
    
    print("\n--- Output Shape ---")
    print(f"  Output shape: {model.output_shape}")
    output_dim = model.output_shape[-1]
    
    if output_dim == 1:
        print(f"  ✓ Single output (sigmoid/binary classification)")
        print(f"  Output > 0.5 → Class 1")
        print(f"  Output < 0.5 → Class 0")
    elif output_dim == 2:
        print(f"  ✓ Two outputs (softmax/categorical)")
        print(f"  Class 0: [1, 0]")
        print(f"  Class 1: [0, 1]")
    else:
        print(f"  ⚠ Unusual output dimension: {output_dim}")
    
    print("\n--- Last Layer ---")
    last_layer = model.layers[-1]
    print(f"  Type: {last_layer.__class__.__name__}")
    if hasattr(last_layer, 'activation'):
        print(f"  Activation: {last_layer.activation.__name__ if last_layer.activation else 'None'}")
    
    return model, output_dim


def test_prediction(model, test_input=None):
    """Run a test prediction"""
    print("\n" + "=" * 60)
    print("Test Prediction")
    print("=" * 60)
    
    # Create a dummy input if not provided
    if test_input is None:
        input_shape = model.input_shape[1:]
        test_input = np.random.rand(1, *input_shape).astype('float32')
    
    print(f"\nInput shape: {test_input.shape}")
    prediction = model.predict(test_input, verbose=0)
    
    print(f"Raw output: {prediction}")
    print(f"Output shape: {prediction.shape}")
    
    if prediction.shape[-1] == 1:
        prob = float(prediction[0][0])
        print(f"\nSigmoid output: {prob:.4f}")
        print(f"  > 0.5: {'YES' if prob > 0.5 else 'NO'}")
        print(f"  Predicted class: {1 if prob > 0.5 else 0}")
    else:
        print(f"\nSoftmax outputs:")
        for i, p in enumerate(prediction[0]):
            print(f"  Class {i}: {p:.4f}")
        predicted_class = np.argmax(prediction[0])
        print(f"Predicted class: {predicted_class}")


def infer_class_mapping(data_dir='archive (1)/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification'):
    """
    Infer class mapping from directory structure.
    When using flow_from_directory, classes are sorted alphabetically.
    """
    print("\n" + "=" * 60)
    print("Class Mapping Inference")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("\nLooking for similar directories...")
        for root, dirs, files in os.walk('.'):
            if 'train' in dirs or 'test' in dirs:
                print(f"  Found: {root}")
        return None
    
    # Check for train directory
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"❌ Train directory not found: {train_dir}")
        return None
    
    # Get class folders
    class_folders = sorted([d for d in os.listdir(train_dir) 
                           if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"\n✓ Found training directory: {train_dir}")
    print(f"\nClass folders (sorted alphabetically):")
    for idx, folder in enumerate(class_folders):
        folder_path = os.path.join(train_dir, folder)
        num_files = len([f for f in os.listdir(folder_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  Class {idx}: '{folder}' ({num_files} images)")
    
    if len(class_folders) == 2:
        print(f"\n📊 Class Mapping:")
        print(f"  Class 0 = '{class_folders[0]}'")
        print(f"  Class 1 = '{class_folders[1]}'")
        
        # Infer which is fractured
        class_0_name = class_folders[0].lower()
        class_1_name = class_folders[1].lower()
        
        print(f"\n💡 Sigmoid Interpretation:")
        if 'fracture' in class_0_name or 'fractured' in class_0_name:
            print(f"  output > 0.5 → Class 1 ('{class_folders[1]}')")
            print(f"  output < 0.5 → Class 0 ('{class_folders[0]}')")
        else:
            print(f"  output > 0.5 → Class 1 ('{class_folders[1]}')")
            print(f"  output < 0.5 → Class 0 ('{class_folders[0]}')")
    
    return class_folders


def main():
    print("\n🔍 Inspecting TensorFlow Fracture Classification Model\n")
    
    # Step 1: Inspect model
    model, output_dim = inspect_model()
    if model is None:
        return
    
    # Step 2: Test prediction
    test_prediction(model)
    
    # Step 3: Infer class mapping from directory structure
    class_mapping = infer_class_mapping()
    
    # Step 4: Recommendation
    if class_mapping:
        print("\n" + "=" * 60)
        print("Recommendation")
        print("=" * 60)
        
        class_0 = class_mapping[0].lower()
        class_1 = class_mapping[1].lower()
        
        print(f"\nBased on alphabetical order:")
        print(f"  Sigmoid output > 0.5 → Class 1: '{class_mapping[1]}'")
        print(f"  Sigmoid output < 0.5 → Class 0: '{class_mapping[0]}'")
        
        if 'fracture' in class_0 or 'fractured' in class_0:
            print(f"\n⚠️  Class 0 contains 'fractured'")
            print(f"    So sigmoid < 0.5 predicts 'fractured'")
            print(f"    This might be reversed from typical convention!")
        else:
            print(f"\n✓ Class 0 is '{class_mapping[0]}'")
            print(f"  Class 1 is '{class_mapping[1]}'")


if __name__ == '__main__':
    main()

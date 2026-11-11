import json

with open("model/training_history.json", "r") as f:
    history = json.load(f)

print("\n--- Model Performance Results ---")

print(f"Final Training Accuracy: {history['accuracy'][-1] * 100:.2f}%")
print(f"Final Validation Accuracy: {history['val_accuracy'][-1] * 100:.2f}%")

print(f"Final Training Loss: {history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")

print(f"Best Training Accuracy: {max(history['accuracy']) * 100:.2f}%")
print(f"Best Validation Accuracy: {max(history['val_accuracy']) * 100:.2f}%")
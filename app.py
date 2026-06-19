import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "waste_classifier.h5"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Your two classes
CLASS_NAMES = [
    "Organic",
    "Recyclable"
]

def predict_waste(image):
    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]

    # Return class probabilities
    results = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(min(len(CLASS_NAMES), len(predictions)))
    }

    return results


demo = gr.Interface(
    fn=predict_waste,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="♻️ Smart Waste Classification",
    description="Upload a waste image and the model will classify it as Organic or Recyclable."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
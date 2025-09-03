from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
MODEL_PATH = os.path.join("model", "brain_tumor_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Set to match model input shape
INPUT_SIZE = (128, 128)

# Class labels
class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']

# ✅ Grad-CAM helper
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ✅ Prediction and Grad-CAM pipeline
def predict_and_explain(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, INPUT_SIZE)
    img_array = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    pred_probs = model.predict(img_input)
    pred_class = np.argmax(pred_probs[0])
    pred_label = class_names[pred_class]

    # ✅ Use the actual layer name from your model
    heatmap = make_gradcam_heatmap(img_input, model, 'target_conv_layer')

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.uint8(img), 0.6, heatmap_colored, 0.4, 0)

    heatmap_path = os.path.join(UPLOAD_FOLDER, 'gradcam_output.png')
    cv2.imwrite(heatmap_path, superimposed_img)

    return pred_label, heatmap_path


# ✅ NVIDIA chatbot (OpenAI-compatible API)
def ask_nvidia_chatbot(message):
    api_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    api_key="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" # Store in Render environment

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 300
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Chatbot error: {response.status_code}"
    except Exception as e:
        return f"Error contacting chatbot: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('index.html', error="No file selected.")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        label, gradcam_path = predict_and_explain(filepath)
        return render_template('index.html', label=label, gradcam_path=gradcam_path, uploaded_image=filepath)

    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    pred_label = request.json.get('prediction')

    if not user_message or not pred_label:
        return jsonify({"response": "Missing user message or prediction."}), 400

    full_prompt = f"I am suffering with {pred_label}. {user_message}"
    bot_response = ask_nvidia_chatbot(full_prompt)
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(debug=True)

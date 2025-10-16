import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import torch
import cv2
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model

# === SHAP Explanation ===
def explain_shap(model, X, feature_names, max_samples=100):
    explainer = shap.Explainer(model.predict_proba, X[:max_samples])
    shap_values = explainer(X[:max_samples])
    shap.summary_plot(shap_values, features=X[:max_samples], feature_names=feature_names)

# === LIME Explanation ===
def explain_lime(model, X, feature_names, class_names, instance_idx=0):
    explainer = LimeTabularExplainer(
        X, 
        feature_names=feature_names, 
        class_names=class_names, 
        discretize_continuous=True, 
        mode='classification'
    )
    exp = explainer.explain_instance(
        X[instance_idx], 
        model.predict_proba, 
        top_labels=1, 
        num_features=15
    )
    exp.show_in_notebook(show_table=True)
    fig = exp.as_pyplot_figure()
    plt.title(f'LIME Explanation for Sample #{instance_idx}')
    plt.tight_layout()
    plt.show()

# === Grad-CAM Explanation ===
def gradcam(image_path, save_path=None, layer_name='block14_sepconv2_act'):
    model = Xception(weights='imagenet', include_top=True)
    img = load_img(image_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap, (299, 299))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    original = cv2.imread(image_path)
    original = cv2.resize(original, (299, 299))
    superimposed = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    if save_path:
        cv2.imwrite(save_path, superimposed)
    else:
        plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Grad-CAM')
        plt.show()

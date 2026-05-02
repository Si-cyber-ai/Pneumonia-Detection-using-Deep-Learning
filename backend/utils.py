import cv2
import numpy as np
import tensorflow as tf
import base64

def preprocess_image(image_bytes):
    if not image_bytes:
        raise ValueError("The uploaded file is empty.")

    # Decode incoming bytes into cv2 image
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Unsupported or corrupted image file. Please upload a valid JPG or PNG chest X-ray.")

    # Convert BGR to RGB for clinical accuracy
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Highly refined interpolation
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return np.expand_dims(img_normalized, axis=0), img_resized

def check_image_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    
    is_poor_quality = blur_score < 100 or brightness < 50 or brightness > 200
    
    return {
        "blur_score": float(blur_score),
        "brightness": float(brightness),
        "is_poor_quality": bool(is_poor_quality)
    }

def find_last_conv_layer(model):
    # Traverse backwards to find the last 4D output (Convolutional/Pooling)
    for layer in reversed(model.layers):
        output_shape = getattr(layer, 'output_shape', None)
        if output_shape is not None and len(output_shape) == 4 and 'conv' in layer.name.lower():
            return layer.name
    for layer in reversed(model.layers):
        output_shape = getattr(layer, 'output_shape', None)
        if output_shape is not None and len(output_shape) == 4:
            return layer.name
    raise ValueError("Could not find a valid target conv layer for Grad-CAM.")

def build_connected_grad_model(model):
    conv_candidates = []
    fallback_candidates = []

    for layer in reversed(model.layers):
        output_shape = getattr(layer, 'output_shape', None)
        if output_shape is None:
            continue

        if isinstance(output_shape, (list, tuple)) and len(output_shape) == 4:
            if 'conv' in layer.name.lower():
                conv_candidates.append(layer)
            else:
                fallback_candidates.append(layer)

    for layer in conv_candidates + fallback_candidates:
        try:
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[layer.output, model.output],
            )
            return grad_model, layer.name
        except ValueError:
            # Some nested-layer tensors are not connected to the outer model graph in Keras 3.
            continue

    raise ValueError("Could not build a connected Grad-CAM model from available 4D layers.")

def make_gradcam_heatmap(img_array, model, pred_index=None):
    try:
        grad_model, last_conv_layer_name = build_connected_grad_model(model)

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = 0 if preds.shape[-1] == 1 else tf.argmax(preds[0])

            # Binary classifiers often expose a single sigmoid output; multi-class models use the selected class index.
            class_channel = preds[:, 0] if preds.shape[-1] == 1 else preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
    except ValueError:
        # Fallback for models where intermediate conv tensors are not graph-connected in Keras 3.
        with tf.GradientTape() as tape:
            input_tensor = tf.cast(img_array, tf.float32)
            tape.watch(input_tensor)
            preds = model(input_tensor)
            if pred_index is None:
                pred_index = 0 if preds.shape[-1] == 1 else tf.argmax(preds[0])
            class_channel = preds[:, 0] if preds.shape[-1] == 1 else preds[:, pred_index]

        input_grads = tape.gradient(class_channel, input_tensor)[0]
        heatmap = tf.reduce_mean(tf.abs(input_grads), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.math.reduce_max(heatmap)
        heatmap = heatmap / tf.where(max_value != 0, max_value, tf.ones_like(max_value))
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        return heatmap
    
    # Use Grad-CAM++ approach for gradients
    first_derivative = grads
    second_derivative = tf.square(first_derivative)
    third_derivative = first_derivative * second_derivative
    
    global_sum = tf.reduce_sum(last_conv_layer_output, axis=(0, 1, 2))
    alpha_num = second_derivative
    alpha_denom = 2.0 * second_derivative + third_derivative * global_sum[..., tf.newaxis, tf.newaxis]
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom
    
    weights = tf.maximum(first_derivative, 0.0)
    alpha_normalization_constant = tf.reduce_sum(alphas, axis=(0, 1, 2))
    alphas /= tf.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, tf.ones_like(alpha_normalization_constant))
    
    deep_linearization_weights = tf.reduce_sum(weights * alphas, axis=(1, 2))
    
    heatmap = tf.reduce_sum(tf.multiply(deep_linearization_weights[0], last_conv_layer_output[0]), axis=-1)
    
    # ReLU
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.math.reduce_max(heatmap)
    heatmap = heatmap / tf.where(max_value != 0, max_value, tf.ones_like(max_value))
    heatmap = heatmap.numpy()
    
    # Resize with CUBIC, apply Gaussian smoothing
    heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    # Re-normalize
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
        
    return heatmap

def extract_hotspots(heatmap, threshold=0.6, min_area=25):
    mask = np.uint8(heatmap >= threshold)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    hotspots = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        centroid_x, centroid_y = centroids[label_idx]

        hotspots.append({
            "bbox": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },
            "centroid": {
                "x": float(centroid_x),
                "y": float(centroid_y),
            },
            "area": area,
            "mean_activation": float(np.mean(heatmap[labels == label_idx]))
        })

    hotspots.sort(key=lambda item: item["area"], reverse=True)
    return hotspots

def image_to_base64(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def generate_gradcam_base64(img, heatmap):
    # Returns a colored pure heatmap to allow frontend blending and a grayscale mask for precise inspection.
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # TURBO provides much better clinical contrast than JET
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    grayscale_mask = cv2.cvtColor(heatmap_uint8, cv2.COLOR_GRAY2RGB)
    return image_to_base64(colored_heatmap), image_to_base64(grayscale_mask), image_to_base64(img)

def calculate_focus_score(heatmap):
    high_activation = np.sum(heatmap > 0.6)
    total_pixels = heatmap.size
    return float(high_activation / total_pixels)

def get_severity(focus_score):
    if focus_score < 0.1:
        return "NONE"
    elif focus_score < 0.25:
        return "MILD"
    elif focus_score < 0.5:
        return "MODERATE"
    else:
        return "SEVERE"

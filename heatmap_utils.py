import cv2
import numpy as np

def overlay_heatmap(heatmap, original_input, output_path=None, alpha=0.5):
    # Load original image or frame
    if isinstance(original_input, str):
        original = cv2.imread(original_input)
        if original is None:
            raise ValueError(f"Failed to load image: {original_input}")
    elif isinstance(original_input, np.ndarray):
        original = original_input.copy()
    else:
        raise TypeError("Expected a file path (str) or image/frame (np.ndarray)")

    # Ensure original has 3 channels
    if len(original.shape) < 3 or original.shape[2] == 1:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # Normalize heatmap
    heatmap = np.nan_to_num(heatmap)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    # Resize heatmap to match original size
    try:
        heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    except Exception as e:
        print(f"[ERROR] Failed to resize heatmap: {e}")
        raise

    # Apply colormap (e.g., INFERNO or other stylized ones)
    try:
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_INFERNO)
    except Exception as e:
        print(f"[ERROR] Failed to apply colormap: {e}")
        raise

    # Blend original + heatmap
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)

    # Save result if path provided
    if output_path:
        try:
            cv2.imwrite(output_path, overlay)
        except Exception as e:
            print(f"[ERROR] Failed to save overlay: {e}")

    return overlay

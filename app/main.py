import os
import time
import json
import base64
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import re

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Age Prediction Visualizer")

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Load predictions
predictions_df = pd.read_csv("predictions.csv", header=None, names=["image", "prediction"])
predictions_df["image"] = predictions_df["image"].str.replace("'", "")
predictions_df["prediction"] = predictions_df["prediction"].str.replace("'", "")
predictions_df = predictions_df.set_index("image")

# Helper function to extract age from filename
def extract_age_from_filename(filename):
    match = re.match(r'(\d+)-[mf]-\d+\.png', filename)
    if match:
        age = int(match.group(1))
        if 7 <= age <= 10:
            return "Age_7-10"
        elif 11 <= age <= 14:
            return "Age_11-14"  
        elif 15 <= age <= 18:
            return "Age_15-18"
        elif 19 <= age <= 24:
            return "Age_19-24"
        else:
            # Default if age doesn't match any category
            return f"Age_{age}"
    return "Unknown"

# Get all image files from the dataset folder
def get_all_images():
    image_files = [f for f in os.listdir("dataset") if f.endswith(".png")]
    return sorted(image_files)

# Get images that are in predictions.csv
def get_images_with_predictions():
    all_images = get_all_images()
    return [img for img in all_images if img in predictions_df.index]

# Function to visualize masks
def visualize_masks(image_path):
    base_name = os.path.basename(image_path)
    mask_path = os.path.join("dataset", base_name.replace(".png", "_m3.json"))
    
    # Load the original image
    original_image = cv2.imread(os.path.join("dataset", base_name))
    if original_image is None:
        print(f"Image file not found: {image_path}")
        return np.full((500, 500, 3), 127, dtype=np.uint8)  # Return a default gray image
    
    # Convert to grayscale for mask processing while keeping original for overlay
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = gray_image.shape[:2]
    
    # If no mask file, return original image with an overlay message
    if not os.path.exists(mask_path):
        cv2.putText(original_image, "No mask found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return original_image
    
    # Load the mask data
    try:
        with open(mask_path, 'r') as f:
            mask_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading mask file: {e}")
        cv2.putText(original_image, "Error loading mask", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return original_image
    
    # Create a copy of the original for overlay
    overlay_image = original_image.copy()
    mask_found = False
    
    # First, process overlays for all masks to highlight regions
    for shape in mask_data.get('shapes', []):
        label = shape.get('label', '')
        points = shape.get('points', [])
        
        if not points or len(points) < 2:
            continue
            
        # Calculate bounding box
        x_min = int(min([p[0] for p in points]))
        x_max = int(max([p[0] for p in points]))
        y_min = int(min([p[1] for p in points]))
        y_max = int(max([p[1] for p in points]))
        
        # Check if label is either 'm3ll' or 'm3_ll' for green color
        is_m3ll_mask = 'm3ll' in label or 'm3_ll' in label
        
        # Draw a semi-transparent rectangle on the region
        color = (0, 255, 0) if is_m3ll_mask else (0, 0, 255)  # Green for m3ll/m3_ll, red for others
        cv2.rectangle(overlay_image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Create a semitransparent rectangle
        mask = np.zeros_like(original_image)
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), color, -1)
        overlay_image = cv2.addWeighted(overlay_image, 0.85, mask, 0.15, 0)
        
        # Add label text
        cv2.putText(overlay_image, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        mask_found = True
    
    # If no mask is found, return the original image with a message
    if not mask_found:
        cv2.putText(original_image, "No valid masks found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return original_image
    
    # Now specifically look for m3ll/m3_ll mask for enhancement
    enhanced_region = None
    
    for shape in mask_data.get('shapes', []):
        label = shape.get('label', '')
        points = shape.get('points', [])
        
        # Check if label contains either 'm3ll' or 'm3_ll'
        if not ('m3ll' in label or 'm3_ll' in label) or not points or len(points) < 2:
            continue
            
        # Calculate bounding box
        x_min = int(min([p[0] for p in points]))
        x_max = int(max([p[0] for p in points]))
        y_min = int(min([p[1] for p in points]))
        y_max = int(max([p[1] for p in points]))
        
        # Ensure coordinates are within image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_width, x_max)
        y_max = min(image_height, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            continue
            
        # Crop the original image using bounding box
        cropped_image = gray_image[y_min:y_max, x_min:x_max]
        
        # Image enhancement steps
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(cropped_image)
        
        denoised = cv2.GaussianBlur(enhanced, (3,3), 0)
        
        gaussian_3 = cv2.GaussianBlur(denoised, (9,9), 2.0)
        unsharp_image = cv2.addWeighted(denoised, 1.5, gaussian_3, -0.5, 0)
        
        normalized = cv2.normalize(unsharp_image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to color for overlay
        enhanced_color = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        
        # Add a colored border to the enhanced region
        border_size = 2
        enhanced_color = cv2.copyMakeBorder(enhanced_color, border_size, border_size, border_size, border_size, 
                                          cv2.BORDER_CONSTANT, value=(0, 255, 0))
        
        # Update the region size after adding the border
        y_min -= border_size
        x_min -= border_size
        y_max += border_size
        x_max += border_size
        
        # Ensure coordinates are still within image boundaries
        y_min = max(0, y_min)
        x_min = max(0, x_min)
        y_max = min(image_height, y_max)
        x_max = min(image_width, x_max)
        
        # Show both the enhanced m3ll region and the overlay
        # Create a picture-in-picture effect in the corner
        pip_height, pip_width = enhanced_color.shape[:2]
        pip_scale = 0.3  # Scale for picture-in-picture
        pip_resized = cv2.resize(enhanced_color, (int(pip_width * pip_scale), int(pip_height * pip_scale)))
        
        # Place in top-right corner with padding
        padding = 10
        h, w = pip_resized.shape[:2]
        overlay_image[padding:padding+h, overlay_image.shape[1]-w-padding:overlay_image.shape[1]-padding] = pip_resized
        
        break  # Only process the first m3ll/m3_ll mask
    
    return overlay_image

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Only show images that are in predictions.csv
    images = get_images_with_predictions()
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "images": images}
    )

@app.post("/process", response_class=HTMLResponse)
async def process_image(request: Request, image_file: str = Form(...)):
    # Get true label from filename
    true_label = extract_age_from_filename(image_file)
    
    # Check if image exists in predictions dataframe
    if image_file not in predictions_df.index:
        return templates.TemplateResponse(
            "result.html", 
            {
                "request": request,
                "image_file": image_file,
                "true_label": true_label,
                "prediction": "No prediction available",
                "is_correct": False,
                "error_message": "This image does not have a prediction in the dataset."
            },
            status_code=404
        )
    
    # Get prediction from CSV
    try:
        pred_value = predictions_df.loc[image_file, "prediction"]
        # Handle case where prediction is a Series
        if isinstance(pred_value, pd.Series):
            prediction = pred_value.iloc[0] if not pred_value.empty else "Unknown"
        else:
            prediction = pred_value
            
        # Normalize prediction string
        prediction = str(prediction).strip()
    except Exception as e:
        prediction = "Error retrieving prediction"
        print(f"Error processing prediction for {image_file}: {str(e)}")
    
    # Check if prediction is correct
    # Normalize both strings for comparison
    normalized_true = true_label.strip().lower()
    normalized_pred = str(prediction).strip().lower()
    is_correct = normalized_true == normalized_pred
    
    # Debug information
    print(f"Image: {image_file}")
    print(f"True label: '{normalized_true}'")
    print(f"Prediction: '{normalized_pred}'")
    print(f"Match: {is_correct}")
    
    return templates.TemplateResponse(
        "result.html", 
        {
            "request": request,
            "image_file": image_file,
            "true_label": true_label,
            "prediction": prediction,
            "is_correct": is_correct,
            "debug_info": {
                "true_norm": normalized_true,
                "pred_norm": normalized_pred,
                "length_true": len(normalized_true),
                "length_pred": len(normalized_pred)
            }
        }
    )

@app.get("/image/{image_name}")
async def get_original_image(image_name: str):
    image_path = os.path.join("dataset", image_name)
    
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            # Return a placeholder image or error response
            placeholder = np.full((500, 500, 3), 200, dtype=np.uint8)  # Gray image
            cv2.putText(placeholder, "Image not found", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            success, encoded_image = cv2.imencode(".png", placeholder)
            return StreamingResponse(BytesIO(encoded_image.tobytes()), media_type="image/png")
        
        # Read image as bytes
        with open(image_path, "rb") as f:
            img = f.read()
        
        # Return the image as a streaming response
        return StreamingResponse(BytesIO(img), media_type="image/png")
    except Exception as e:
        print(f"Error loading image {image_name}: {str(e)}")
        # Return error image
        error_img = np.full((500, 500, 3), 200, dtype=np.uint8)
        cv2.putText(error_img, f"Error: {str(e)[:30]}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        success, encoded_image = cv2.imencode(".png", error_img)
        return StreamingResponse(BytesIO(encoded_image.tobytes()), media_type="image/png")

@app.get("/masked-image/{image_name}")
async def get_masked_image(image_name: str):
    image_path = os.path.join("dataset", image_name)
    
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            # Return a placeholder image or error response
            placeholder = np.full((500, 500, 3), 200, dtype=np.uint8)  # Gray image
            cv2.putText(placeholder, "Image not found", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            success, encoded_image = cv2.imencode(".png", placeholder)
            return StreamingResponse(BytesIO(encoded_image.tobytes()), media_type="image/png")
        
        # Generate masked image
        masked_img = visualize_masks(image_path)
        
        # Convert to bytes
        success, encoded_image = cv2.imencode(".png", masked_img)
        if not success:
            return {"error": "Failed to encode image"}
        
        # Return the masked image as a streaming response
        return StreamingResponse(BytesIO(encoded_image.tobytes()), media_type="image/png")
    except Exception as e:
        print(f"Error processing masked image {image_name}: {str(e)}")
        # Return error image
        error_img = np.full((500, 500, 3), 200, dtype=np.uint8)
        cv2.putText(error_img, f"Error: {str(e)[:30]}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        success, encoded_image = cv2.imencode(".png", error_img)
        return StreamingResponse(BytesIO(encoded_image.tobytes()), media_type="image/png")

@app.get("/stats")
async def get_stats():
    # Only consider images that are in predictions.csv
    images = get_images_with_predictions()
    
    total = len(images)
    correct = 0
    incorrect = 0
    
    for img in images:
        try:
            true_label = extract_age_from_filename(img)
            
            # Get prediction and ensure it's a string
            pred_value = predictions_df.loc[img, "prediction"]
            if isinstance(pred_value, pd.Series):
                # If it's a Series (multiple matches), take the first value
                pred_label = pred_value.iloc[0] if not pred_value.empty else "Unknown"
            else:
                pred_label = pred_value
            
            # Normalize strings for comparison
            true_norm = true_label.strip().lower()
            pred_norm = str(pred_label).strip().lower()
            
            if true_norm == pred_norm:
                correct += 1
            else:
                incorrect += 1
                
        except Exception as e:
            print(f"Error processing {img}: {str(e)}")
            incorrect += 1
    
    # Ensure total matches the sum of correct and incorrect
    if correct + incorrect != total:
        incorrect = total - correct
    
    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": round((correct / total * 100), 1) if total > 0 else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 
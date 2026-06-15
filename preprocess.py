import os
import json
import cv2
from shapely import wkt
from PIL import Image

def extract_and_crop_buildings(image_path, json_path, output_dir):
    """
    Parses geographic JSON metadata, extracts building boundaries via WKT,
    crops footprints from high-resolution satellite imagery using OpenCV,
    and sorts crops into structural class directories.
    """
    # 1. Setup Directories
    damaged_dir = os.path.join(output_dir, 'Damaged')
    safe_dir = os.path.join(output_dir, 'Safe')
    os.makedirs(damaged_dir, exist_ok=True)
    os.makedirs(safe_dir, exist_ok=True)
    
    # 2. Load Raw Image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target satellite image not found at: {image_path}")
        
    image = cv2.imread(image_path)
    # Convert BGR to RGB for neural network compatibility
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. Load JSON Metadata
    with open(json_path, 'r') as file:
        metadata = json.load(file)
        
    building_features = metadata.get('features', {}).get('xy', [])
    
    crop_count = 0
    error_count = 0
    
    # 4. Extraction Loop
    for idx, feature in enumerate(building_features):
        try:
            properties = feature.get('properties', {})
            damage_subtype = properties.get('subtype', 'no-damage')
            
            # Map classes
            if damage_subtype == 'no-damage':
                target_folder = safe_dir
            else:
                target_folder = damaged_dir
                
            wkt_geometry = feature.get('wkt', '')
            if not wkt_geometry:
                continue
                
            # Parse WKT to Cartesian bounds
            polygon = wkt.loads(wkt_geometry)
            min_x, min_y, max_x, max_y = polygon.bounds
            
            # Add 5-pixel padding
            pixel_padding = 5
            img_height, img_width, _ = image_rgb.shape
            
            start_y = max(0, int(min_y) - pixel_padding)
            end_y = min(img_height, int(max_y) + pixel_padding)
            start_x = max(0, int(min_x) - pixel_padding)
            end_x = min(img_width, int(max_x) + pixel_padding)
            
            # Crop using OpenCV slicing
            cropped_matrix = image_rgb[start_y:end_y, start_x:end_x]
            
            # Skip invalid arrays
            if cropped_matrix.size == 0 or cropped_matrix.shape[0] == 0 or cropped_matrix.shape[1] == 0:
                error_count += 1
                continue
                
            # Convert to PIL and standardize size to 128x128
            cropped_image = Image.fromarray(cropped_matrix)
            standardized_crop = cropped_image.resize((128, 128), Image.Resampling.LANCZOS)
            
            # Save to disk
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_filename}_building_{idx}.png"
            final_output_path = os.path.join(target_folder, output_filename)
            
            standardized_crop.save(final_output_path, "PNG")
            crop_count += 1
            
        except Exception:
            error_count += 1
            continue
            
    print(f"--- Processing Execution Report ---")
    print(f"Successfully Structured Crops: {crop_count}")
    print(f"Skipped Edge/Degenerate Polygons: {error_count}\n")

if __name__ == "__main__":
    # Update these paths to match your actual file locations
    extract_and_crop_buildings(
        image_path="guatemala_post_disaster.png",
        json_path="guatemala_post_disaster.json",
        output_dir="dataset_extracted"
    )

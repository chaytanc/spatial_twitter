import os
import glob
import shutil
import yaml
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image


def load_config(config_file="params.yaml"):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None


def create_spatial_animations(config, base_dir="analysis/spatial_results", output_dir="animations"):
    """
    Find existing PNG files in the spatial results directories and create animations.
    
    Args:
        base_dir: Base directory containing the spatial results folders (0-20)
        output_dir: Directory to save the animations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file patterns to search for
    file_patterns = {
        "accuracy_comparison": "model_accuracy_comparison.png",
        "comparison_maps": "model_comparison_maps.png",
        "comparison_metrics": "model_comparison_metric.png",
        "variance_maps": "variance_comparison_maps.png"
    }
    
    # Dictionary to collect image paths by type
    animation_files = {pattern: [] for pattern in file_patterns}
    
    # Find all relevant images for each sample (0-20)
    for sample_num in range(config.get("N_CLUSTERS")): 
        sample_dir = os.path.join(base_dir, str(sample_num))
        
        # Skip if directory doesn't exist
        if not os.path.exists(sample_dir):
            print(f"Warning: Sample directory {sample_num} not found. Skipping.")
            continue
        
        # Find each image type in this sample directory
        for img_type, file_pattern in file_patterns.items():
            img_path = os.path.join(sample_dir, file_pattern)
            if os.path.exists(img_path):
                animation_files[img_type].append((sample_num, img_path))
                print(f"Found {img_type} for sample {sample_num}: {img_path}")
            else:
                print(f"Missing {img_type} for sample {sample_num}")
    
    # Create animations
    for img_type, files in animation_files.items():
        if not files:
            print(f"No images found for {img_type} animation")
            continue
            
        # Sort by sample number
        files.sort(key=lambda x: x[0])
        image_paths = [path for _, path in files]
        
        # Create output paths
        gif_path = os.path.join(output_dir, f"{img_type}_animation.gif")
        mp4_path = os.path.join(output_dir, f"{img_type}_animation.mp4")
        
        # Load images
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                # Add sample number as text overlay
                sample_num = int(Path(img_path).parts[-2])  # Extract sample number from path
                
                # Convert to numpy array for processing
                img_array = np.array(img)
                images.append(img_array)
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
        
        if not images:
            print(f"No valid images loaded for {img_type}")
            continue
            
        print(f"Creating {img_type} animation with {len(images)} frames")
        
        # Create GIF
        try:
            imageio.mimsave(gif_path, images, duration=1, loop=0)
            print(f"✓ Created GIF: {gif_path}")
        except Exception as e:
            print(f"Error creating GIF: {e}")
        
        # Create MP4 if possible
        try:
            imageio.mimsave(mp4_path, images, fps=2)
            print(f"✓ Created MP4: {mp4_path}")
        except Exception as e:
            print(f"Could not create MP4 (requires imageio-ffmpeg): {e}")
            
    print(f"\nAll animations created in {output_dir}")
    return animation_files

if __name__ == "__main__":
    config = load_config()
    results = create_spatial_animations(config)


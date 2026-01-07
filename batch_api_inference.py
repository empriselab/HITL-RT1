#!/usr/bin/env python3
"""
Batch inference script using api_client.py to process images from folders.
Stores predictions with mean_prediction and total_variance.
"""

import os
import json
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from api_client import predict_bbox, encode_image_to_base64

def create_variance_histograms(all_results, output_dir, output_basename):
    """
    Create histograms for total_variance and centroid_variance distributions.
    
    Args:
        all_results: List of prediction results
        output_dir: Directory to save histograms
        output_basename: Base name for output files
    """
    if not all_results:
        print("[WARNING] No results to create histograms from")
        return
    
    # Extract variance values
    total_variances = [r['total_variance'] for r in all_results if r.get('total_variance') is not None]
    centroid_variances = [r['centroid_variance'] for r in all_results if r.get('centroid_variance') is not None]
    
    if not total_variances and not centroid_variances:
        print("[WARNING] No variance data found for histograms")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram 1: Total Variance
    if total_variances:
        ax1.hist(total_variances, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Total Variance', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Distribution of Total Variance\n(n={len(total_variances)}, mean={np.mean(total_variances):.6f}, std={np.std(total_variances):.6f})', fontsize=11)
        ax1.grid(True, alpha=0.3)
        # Add vertical line for mean
        ax1.axvline(np.mean(total_variances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(total_variances):.6f}')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No total_variance data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Distribution of Total Variance')
    
    # Histogram 2: Centroid Variance
    if centroid_variances:
        ax2.hist(centroid_variances, bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax2.set_xlabel('Centroid Variance', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Distribution of Centroid Variance\n(n={len(centroid_variances)}, mean={np.mean(centroid_variances):.6f}, std={np.std(centroid_variances):.6f})', fontsize=11)
        ax2.grid(True, alpha=0.3)
        # Add vertical line for mean
        ax2.axvline(np.mean(centroid_variances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(centroid_variances):.6f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No centroid_variance data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Distribution of Centroid Variance')
    
    plt.tight_layout()
    
    # Save combined histogram
    histogram_path = output_path / f"{output_basename}_variance_histograms.png"
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved variance histograms to: {histogram_path}")
    plt.close()
    
    # Also create individual histograms
    if total_variances:
        plt.figure(figsize=(8, 6))
        plt.hist(total_variances, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel('Total Variance', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of Total Variance\n(n={len(total_variances)}, mean={np.mean(total_variances):.6f}, std={np.std(total_variances):.6f})', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.axvline(np.mean(total_variances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(total_variances):.6f}')
        plt.legend()
        plt.tight_layout()
        total_hist_path = output_path / f"{output_basename}_total_variance_histogram.png"
        plt.savefig(total_hist_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved total variance histogram to: {total_hist_path}")
        plt.close()
    
    if centroid_variances:
        plt.figure(figsize=(8, 6))
        plt.hist(centroid_variances, bins=30, edgecolor='black', alpha=0.7, color='coral')
        plt.xlabel('Centroid Variance', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of Centroid Variance\n(n={len(centroid_variances)}, mean={np.mean(centroid_variances):.6f}, std={np.std(centroid_variances):.6f})', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.axvline(np.mean(centroid_variances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(centroid_variances):.6f}')
        plt.legend()
        plt.tight_layout()
        centroid_hist_path = output_path / f"{output_basename}_centroid_variance_histogram.png"
        plt.savefig(centroid_hist_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved centroid variance histogram to: {centroid_hist_path}")
        plt.close()

def get_image_files(input_dir, recursive=False):
    """Get all image files from directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    if recursive:
        # Recursively search through subdirectories
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, filename))
    else:
        # Only search in the specified directory
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(input_dir, filename))
    
    image_files.sort()  # Sort for consistent ordering
    return image_files

def process_images(api_url, input_dir, output_file, action=None, food_type=None, 
                   text_instruction="Skewer the food item", recursive=False):
    """
    Process all images in directory/folders and store predictions.
    
    Args:
        api_url: API server URL
        input_dir: Directory containing images (or folders of images)
        output_file: Output JSON/CSV file path
        action: Action type (optional)
        food_type: Food type (optional)
        text_instruction: Text instruction for prediction
        recursive: If True, process images recursively in subdirectories
    """
    # Get all image files
    print(f"[INFO] Searching for images in: {input_dir}")
    image_files = get_image_files(input_dir, recursive=recursive)
    
    if not image_files:
        raise ValueError(f"No image files found in: {input_dir}")
    
    print(f"[INFO] Found {len(image_files)} images to process")
    
    # Store all results
    all_results = []
    failed_images = []
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"\n[INFO] Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Call API
            result = predict_bbox(
                api_url=api_url,
                image_path=image_path,
                text_instruction=text_instruction,
                action=action,
                food_type=food_type
            )
            
            # Extract relevant data
            if result.get('success'):
                mean_pred = result.get('mean_prediction', {})
                stats = result.get('statistics', {})
                
                prediction_data = {
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'folder': os.path.dirname(image_path),
                    'start_x': mean_pred.get('start_x'),
                    'start_y': mean_pred.get('start_y'),
                    'end_x': mean_pred.get('end_x'),
                    'end_y': mean_pred.get('end_y'),
                    'skewer_x': mean_pred.get('skewer_x'),
                    'skewer_y': mean_pred.get('skewer_y'),
                    'direction_degrees': mean_pred.get('direction_degrees'),
                    'total_variance': stats.get('total_variance'),
                    'centroid_variance': stats.get('centroid_variance'),
                    'distance_variance': stats.get('distance_variance'),
                    'size_variance': stats.get('size_variance'),
                    'num_samples': stats.get('num_samples'),
                    'input_action': result.get('input_action'),
                    'input_food_type': result.get('input_food_type'),
                    'input_text': result.get('input_text'),
                    'batch_size': result.get('batch_size'),
                    'enable_dropout': result.get('enable_dropout')
                }
                
                all_results.append(prediction_data)
                print(f"  ✓ Success - Total variance: {stats.get('total_variance', 'N/A'):.6f}")
            else:
                print(f"  ✗ API returned success=False")
                failed_images.append({'image_path': image_path, 'error': 'API returned success=False'})
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_images.append({'image_path': image_path, 'error': str(e)})
            continue
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({
            'results': all_results,
            'failed_images': failed_images,
            'summary': {
                'total_images': len(image_files),
                'successful': len(all_results),
                'failed': len(failed_images)
            }
        }, f, indent=2)
    print(f"\n[INFO] Saved JSON results to: {json_path}")
    
    # Save as CSV
    if all_results:
        csv_path = output_path.with_suffix('.csv')
        fieldnames = [
            'image_name', 'image_path', 'folder',
            'start_x', 'start_y', 'end_x', 'end_y',
            'skewer_x', 'skewer_y', 'direction_degrees',
            'total_variance', 'centroid_variance', 'distance_variance', 'size_variance',
            'num_samples', 'input_action', 'input_food_type', 'input_text',
            'batch_size', 'enable_dropout'
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)
        print(f"[INFO] Saved CSV results to: {csv_path}")
    
    # Create variance histograms
    if all_results:
        output_basename = output_path.stem
        create_variance_histograms(all_results, output_path.parent, output_basename)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Successful: {len(all_results)}")
    print(f"Failed: {len(failed_images)}")
    if failed_images:
        print(f"\nFailed images:")
        for failed in failed_images:
            print(f"  - {failed['image_path']}: {failed['error']}")
    print(f"{'='*60}")
    
    return all_results, failed_images

def main():
    parser = argparse.ArgumentParser(
        description='Batch process images using RT-1 API and store predictions'
    )
    parser.add_argument('--api_url', required=True, help='API server URL')
    parser.add_argument('--input_dir', required=True, help='Directory containing images or folders')
    parser.add_argument('--output_file', required=True, help='Output file path (JSON/CSV will be created)')
    parser.add_argument('--action', help='Action type (skewering, scooping, twirling)')
    parser.add_argument('--food_type', help='Food type (chicken, banana, etc.)')
    parser.add_argument('--text', default='Skewer the food item', help='Text instruction')
    parser.add_argument('--recursive', action='store_true', 
                       help='Recursively process images in subdirectories')
    
    args = parser.parse_args()
    
    try:
        process_images(
            api_url=args.api_url,
            input_dir=args.input_dir,
            output_file=args.output_file,
            action=args.action,
            food_type=args.food_type,
            text_instruction=args.text,
            recursive=args.recursive
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())


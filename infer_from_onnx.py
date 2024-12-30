#!/usr/bin/env python3
import cv2
from model import CustomDataset
import argparse
import onnxruntime as ort
from calculate_statistics import calculate_statistics
from edit_image import edit_image
from pathlib import Path
from typing import List

# Notes
# Importing CustomDataset from model takes time due to torch. In the future, use a saved JSON of image_parameters dict.
# CustomDataset is only imported in order to get min and max values from image_parameters dict.

def infer(image, onnx_file):
    # Prepare model input (the image stats)
    image_stats = calculate_statistics(image)
    dataset = CustomDataset()
    model_input = [dataset.normalize_full(image_stats, to_numpy=True)]  # Use normalize_full to get all 21 features

    # Onnx Runtime Session
    ort_sess = ort.InferenceSession(onnx_file)
    input_name = ort_sess.get_inputs()[0].name
    output_name = [ort_sess.get_outputs()[0].name]

    # Inference
    output = ort_sess.run(output_name, {input_name: model_input})
    # De-normalize
    output = dataset.denormalize(output[0][0])

    # Edit the image based on output of the model
    output_image = edit_image(image, output)
    return output_image

def get_image_files(input_path: Path) -> List[Path]:
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    return [p for p in input_path.iterdir() if p.suffix.lower() in supported_extensions and p.is_file()]

def process_image(image_path: Path, output_dir: Path, onnx_file: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Unable to read image {image_path}. Skipping.")
        return
    output_image = infer(image, str(onnx_file))
    output_filename = output_dir / f"{image_path.stem}_out{image_path.suffix}"
    cv2.imwrite(str(output_filename), output_image)
    print(f"Processed {image_path} -> {output_filename}")

if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser("Inference using saved ONNX model")
    parser.add_argument('-m', '--model', required=True, help="Path to .onnx model")
    parser.add_argument('-i', '--input', required=True, help="Path to input image or folder")
    parser.add_argument('-o', '--output', help="Output image path or folder")
    args = parser.parse_args()

    onnx_file = Path(args.model)
    input_path = Path(args.input)

    if not onnx_file.is_file():
        print(f"Error: The model file {onnx_file} does not exist.")
        exit(1)

    if input_path.is_file():
        # Single image inference
        if args.output:
            output_path = Path(args.output)
            # If output is a directory, ensure it exists
            if output_path.is_dir():
                print("Error: When input is a file, output should be a file path, not a directory.")
                exit(1)
        else:
            output_path = input_path.with_stem(input_path.stem + '_out')
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Infer
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Error: Unable to read image {input_path}.")
            exit(1)
        output_image = infer(image, str(onnx_file))
        cv2.imwrite(str(output_path), output_image)
        print(f"Processed {input_path} -> {output_path}")
    elif input_path.is_dir():
        # Batch processing
        image_files = get_image_files(input_path)
        if not image_files:
            print(f"No supported image files found in directory {input_path}.")
            exit(1)
        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path.cwd() / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)
        # Process each image
        for img_path in image_files:
            process_image(img_path, output_dir, onnx_file)
    else:
        print(f"Error: The input path {input_path} is neither a file nor a directory.")
        exit(1)

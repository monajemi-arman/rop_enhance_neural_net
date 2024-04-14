#!/usr/bin/environ python3
import cv2
from model import CustomDataset
import argparse
import onnxruntime as ort
from calculate_statistics import calculate_statistics
from edit_image import edit_image
from pathlib import Path
import pickle


# Notes
# Importing CustomDataset from model takes time due to torch. In the future, use a saved JSON of image_parameters dict.
# CustomDataset is only imported in order to get min and max values from image_parameters dict.

def infer(image, onnx_file):
    # Prepare model input (the image stats)
    image_stats = calculate_statistics(image)
    dataset = CustomDataset()
    model_input = [dataset.normalize(image_stats, to_numpy=True)]

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


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser("Inference using saved ONNX model")
    parser.add_argument('-m', '--model', required=True, help="Path to .onnx of model")
    parser.add_argument('-i', '--image', required=True, help="Path to input image")
    parser.add_argument('-o', '--output', help="Output image path")
    args = parser.parse_args()
    onnx_file = args.model
    image_path = args.image
    if args.output:
        output_path = args.output
    else:
        image_path = Path(image_path)
        output_path = image_path.with_stem(image_path.stem + '_out')
        # cv2 doesn't like Path objects as filename apparently
        image_path, output_path = str(image_path), str(output_path)
    # Infer
    image = cv2.imread(image_path)
    output_image = infer(image, onnx_file)
    cv2.imwrite(output_path, output_image)

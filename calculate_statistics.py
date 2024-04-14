#!/usr/bin/env python3
# This script extracts image statistics

import cv2
import os
from PIL import Image
import numpy as np
from pathlib import Path


def calculate_statistics(image, window_size=3):
    '''
    image -> the result of cv2.imread result OR filename
    '''
    # Load image if filename is passed
    if isinstance(image, (str, Path)):
        image = cv2.imread(image)
    # --- Create Mask --- #
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 45, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    # Draw the contours on the mask with white color
    mask = cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    # --- End of Create Mask --- #

    image_data = np.array(image)
    mask_data = np.array(mask) // 255
    # Calculate Contrast Index
    white = 255
    sum_min = 0
    sum_max = 0
    number_of_windows = 0

    image_bak, mask_bak = image, mask
    image, mask = np.array(image), np.array(mask)
    # Iterate over the image by windows of the specified size
    for row in range(0, image.shape[0], window_size):
        for col in range(0, image.shape[1], window_size):
            # Ensure that we don't exceed image dimensions
            end_row = min(row + window_size, image.shape[0])
            end_col = min(col + window_size, image.shape[1])

            image_window = image[row:end_row, col:end_col]
            mask_window = mask[row:end_row, col:end_col]

            # Consider only pixels that are white in the mask
            valid_pixels = image_window[mask_window == white]

            # If there are no valid pixels in this window, skip to the next
            if valid_pixels.size == 0:
                continue

            # Calculate minimum and maximum intensity values of valid pixels
            Imin = np.min(valid_pixels)
            Imax = np.max(valid_pixels)

            sum_min += Imin
            sum_max += Imax

            number_of_windows += 1

    # Calculate average min and max intensity values
    B = sum_min / number_of_windows if number_of_windows else 0
    R = sum_max / number_of_windows if number_of_windows else 0

    contrast_index = (R - B) / (R + B)

    # Calculate other stats...
    image, mask = image_bak, mask_bak
    # Ensure that the mask and image dimensions are consistent
    if mask_data.shape != image_data.shape[:2]:
        raise ValueError("The dimensions of the mask do not match the dimensions of the image.")

    output_statistics = {}

    for channel, color in zip(range(3), ['red', 'green', 'blue']):

        channel_data = image_data[:, :, channel] * mask_data

        # We should only consider non-zero elements (non-masked areas)
        masked_channel_data = channel_data[channel_data != 0]

        # Calculate the mode, standard deviation, and average for masked_channel_data
        if masked_channel_data.size > 0:
            unique_values, counts = np.unique(masked_channel_data, return_counts=True)
            mode_intensity = unique_values[np.argmax(counts)]
            mode_frequency = np.max(counts)

            frequency_average = np.mean(counts)
            frequency_std_dev = np.std(counts)

            average_intensity = np.mean(masked_channel_data)
            std_dev_intensity = np.std(masked_channel_data)

            output_statistics[color] = {
                'mode_intensity': mode_intensity,
                'mode_frequency': mode_frequency,
                'frequency_average': frequency_average,
                'frequency_std_dev': frequency_std_dev,
                'intensity_average': average_intensity,
                'standard_deviation': std_dev_intensity,
                'contrast_index': contrast_index
            }
        else:
            output_statistics[color] = {
                'mode_intensity': None,
                'mode_frequency': None,
                'frequency_average': None,
                'frequency_std_dev': None,
                'intensity_average': None,
                'standard_deviation': None,
                'contrast_index': None
            }

    return output_statistics

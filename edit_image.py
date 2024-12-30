#!/usr/bin/env python3
from skimage.restoration import estimate_sigma, denoise_wavelet
from skimage.util import random_noise
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


# Contrast and Brightness
# Contrast: < 1.0 >
def adjust_contrast(image, contrast):
    new_image = np.int16(image)
    new_image = new_image * contrast
    new_image = np.clip(new_image, 0, 255)
    new_image = np.uint8(new_image)
    return new_image


# Brightness
# 0 - 255
# brightness value would be the amount to increase or decrease each pixel value
def adjust_brightness(image, brightness=0):
    brightness = int(brightness)  # Model may not return int
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if brightness > 0:
        # Use cv2.add to handle overflow
        v = cv2.add(v, brightness)
    elif brightness < 0:
        # Use cv2.subtract to handle underflow
        v = cv2.subtract(v, abs(brightness))
    # If brightness == 0, no change

    hsv_new = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)


# Gamma
# < 1.0 >
def adjust_gamma(image, gamma=1.0):
    table = []
    for i in np.arange(0, 256):
        i /= 255.0
        i **= 1.0 / gamma
        table.append(i * 255.0)
    table = np.array(table, dtype=np.uint8)

    return cv2.LUT(image, table)


def adjust_cliplimit(image, cliplimit=1.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tileGridSize)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(image)
    L = clahe.apply(L)
    image = cv2.cvtColor(cv2.merge((L, A, B)), cv2.COLOR_LAB2BGR)
    return image


# Saturation
def adjust_saturation(image, saturation=0):
    cvImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensorImg = transforms.ToTensor()(cvImg)
    pic = F.adjust_saturation(tensorImg, saturation)
    pic_saturation = T.ToPILImage()(pic)

    numpy_image = np.array(pic_saturation)
    pic_saturation = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return pic_saturation


def adjust_strength(image, strength=0.0, sigma=1.0, kernel_size=(5, 5)):
    # Ignore small amounts
    if strength < 0.5:
        return image

    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(strength + 1) * image - float(strength) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))

    return sharpened.round().astype(np.uint8)


# Denoise
def adjust_denoise(image, denoise):
    # Ignore small denoise as no change
    if denoise == 0:
        return image

    image = image.astype('float32') / 255

    # Estimate noise standard deviation
    sigma_est = estimate_sigma(image, channel_axis=-1, average_sigmas=True)

    # Denoise the image
    denoised = denoise_wavelet(image, channel_axis=-1, convert2ycbcr=True,
                               method='VisuShrink', mode='soft',
                               sigma=sigma_est / denoise, rescale_sigma=True)

    # Clip pixel values to the valid range
    noisy_image = np.clip(image, 0, 1)
    denoised = np.clip(denoised, 0, 1)
    denoised = (denoised * 255.0).astype('uint8')

    return denoised


image_functions = {
    'contrast': adjust_contrast,
    'brightness': adjust_brightness,
    'gamma': adjust_gamma,
    'cliplimit': adjust_cliplimit,
    'strength': adjust_strength,
    'saturation': adjust_saturation
}


def edit_image(image, properties):
    for key, value in properties.items():
        if key in image_functions:
            image = image_functions[key](image, value)
        else:
            print(f"Warning: No function defined for key '{key}'. Skipping.")
    return image

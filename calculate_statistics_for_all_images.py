#!/usr/bin/env python3

import json
import os
from pathlib import Path
import sys
import pickle
from calculate_statistics import *

# Change these
original_gt = 'original_gt'  # Directory containing original ground truth in 'low' and 'high' directories
model_output_json = 'model_output.json'
model_input_pkl = 'model_input.pkl'

# Create model INPUT JSON
image_stats = []
# Find image for each directory
low_images = []
for fn in os.listdir(os.path.join(original_gt, 'low')):
    if Path(fn).suffix.lower() == '.jpg' and '_mask.' not in fn:
        low_images.append(fn)
# Process image
for fn in low_images:
    image_path = os.path.join(original_gt, 'low', fn)
    image_stats.extend([calculate_statistics(image_path)])

with open(model_input_pkl, 'wb') as fh:
    pickle.dump(image_stats, fh)


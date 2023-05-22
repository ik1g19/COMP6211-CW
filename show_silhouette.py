import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers
import numpy as np
import csv
import os
import argparse




x = 0.536  # @param {type:"slider", min:0, max:1, step:0.01}
y = 0.59  # @param {type:"slider", min:0, max:1, step:0.01}

BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (255, 255, 255)  # white

RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

# Create the options that will be used for InteractiveSegmenter
base_options = python.BaseOptions(model_asset_path='./magic_touch.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

OVERLAY_COLOR = (100, 100, 0)  # cyan


# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480




def write_to_csv(filename, data):
  with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

def append_to_csv(filename, data):
  with open(filename, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)






# Performs resizing and showing the image
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow("test", img)
  cv2.waitKey(10000)





def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px



def split_into_segments(filename):
  # Create the segmenter
  with python.vision.InteractiveSegmenter.create_from_options(options) as segmenter:

    # Create the MediaPipe Image
    image = mp.Image.create_from_file(filename)

    # Retrieve the category masks for the image
    roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                           keypoint=NormalizedKeypoint(x, y))
    segmentation_result = segmenter.segment(image, roi)
    category_mask = segmentation_result.category_mask

    # Convert the BGR image to RGB
    image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)

    # Create an overlay image with the desired color (e.g., (255, 0, 0) for red)
    overlay_image = np.zeros(image_data.shape, dtype=np.uint8)
    overlay_image[:] = OVERLAY_COLOR

    # Create the condition from the category_masks array
    alpha = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1

    # Create an alpha channel from the condition with the desired opacity (e.g., 0.7 for 70%)
    alpha = alpha.astype(float) * 1.0

    # Blend the original image and the overlay image based on the alpha channel
    #output_image = image_data * (1 - alpha) + overlay_image * alpha
    output_image = image_data * (1 - alpha)
    output_image = output_image.astype(np.uint8)

    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    _, output_image = cv2.threshold(output_image, 1, 255, cv2.THRESH_BINARY)

    # # Draw a white dot with black border to denote the point of interest
    # thickness, radius = 6, -1
    # keypoint_px = _normalized_to_pixel_coordinates(x, y, image.width, image.height)
    # cv2.circle(output_image, keypoint_px, thickness + 5, (0, 0, 0), radius)
    # cv2.circle(output_image, keypoint_px, thickness, (255, 255, 255), radius)

    # Find the coordinates of the non-zero pixels in the mask
    points = cv2.findNonZero(output_image)

    if points is not None:
      # Find the bounding box coordinates of the contours
      _x, _y, _w, _h = cv2.boundingRect(points)
      # Crop the image to the bounding box
      output_image = output_image[_y:_y + _h, _x:_x + _w]

      # Calculate the height of the top 1/10th portion
      height, width = output_image.shape
      crop_height = int(height / 6)
      # Crop the image to the top portion
      output_image = output_image[:crop_height, :]

    filename = filename.split('/')
    print(filename[-1])
    #cv2.imwrite('./silhouettes/head_test/' + filename[-1], output_image)

    return output_image





def hu_moments(img):
  silhouette = split_into_segments(img)

  # Calculate Moments
  moments = cv2.moments(silhouette)
  # Calculate Hu Moments
  huMoments = cv2.HuMoments(moments)
  huMoments = huMoments.flatten().tolist()
  return huMoments








huMoments = hu_moments('./data/training/016z052ps.JPG')
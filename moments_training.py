import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers
import numpy as np
import csv_utils
import os
import argparse
from scipy import interpolate




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








# Performs resizing and showing the image
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow("test", img)
  cv2.waitKey(10000)

def resize(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
  return img





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



def split_into_segments(filename, com_x, com_y):
  # Create the segmenter
  with python.vision.InteractiveSegmenter.create_from_options(options) as segmenter:
    # Create the MediaPipe Image
    image = mp.Image.create_from_file(filename)

    # Retrieve the category masks for the image
    roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                           keypoint=NormalizedKeypoint(com_x, com_y))
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

    #print(f'{image_file_name}:')
    #resize_and_show(output_image)


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

    return output_image





def hu_moments(img, com_x, com_y):
  silhouette = split_into_segments(img, com_x, com_y)

  # Calculate Moments
  moments = cv2.moments(silhouette)
  # Calculate Hu Moments
  huMoments = cv2.HuMoments(moments)
  huMoments = huMoments.flatten().tolist()

  return huMoments




def fourier_descriptors(img, com_x, com_y):
  silhouette = split_into_segments(img, com_x, com_y)

  # Find contours
  contours, _ = cv2.findContours(silhouette, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Number of points you want
  num_points = 100

  if contours:
    # Get the longest contour
    contour = max(contours, key=cv2.contourArea)

    # Reshape contour array and separate x and y coordinates
    contour = contour.squeeze()
    x = contour[:, 0]
    y = contour[:, 1]

    # Create parameterization variable t along the contour
    t = np.arange(len(contour))

    # Create interpolation functions, based on the existing points
    x_interpolate = interpolate.interp1d(t, x, kind='linear')
    y_interpolate = interpolate.interp1d(t, y, kind='linear')

    # Create new parameterization variable for the desired number of points
    tnew = np.linspace(0, len(contour) - 1, num_points)

    # Create new points
    new_points = np.column_stack((x_interpolate(tnew), y_interpolate(tnew)))

    # Combine the x and y coordinates of new points
    complex_contour = new_points[:, 0] + 1j * new_points[:, 1]

    # Perform Fourier Transform and get the Fourier Descriptors
    fourier_descriptors = np.fft.fft(complex_contour)

    # Calculate the absolute values of the Fourier descriptors
    fourier_descriptors_abs = np.abs(fourier_descriptors)

    # Convert to list
    fourier_descriptors_list = fourier_descriptors_abs.tolist()
    return fourier_descriptors_list

  else: return [0] * num_points







def main(arg1, arg2, arg3):
  training_dir = arg1
  csv_path = arg2
  front_first = int(arg3)

  data = csv_utils.data_from_csv_with_label(csv_path)

  if (front_first == 1): front = True
  else: front = False

  initial = True
  for index, filename in enumerate(os.listdir(training_dir)):
    if os.path.isfile(os.path.join(training_dir, filename)):
      # Process the file here

      path = training_dir + '/' + filename

      if not front:
        center_x = data[math.floor(index/2)][1]
        center_y = data[math.floor(index / 2)][2]

        huMoments = hu_moments(path, center_x, center_y)

        data[math.floor(index / 2)] = data[math.floor(index/2)][:1] + data[math.floor(index/2)][3:]
        data[math.floor(index/2)] = data[math.floor(index/2)] + huMoments

      front = not front

  csv_utils.write_to_csv(csv_path, data)





if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add command-line argument options
    parser.add_argument("arg1", help="Description of arg1")
    parser.add_argument("arg2", help="Description of arg2")
    parser.add_argument("arg3", help="Description of arg3")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.arg1, args.arg2, args.arg3)
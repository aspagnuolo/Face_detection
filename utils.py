import os
import cv2
import random
from skimage.feature import hog
from skimage import exposure
import numpy as np

def load_positive_images(positive_images_folder):
    """
    Load positive images from the specified folder.
    
    Args:
        positive_images_folder (str): The path to the folder containing positive images.
        
    Returns:
        list: A list of loaded positive images as NumPy arrays.
    """
    positive_images = []
    # Iterate over the files in the specified folder
    for filename in os.listdir(positive_images_folder):
        # Check if the file has a .jpg or .png extension
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image using OpenCV
            img = cv2.imread(os.path.join(positive_images_folder, filename))
            # Check if the image was read successfully
            if img is not None:
                # Append the loaded image to the list of positive_images
                positive_images.append(img)
    return positive_images


def load_negative_images(negative_images_folder):
    """
    Load negative images with '.jpg' extension from the specified folder.
    
    Args:
        negative_images_folder (str): The path to the folder containing negative images.
        
    Returns:
        list: A list of loaded negative images as NumPy arrays.
    """
    negative_images = []
    # Iterate over the files in the specified folder
    for filename in os.listdir(negative_images_folder):
        # Check if the file has a .jpg extension
        if filename.endswith(".jpg"):
            # Read the image using OpenCV
            img = cv2.imread(os.path.join(negative_images_folder, filename))
            # Check if the image was read successfully
            if img is not None:
                # Append the loaded image to the list of negative_images
                negative_images.append(img)
    return negative_images

def extract_random_patches(image, num_patches, patch_size):
    """
    Extract random patches from an image.
    
    Args:
        image (numpy.ndarray): The input image from which patches will be extracted.
        num_patches (int): The number of patches to extract.
        patch_size (tuple): A tuple representing the size of each patch in (height, width).
        
    Returns:
        list: A list of extracted patches as NumPy arrays.
    """
    # Get the height and width of the input image
    height, width = image.shape[:2]
    patches = []
    # Extract the specified number of random patches
    for _ in range(num_patches):
        # Generate random coordinates for the top-left corner of the patch
        x = random.randint(0, width - patch_size[0])
        y = random.randint(0, height - patch_size[1])    
        # Extract the patch using the generated coordinates
        patch = image[y:y + patch_size[1], x:x + patch_size[0]]      
        # Append the extracted patch to the list of patches
        patches.append(patch)
    return patches

def simulate_distance_variation(image, scale_factors):
    """
    Simulate distance variation by resizing an image with different scale factors.
    
    Args:
        image (numpy.ndarray): The input image to be resized.
        scale_factors (list): A list of scale factors to simulate distance variation.
        
    Returns:
        list: A list of resized images as NumPy arrays.
    """
    augmented_images = []
    # Resize the input image with different scale factors
    for scale in scale_factors:
        # Resize the image using OpenCV's resize function
        resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)  
        # Append the resized image to the list of augmented_images
        augmented_images.append(resized_image)
    return augmented_images

def add_mirror_images(images):
    """
    Add mirrored images to a list of images by flipping them horizontally.
    
    Args:
        images (list): A list of input images as NumPy arrays.
        
    Returns:
        list: A list containing the original images followed by their horizontally mirrored versions.
    """
    mirrored_images = [cv2.flip(img, 1) for img in images] 
    # Combine the original images with their mirrored versions
    augmented_images = images + mirrored_images
    return augmented_images

def resize_image(image, width, height):
    """
    Resize an image to the specified width and height.
    
    Args:
        image (numpy.ndarray): The input image to be resized.
        width (int): The desired width of the resized image.
        height (int): The desired height of the resized image.
        
    Returns:
        numpy.ndarray: The resized image as a NumPy array.
    """
    # Resize the image using OpenCV's resize function
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def get_hog_features(image, resize_width=64, resize_height=64, orient = 9, pix_per_cell = 8, cell_per_block = 2):
    """
    Extract HOG features from an image.
    
    Args:
        image (numpy.ndarray): The input image to extract HOG features from.
        
    Returns:
        numpy.ndarray: Extracted HOG features as a 1D array.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to the desired input size for HOG feature extraction
    resized_image = resize_image(gray, resize_width, resize_height)

    # Extract HOG features using the skimage hog function
    features = hog(resized_image, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys', transform_sqrt=True,
                   feature_vector=True)

    # Apply histogram equalization to improve feature visibility
    features = exposure.equalize_hist(features)
    return features

def sliding_window(image, stepSize, windowSize):
    """
    Generate sliding windows over an image.
    
    Args:
        image (numpy.ndarray): The input image to slide the windows over.
        stepSize (int): The step size for moving the sliding window.
        windowSize (tuple): A tuple representing the size of the sliding window in (width, height).
        
    Yields:
        tuple: A tuple containing the coordinates (x, y) of the top-left corner of the window
               and the cropped window as a NumPy array.
    """
    for y in range(0, image.shape[0] - windowSize[1], stepSize):
        for x in range(0, image.shape[1] - windowSize[0], stepSize):
            # Yield the window's top-left coordinates and the cropped window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def hard_negative_mining(svm, negative_images, stepSize, windowSize = (64, 64)):
    """
    Perform hard negative mining using a trained SVM classifier.
    
    Args:
        svm (object): A trained SVM classifier.
        negative_images (list): A list of negative images to mine hard negatives from.
        stepSize (int): The step size for moving the sliding window.
        
    Returns:
        list: A list of hard negative windows as NumPy arrays.
    """
    hard_negatives = [] 
    # Iterate over each negative image
    for image in negative_images:
        # Slide the window over the image
        for (x, y, window) in sliding_window(image, stepSize, windowSize):
            # Extract HOG features from the window
            features = get_hog_features(window)  
            # Make a prediction using the trained SVM
            pred = svm.predict([features])
            # If the prediction is positive, consider the window as a hard negative
            if pred[0] == 1:
                hard_negatives.append(window)
    return hard_negatives

def non_max_suppression(boxes, overlap_threshold):
    """
    Apply non-maximum suppression to a list of bounding boxes.
    
    Args:
        boxes (numpy.ndarray): An array of bounding boxes in [x1, y1, x2, y2] format.
        overlap_threshold (float): The threshold for considering two boxes as overlapping.
        
    Returns:
        numpy.ndarray: An array of selected bounding boxes after non-maximum suppression.
    """
    if len(boxes) == 0:
        return []
    # Convert bounding box coordinates to float
    boxes = boxes.astype("float")
    # Initialize the list of selected bounding boxes
    selected_boxes = []
    # Get coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # Calculate area of each bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # Sort the bounding box indexes based on y2 coordinate
    indexes = np.argsort(y2)
    while len(indexes) > 0:
        # Take the last bounding box in the index list and add it to selected_boxes
        last = len(indexes) - 1
        i = indexes[last]
        selected_boxes.append(i)
        # Calculate the intersection bounding box coordinates (maximum x1 and y1 among bounding boxes)
        xx1 = np.maximum(x1[i], x1[indexes[:last]])
        yy1 = np.maximum(y1[i], y1[indexes[:last]])
        xx2 = np.minimum(x2[i], x2[indexes[:last]])
        yy2 = np.minimum(y2[i], y2[indexes[:last]])
        # Calculate width and height of intersection
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # Calculate intersection area and overlap ratio with respect to original bounding boxes' areas
        overlap = (w * h) / areas[indexes[:last]]
        # Remove indexes of elements that have overlap higher than the threshold
        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
    # Return the selected bounding boxes
    return boxes[selected_boxes].astype("int")
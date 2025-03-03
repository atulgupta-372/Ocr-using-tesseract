import cv2
import pytesseract
import numpy as np

# Set the path for the Tesseract executable (Windows-only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update path if needed

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to make the text stand out
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Denoising (removes small noise that can confuse OCR)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)

    return denoised

def rotate_image(image, angle):
    # Get the image center
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the bounding box size after rotation to prevent clipping
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # Calculate new width and height to fit the rotated image
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to take into account the new dimensions
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation and return the rotated image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return rotated_image

def get_best_orientation(image):
    # Preprocess the image for better OCR detection
    preprocessed_image = preprocess_image(image)

    # Try rotating the image by 0, 90, 180, 270 degrees and check which gives the best OCR result
    angles = [0, 90, 180, 270]
    best_text = ""
    best_angle = 0
    best_confidence = 0

    for angle in angles:
        rotated_image = rotate_image(image, angle)
        text = pytesseract.image_to_string(rotated_image)

        # Compute the confidence for the current text recognition
        confidence = compute_text_confidence(text)

        # Check if this rotation gives better confidence
        if confidence > best_confidence:
            best_confidence = confidence
            best_text = text
            best_angle = angle

    # Return the best rotation and the corresponding text
    return best_angle, best_text

def compute_text_confidence(text):
    # For simplicity, let's assume the confidence is based on the length of the detected text
    # You can improve this by implementing a more sophisticated confidence measure if needed
    return len(text)

def show_image(image):
    # Display the corrected image
    cv2.imshow("Corrected Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to your disoriented image
image_path = 'C:/Users/DELL/Desktop/Atul/PyProject/Ocr/image/img8.jpeg'

# Read the image
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image.")
else:
    # Get the best orientation and text
    best_angle, best_text = get_best_orientation(image)

    # Rotate the image to the best orientation
    rotated_image = rotate_image(image, best_angle)

    # Show the best detected text and rotated image
    print("Detected Text: ", best_text)
    print(f"Best rotation angle: {best_angle} degrees")

    # Show the corrected image
    show_image(rotated_image)

    # Optionally, save the rotated image
    cv2.imwrite('rotated_image.jpg', rotated_image)

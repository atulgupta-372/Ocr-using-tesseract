import cv2
import pytesseract
import numpy as np

# Set the path for the Tesseract executable (Windows-only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update path if needed

def correct_orientation(image_path):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        return None

    # Get orientation and script detection data using pytesseract
    osd = pytesseract.image_to_osd(image)
    print("Orientation and Script Detection:", osd)

    # Extract rotation angle from the result (rotation angle is in degrees)
    angle = int(osd.split('\n')[1].split(':')[1].strip())

    # If the angle is not zero, rotate the image to correct orientation
    if angle != 0:
        print(f"Rotating image by {angle} degrees")
        image = rotate_image(image, angle)

    return image

def rotate_image(image, angle):
    # Get the image center
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the bounding box size of the rotated image to prevent clipping
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # Calculate new width and height to fit the rotated image
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to take into account the new dimensions
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation and keep the entire image by adjusting the bounding box
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return rotated_image

def extract_text(image):
    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(image)
    return text

def show_image(image):
    # Display the image
    cv2.imshow("Corrected Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to your disoriented image
image_path = 'C:/Users/DELL/Desktop/Atul/PyProject/Ocr/image/img8.jpeg'

# Correct the image orientation
corrected_image = correct_orientation(image_path)

# If the image was successfully corrected
if corrected_image is not None:
    # Extract text from the corrected image
    extracted_text = extract_text(corrected_image)
    print("Extracted Text:\n", extracted_text)

    # Show the corrected image
    show_image(corrected_image)

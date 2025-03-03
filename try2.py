import cv2
import pytesseract
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import threading
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Application")
        self.root.geometry("400x300")
        
        self.video_thread = None
        self.capture = None
        self.running = False

        self.start_button = tk.Button(self.root, text="Start Webcam", command=self.start_webcam)
        self.start_button.pack(pady=10)

        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.quit_button = tk.Button(self.root, text="Quit", command=self.quit)
        self.quit_button.pack(pady=10)

    def start_webcam(self):
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        self.video_thread = threading.Thread(target=self.capture_from_webcam)
        self.video_thread.start()

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if file_path:
            self.process_image(file_path)

    def capture_from_webcam(self):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open the webcam.")
            return
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
            cv2.imshow("OCR (Webcam)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Error", "Could not read the image.")
            return
        self.process_frame(image)
        cv2.imshow("OCR (Image)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_frame(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(threshold_img)
        print("Detected Text:", text)

        h, w, _ = image.shape
        boxes = pytesseract.image_to_boxes(threshold_img)
        for box in boxes.splitlines():
            b = box.split()
            x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(image, (x1, h - y1), (x2, h - y2), (0, 255, 0), 2)
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def stop(self):
        self.running = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join()
        cv2.destroyAllWindows()
        self.start_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def quit(self):
        self.running = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join()
        cv2.destroyAllWindows()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()

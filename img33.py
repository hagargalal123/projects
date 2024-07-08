import tkinter as tk
from tkinter import Toplevel, filedialog, ttk
import cv2 # type: ignore
from PIL import Image, ImageTk
import os
from matplotlib import figure
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class ScrollingImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Scrolling Image Processor")

        self.image = None
        self.original_image = None
        self.processed_image = None
        self.file_name_label = None  # Label to display the file name

        # Styling
        self.root.geometry("1200x600")
        self.root.configure(bg="#ECECEC")

        # Create main frame
        self.main_frame = tk.Frame(root, bg="#ECECEC")
        self.main_frame.pack(expand=True, fill="both")

        # Create canvas for scrolling
        self.canvas = tk.Canvas(self.main_frame, bg="#fff", width=1200, height=400, scrollregion=(0, 0, 2400, 800))
        self.canvas.pack(side=tk.LEFT, fill="both", expand=True)

        # Scrollbars
        self.y_scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.y_scrollbar.pack(side=tk.LEFT, fill="y")

        # Configure canvas scrolling
        self.canvas.config(yscrollcommand=self.y_scrollbar.set)

        # Create widgets inside canvas
        self.inner_frame = tk.Frame(self.canvas, bg="#fff")
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        self.header_label = tk.Label(
        self.inner_frame,
        text="Scrolling Image Processor",
        font=("Arial", 22, "italic bold"),  # Changed font family, size, and style
        bg="#285059",  # Changed background color
        fg="white",  # Changed text color to black
        pady=12,  # Adjusted padding on the y-axis
        padx=8,  # Adjusted padding on the x-axis
        borderwidth=3,  # Increased border width
        # Changed border relief style to 'raised'
        )
        self.header_label.pack(fill="x")
        
        self.canvas_image_original = tk.Canvas(self.inner_frame, bg="#fff", width=600, height=400,highlightthickness=2,highlightbackground="black")
        self.canvas_image_original.pack(side=tk.LEFT)

        self.canvas_image_processed = tk.Canvas(self.inner_frame, bg="#fff", width=600, height=400,highlightthickness=2,highlightbackground="black")
        self.canvas_image_processed.pack(side=tk.RIGHT)

        self.upload_button = tk.Button(self.inner_frame, text="Upload Image", command=self.upload_image, font=("Helvetica", 12), bg="#285059", fg="white")
        self.upload_button.pack(pady=5)
        # Options for image processing
        self.processing_options = ttk.Combobox(self.inner_frame, values=self.get_processing_options(), font=("Helvetica", 12))
        self.processing_options.pack(pady=1)
        # New Button to apply color-related operations
        self.change_color_button = tk.Button(self.inner_frame, text="Change Color", command=self.apply_color_operation, font=("Helvetica", 12), bg="#285059", fg="white")
        self.change_color_button.pack(pady=5)
        # New Combobox for color-related operations
        self.color_options = ttk.Combobox(self.inner_frame, values=self.get_color_options(), font=("Helvetica", 12))
        self.color_options.pack(pady=5)
        # New Button to apply color-brightness operations
        self.brightness_button = tk.Button(self.inner_frame, text="Change brightness", command=self.apply_color_brightness, font=("Helvetica", 12), bg="#285059", fg="white")
        self.brightness_button.pack(pady=5)
        # New Combobox for color-brightness operations
        self.color_brightness = ttk.Combobox(self.inner_frame, values=self.get_Change_brightness(), font=("Helvetica", 12))
        self.color_brightness.pack(pady=5)
        # New Button to apply morphology operations
        self.morphology_button = tk.Button(self.inner_frame, text="Morphology", command=self.apply_morphology_operation, font=("Helvetica", 12),bg="#285059", fg="white")
        self.morphology_button.pack(pady=5)
        # New Combobox for morphology operations
        self.morphology_options = ttk.Combobox(self.inner_frame, values=self.get_morphology_options(), font=("Helvetica", 12))
        self.morphology_options.pack(pady=5)
        
        # New Button to apply Smoothing operations
        self.Smoothing_button = tk.Button(self.inner_frame, text="Smoothing", command=self.apply_Smoothing_operation, font=("Helvetica", 12), bg="#285059", fg="white")
        self.Smoothing_button.pack(pady=5)
        # New Combobox for Smoothing operations
        self.Smoothing_options = ttk.Combobox(self.inner_frame, values=self.get_Smoothing_options(), font=("Helvetica", 12))
        self.Smoothing_options.pack(pady=5)
        
        # New Button to apply  Contrast operations
        self. Contrast_button = tk.Button(self.inner_frame, text=" Contrast", command=self.apply_Contrast_operation, font=("Helvetica", 12), bg="#285059", fg="white")
        self. Contrast_button.pack(pady=5)
        # New Combobox for Smoothing operations
        self. Contrast_options = ttk.Combobox(self.inner_frame, values=self.get_Contrast_options(), font=("Helvetica", 12))
        self. Contrast_options.pack(pady=5)

        # Entry fields for width and height
        self.width_label = tk.Label(self.inner_frame, text="Width:", font=("Helvetica", 12), bg="white", fg="black")
        self.width_label.pack(pady=(5, 0))

        self.width_entry = tk.Entry(self.inner_frame, font=("Helvetica", 12))
        self.width_entry.pack()

        self.height_label = tk.Label(self.inner_frame, text="Height:", font=("Helvetica", 12), bg="white", fg="black")
        self.height_label.pack(pady=(5, 0))

        self.height_entry = tk.Entry(self.inner_frame, font=("Helvetica", 12))
        self.height_entry.pack()

        # Process Image button
        self.process_button = tk.Button(self.inner_frame, text="Process Image", command=self.process_image, font=("Helvetica", 12), bg="#285059", fg="white")
        self.process_button.pack(side=tk.RIGHT, pady=15, padx=10)

        # Revert to Original button
        self.revert_button = tk.Button(self.inner_frame, text="Revert to Original", command=self.revert_to_original, font=("Helvetica", 12), bg="#285059", fg="white")
        self.revert_button.pack(pady=15)
        
        # Bind the Enter key to the process_image function
        root.bind('<Return>', lambda event: self.process_image())

    def get_processing_options(self):
        return [
            "Resizing",
            "Color Conversion (BGR to Grayscale)",
            "Color Channel Swapping from BGR to RGB",
            "Color Channel Swapping from BGR to GBR",
            "Image Complementing",
            "increase Brightness",
            "Decrease Brightness",
            "Sharpening",
            "Thresholding Segmentation (Global)",
            "Thresholding Segmentation (Adaptive)",
            "histogram gray scale",
            "histogram BGR"
        ]

    def get_color_options(self):
        return [
            "convert to red",
            "convert to green",
            "convert to blue",
            "convert to yellow",
            "convert to purple",
            "convert to cyan"
        ]
    def get_Change_brightness(self):
        return [
            "Change brightness to Red",
            "Change brightness to Green",
            "Change brightness to blue",
            "Change brightness to yellow",
            "Change brightness to purple",
            "Change brightness to cyan"
        ]
    def get_morphology_options(self):
        return [
            "Dilation",
            "Erosion",
            "Opening",
            "Closing",
        ]
    def get_Smoothing_options(self):
        return [
            "Average",
            "Median",
            "Max",
            "Min",
        ]
    def get_Contrast_options(self):
        return [
            "Equalization",
            "Stretching",

        ]
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.original_image = self.image.copy()
            self.display_image()

            # Display the file name in the label
            file_name = os.path.basename(file_path)
            self.file_name_label.config(text=f"Uploaded: {file_name}")

    def process_image(self):
        if self.image is None:
            return

        processed_image = self.image.copy()

        selected_option = self.processing_options.get()
        if selected_option:
            processed_image = self.apply_operation(selected_option)

        self.processed_image = processed_image
        self.display_processed_image()

    def apply_operation(self, operation):
        if self.image is None:
            return

        processed_image = self.image.copy()

        if operation == "Resizing":
            # Get width and height from entry fields
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())

            # Resize the image if valid width and height are provided
            if width > 0 and height > 0:
                processed_image = cv2.resize(processed_image, (width, height))
        elif operation == "Color Conversion (BGR to Grayscale)":
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        elif operation == "Color Channel Swapping from BGR to RGB":
            processed_image = processed_image[:, :, [2, 1, 0]]
        elif operation == "Color Channel Swapping from BGR to GBR":
            processed_image = processed_image[:, :, [1, 0, 2]]
        elif operation == "Image Complementing":
            processed_image = cv2.bitwise_not(processed_image)
        elif operation == "increase Brightness":
            brightness_increase = 50
            processed_image = cv2.convertScaleAbs(processed_image, alpha=1, beta=brightness_increase)
        elif operation == "Decrease Brightness":
            brightness_decrease = 30
            processed_image = cv2.convertScaleAbs(processed_image, alpha=1, beta=-brightness_decrease)
        elif operation == "Sharpening":
            processed_image = cv2.subtract(cv2.dilate( processed_image, None), cv2.erode( processed_image, None))
        elif operation == "Thresholding Segmentation (Global)":
            ret,processed_image = cv2.threshold(processed_image,175,255,cv2.THRESH_BINARY)
        elif operation == "Thresholding Segmentation (Adaptive)":
            gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            processed_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
        elif operation == "histogram BGR":   
            b, g, r = cv2.split(processed_image) 
            hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
            hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
            plt.figure()
            plt.plot(hist_b, color='blue', label='Blue')
            plt.plot(hist_g, color='green', label='Green')
            plt.plot(hist_r, color='red', label='Red')
            plt.title('Histogram for Color Channels')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
            new_window = Toplevel(root)
            new_window.title("Histogram")
            canvas = FigureCanvasTkAgg(FigureCanvasBase, master=new_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side='top', fill='both',expand=1)

        elif operation == "histogram gray scale":
            gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            # Plot the histogram
            plt.figure(figsize=(5, 4), dpi=100)
            plt.plot(hist, color='gray', label='Grayscale')
            plt.title('Histogram (Grayscale)')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

        # Display the plot in a separate window
            new_window = Toplevel(root)
            new_window.title("Histogram")
        canvas = FigureCanvasTkAgg(figure, master=new_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        return processed_image
        
    

    def apply_color_operation(self):
        if self.image is None:
            return

        processed_image = self.image.copy()

        selected_option = self.color_options.get()
        if selected_option:
            processed_image = self.apply_color(selected_option)

        self.processed_image = processed_image
        self.display_processed_image()

    def apply_color(self, color):
        if self.image is None:
            return

        processed_image = self.image.copy()

        if color == "convert to red":
            processed_image[:, :, 1] = 0
            processed_image[:, :, 0] = 0
        elif color == "convert to green":
            processed_image[:, :, 0] = 0
            processed_image[:, :, 2] = 0
        elif color == "convert to blue":
            processed_image[:, :, 1] = 0
            processed_image[:, :, 2] = 0
        elif color == "convert to yellow":
            processed_image[:, :, 0] = 0
        elif color == "convert to purple":
            processed_image[:, :, 1] = 0
        elif color == "convert to cyan":
            processed_image[:, :, 2] = 0

        return processed_image
    def apply_color_brightness(self):
        if self.image is None:
            return
        processed_image = self.image.copy()
        selected_option = self.color_brightness.get()
        if selected_option:
            processed_image = self.Change_brightness(selected_option)
            self.processed_image = processed_image
            self.display_processed_image()
    def Change_brightness(self, operation):
        if self.image is None:
                return
        processed_image = self.image.copy()
        if operation == "Change brightness to Red":
                brightness_increase = 50 
                processed_image[:, :, 2] = cv2.add(processed_image[:, :, 2], brightness_increase)
        elif operation == "Change brightness to Green":
                brightness_increase = 50 
                processed_image[:, :, 1] = cv2.add(processed_image[:, :, 1], brightness_increase)
        elif operation == "Change brightness to blue":
                brightness_increase = 50 
                processed_image[:, :, 0] = cv2.add(processed_image[:, :, 0], brightness_increase)
        elif operation == "Change brightness to yellow":
                brightness_increase = 50 
                processed_image[:, :, 2] = cv2.add(processed_image[:, :, 2], brightness_increase)
                processed_image[:, :, 1] = cv2.add(processed_image[:, :, 1], brightness_increase) 
        elif operation == "Change brightness to purple":
                brightness_increase = 50 
                processed_image[:, :, 2] = cv2.add(processed_image[:, :, 2], brightness_increase)
                processed_image[:, :, 0] = cv2.add(processed_image[:, :, 0], brightness_increase) 
        elif operation == "Change brightness to cyan":
                brightness_increase = 50 
                processed_image[:, :, 1] = cv2.add(processed_image[:, :, 1], brightness_increase)
                processed_image[:, :, 0] = cv2.add(processed_image[:, :, 0], brightness_increase) 

        return processed_image
    
    def apply_morphology_operation(self):
        if self.image is None:
            return

        processed_image = self.image.copy()

        selected_option = self.morphology_options.get()
        if selected_option:
            processed_image = self.apply_morphology(selected_option)

        self.processed_image = processed_image
        self.display_processed_image()

    def apply_morphology(self, operation):
        if self.image is None:
            return

        processed_image = self.image.copy()

        kernel_size = (5, 5)  # You can adjust the kernel size as needed

        if operation == "Dilation":
            processed_image = cv2.dilate(processed_image, None, iterations=1)
        elif operation == "Erosion":
            processed_image = cv2.erode(processed_image, None, iterations=1)
        elif operation == "Opening":
            processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size))
        elif operation == "Closing":
            processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size))

        return processed_image
    def apply_Smoothing_operation(self):
        if self.image is None:
            return

        processed_image = self.image.copy()

        selected_option = self.Smoothing_options.get()
        if selected_option:
            processed_image = self.apply_Smoothing(selected_option)

        self.processed_image = processed_image
        self.display_processed_image()

    def apply_Smoothing(self, operation):
        if self.image is None:
            return

        processed_image = self.image.copy()

        kernel_size = (5, 5)  # You can adjust the kernel size as needed

        if operation == "Average":
            processed_image = cv2.blur(processed_image, (3, 3)) 
        elif operation == " Median":
            processed_image = cv2.medianBlur(processed_image, 3)
        elif operation == "Max":
            processed_image = cv2.dilate(processed_image, np.ones((3, 3), np.uint8))        
        elif operation == "Min":
            processed_image = cv2.erode(processed_image, np.ones((3, 3), np.uint8))

        return processed_image
    def apply_Contrast_operation(self):
        if self.image is None:
            return

        processed_image = self.image.copy()

        selected_option = self.Contrast_options.get()
        if selected_option:
            processed_image = self.apply_Contrast(selected_option)

        self.processed_image = processed_image
        self.display_processed_image()

    def apply_Contrast(self, operation):
        if self.image is None:
            return

        processed_image = self.image.copy()

        kernel_size = (5, 5)  # You can adjust the kernel size as needed

        if operation == "Equalization":
            gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            processed_image = cv2.equalizeHist(gray_image)
        elif operation == "Stretching":
            processed_image = cv2.dilate(processed_image, np.ones((3, 3), np.uint8))
        return processed_image

    def revert_to_original(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.processed_image = None
            self.display_image()

            # Remove the file name label when reverting to the original image
            if self.file_name_label:
                self.file_name_label.destroy()

    def display_image(self):
        self.clear_canvas()

        if self.image is not None:
            image_original = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            image_original = Image.fromarray(image_original)
            image_original = ImageTk.PhotoImage(image=image_original)

            self.canvas_image_original.config(width=image_original.width(), height=image_original.height())
            self.canvas_image_original.create_image(0, 0, anchor=tk.NW, image=image_original)
            self.canvas_image_original.image = image_original

    def display_processed_image(self):
        if self.processed_image is not None:
            image_processed = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            image_processed = Image.fromarray(image_processed)
            image_processed = ImageTk.PhotoImage(image=image_processed)

            self.canvas_image_processed.config(width=image_processed.width(), height=image_processed.height())
            self.canvas_image_processed.create_image(0, 0, anchor=tk.NW, image=image_processed)
            self.canvas_image_processed.image = image_processed

    def clear_canvas(self):
        self.canvas_image_original.delete("all")
        self.canvas_image_processed.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    app = ScrollingImageProcessorApp(root)
    root.mainloop()

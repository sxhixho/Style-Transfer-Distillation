import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from classes.DataProcessor import DataProcessor

class ModelRunner:
    """
    Handles model inference and evaluation (comparison of outputs).
    """
    def __init__(self, data_processor=None):
        self.data_processor = data_processor if data_processor else DataProcessor()

    def compare_outputs(self, content_path, style_path, student_model, teacher_output_dir):
        print(f"Comparing outputs for {os.path.basename(content_path)} and {os.path.basename(style_path)}")
        content_img = self.data_processor.load_and_process_image(content_path)
        style_img = self.data_processor.load_and_process_image(style_path)
        if content_img is None or style_img is None:
            print("Error loading images for comparison.")
            return
        start = time.time()
        student_output = student_model.predict([content_img, style_img])
        student_time = time.time() - start
        student_output_deproc = self.data_processor.deprocess_image(student_output)
        c_name = os.path.splitext(os.path.basename(content_path))[0]
        s_name = os.path.splitext(os.path.basename(style_path))[0]
        teacher_filename = f"{c_name}_style_{s_name}.png"
        teacher_filepath = os.path.join(teacher_output_dir, teacher_filename)
        teacher_output = None
        if os.path.exists(teacher_filepath):
            teacher_output = np.array(Image.open(teacher_filepath))
        else:
            print(f"Teacher output not found: {teacher_filepath}")
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(self.data_processor.deprocess_image(content_img))
        plt.title("Content Image")
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(self.data_processor.deprocess_image(style_img))
        plt.title("Style Image")
        plt.axis('off')
        plt.subplot(1, 4, 3)
        if teacher_output is not None:
            plt.imshow(teacher_output)
            plt.title("Teacher Output")
        else:
            plt.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center')
            plt.title("Teacher Output N/A")
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(student_output_deproc)
        plt.title(f"Student Output ({student_time:.2f}s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    mr = ModelRunner()
    print("ModelRunner loaded. Test comparisons as needed.")

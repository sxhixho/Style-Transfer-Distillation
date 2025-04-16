import os
import time
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
from models.teacher_model import StyleTransferModel
from models.student_model import build_student_model

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['STYLE_FOLDER'] = 'static/styles'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load teacher and student models once
teacher = StyleTransferModel(num_iterations=500)
student = build_student_model()
student.load_weights('models/student_model.weights.h5')  # <- Make sure this exists

def get_model_size(model_path):
    size_bytes = os.path.getsize(model_path)
    return round(size_bytes / (1024 * 1024), 2)  # in MB

def preprocess_image(path, target_size=(512, 512)):
    img = teacher.load_and_process_image(path)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stylize', methods=['POST'])
def stylize():
    if 'content' not in request.files:
        return redirect(url_for('index'))

    content_file = request.files['content']
    style_name = request.form.get('style_name')

    if content_file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(content_file.filename)
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    content_file.save(content_path)

    style_path = os.path.join(app.config['STYLE_FOLDER'], style_name)

    # Preprocess inputs
    content_tensor = preprocess_image(content_path)
    style_tensor = preprocess_image(style_path)

    # -------- TEACHER INFERENCE --------
    start_t = time.time()
    teacher_output = teacher.stylize(content_tensor, style_tensor)
    end_t = time.time()
    teacher_time = round(end_t - start_t, 2)

    teacher_output_path = os.path.join(app.config['RESULT_FOLDER'], f'teacher_{filename}')
    Image.fromarray(teacher_output).save(teacher_output_path)

    # -------- STUDENT INFERENCE --------
    test_input = content_tensor[0] / 255.0
    test_input = np.expand_dims(test_input, axis=0)

    start_s = time.time()
    student_output = student.predict(test_input)[0]
    end_s = time.time()
    student_time = round(end_s - start_s, 2)

    student_output = (student_output * 255).astype('uint8')
    student_output_path = os.path.join(app.config['RESULT_FOLDER'], f'student_{filename}')
    Image.fromarray(student_output).save(student_output_path)

    # -------- Stats --------
    # teacher_size = get_model_size('models/teacher_model.weights.h5')
    student_size = get_model_size('models/student_model.weights.h5')

    teacher_loss = "N/A (Optimization-based)"
    student_loss = "Trained on MSE"

    return render_template('result.html',
        teacher_output='/' + teacher_output_path,
        student_output='/' + student_output_path,
        teacher_size=teacher_size,
        student_size=student_size,
        teacher_time=teacher_time,
        student_time=student_time,
        teacher_loss=teacher_loss,
        student_loss=student_loss
    )

if __name__ == '__main__':
    app.run(debug=True)

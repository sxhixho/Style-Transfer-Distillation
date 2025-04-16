# app.py
import os
import time
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf # Keep TF import for potential config or error handling
from style_transfer_logic import StyleTransferInference, deprocess_image

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key') # Use environment variable or default

# --- Configuration ---
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['STYLE_FOLDER'] = 'static/styles'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['CONTENT_EXAMPLES_FOLDER'] = 'static/content_examples'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# --- Model Configuration ---
STUDENT_MODEL_PATH = 'models\student_model30.keras' # Make sure this path is correct

# --- Ensure directories exist ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['STYLE_FOLDER'], exist_ok=True)
os.makedirs(app.config['CONTENT_EXAMPLES_FOLDER'], exist_ok=True)


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_predefined_styles():
    styles = []
    if os.path.exists(app.config['STYLE_FOLDER']):
        for filename in os.listdir(app.config['STYLE_FOLDER']):
            if allowed_file(filename):
                 # Create a display name (e.g., "Mosaic" from "mosaic.jpg")
                 display_name = os.path.splitext(filename)[0].replace('_', ' ').title()
                 styles.append({'filename': filename, 'display_name': display_name})
    return sorted(styles, key=lambda x: x['display_name'])


# --- Global Inference Runner ---
# Load models once on startup
try:
    inference_runner = StyleTransferInference(student_model_path=STUDENT_MODEL_PATH)
    print("Style Transfer Inference runner initialized successfully.")
except Exception as e:
    inference_runner = None
    print(f"FATAL: Could not initialize StyleTransferInference: {e}")
    # Depending on deployment, you might want the app to fail hard here
    # For development, allowing it to run might be okay, but inference will fail.

# --- Routes ---
@app.route('/')
def index():
    predefined_styles = get_predefined_styles()
    # Add example data if available
    example_content_file = "example_content.jpg"
    example_content_path = os.path.join(app.config['CONTENT_EXAMPLES_FOLDER'], example_content_file)
    has_example = os.path.exists(example_content_path)

    return render_template('index.html',
                           predefined_styles=predefined_styles,
                           has_example=has_example,
                           example_content_file=example_content_file)

@app.route('/stylize', methods=['POST'])
def stylize():
    if inference_runner is None:
        flash("Error: Style transfer model not loaded. Please check server logs.", "danger")
        return redirect(url_for('index'))

    start_total_time = time.time()

    # --- Get Form Data ---
    content_file = request.files.get('content_image')
    style_choice = request.form.get('style_choice') # 'predefined' or 'custom'
    predefined_style_name = request.form.get('predefined_style')
    custom_style_file = request.files.get('custom_style_image')
    try:
        teacher_iterations = int(request.form.get('teacher_iterations', 10))
        student_iterations = int(request.form.get('student_iterations', 1))
    except ValueError:
        flash("Invalid iteration count provided. Using defaults.", "warning")
        teacher_iterations = 10
        student_iterations = 1

    # Use example content if specified
    use_example_content = request.form.get('use_example_content') == 'true'
    example_content_file_from_form = request.form.get('example_content_filename')

    # --- Validate Inputs ---
    content_path = None
    style_path = None
    content_filename_secure = None
    style_filename_secure = None
    unique_id = str(uuid.uuid4())[:8] # Unique ID for output files

    # 1. Content Image
    if use_example_content and example_content_file_from_form:
        content_path = os.path.join(app.config['CONTENT_EXAMPLES_FOLDER'], example_content_file_from_form)
        if not os.path.exists(content_path):
             flash(f"Example content file '{example_content_file_from_form}' not found.", "danger")
             return redirect(url_for('index'))
        content_filename_secure = f"example_{secure_filename(example_content_file_from_form)}"
        print(f"Using example content: {content_path}")
    elif content_file and allowed_file(content_file.filename):
        content_filename_secure = f"content_{unique_id}_{secure_filename(content_file.filename)}"
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename_secure)
        try:
            content_file.save(content_path)
            print(f"Saved uploaded content: {content_path}")
        except Exception as e:
            flash(f"Error saving content file: {e}", "danger")
            return redirect(url_for('index'))
    else:
        flash("No valid content image provided.", "danger")
        return redirect(url_for('index'))

    # 2. Style Image
    if style_choice == 'predefined' and predefined_style_name:
        style_path = os.path.join(app.config['STYLE_FOLDER'], predefined_style_name)
        if not os.path.exists(style_path):
             flash(f"Selected predefined style '{predefined_style_name}' not found.", "danger")
             return redirect(url_for('index'))
        style_filename_secure = secure_filename(predefined_style_name)
        print(f"Using predefined style: {style_path}")
    elif style_choice == 'custom' and custom_style_file and allowed_file(custom_style_file.filename):
        style_filename_secure = f"style_{unique_id}_{secure_filename(custom_style_file.filename)}"
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename_secure)
        try:
            custom_style_file.save(style_path)
            print(f"Saved uploaded style: {style_path}")
        except Exception as e:
            flash(f"Error saving custom style file: {e}", "danger")
            return redirect(url_for('index'))
    else:
        flash("No valid style selected or provided.", "danger")
        # Clean up uploaded content file if style fails
        if content_path and not use_example_content and os.path.exists(content_path):
             try: os.remove(content_path)
             except OSError: pass
        return redirect(url_for('index'))

    # --- Perform Inference ---
    try:
        print(f"Running Teacher Inference (Iterations: {teacher_iterations})...")
        teacher_output_img, teacher_time, teacher_loss = inference_runner.run_teacher(
            content_path, style_path, iterations=teacher_iterations
        )
        teacher_output_filename = f"teacher_{unique_id}_{content_filename_secure.replace('content_', '')}"
        teacher_output_path_abs = os.path.join(app.config['RESULT_FOLDER'], teacher_output_filename)
        Image.fromarray(teacher_output_img).save(teacher_output_path_abs)
        teacher_output_path_rel = os.path.join(os.path.basename(app.config['RESULT_FOLDER']), teacher_output_filename).replace("\\", "/")
        print(f"Teacher output saved to: {teacher_output_path_abs}")

    except Exception as e:
        flash(f"Error during Teacher model inference: {e}", "danger")
        print(f"Teacher Inference Error: {e}")
        # Clean up files
        if content_path and not use_example_content and os.path.exists(content_path): os.remove(content_path)
        if style_path and style_choice == 'custom' and os.path.exists(style_path): os.remove(style_path)
        return redirect(url_for('index'))

    try:
        print(f"Running Student Inference (Iterations: {student_iterations})...")
        student_output_img, student_time, student_stats = inference_runner.run_student(
            content_path, style_path, iterations=student_iterations
        )
        student_output_filename = f"student_{unique_id}_{content_filename_secure.replace('content_', '')}"
        student_output_path_abs = os.path.join(app.config['RESULT_FOLDER'], student_output_filename)
        Image.fromarray(student_output_img).save(student_output_path_abs)
        student_output_path_rel = os.path.join(os.path.basename(app.config['RESULT_FOLDER']), student_output_filename).replace("\\", "/")
        print(f"Student output saved to: {student_output_path_abs}")

    except Exception as e:
        flash(f"Error during Student model inference: {e}", "danger")
        print(f"Student Inference Error: {e}")
         # Clean up files (including teacher output if it succeeded)
        if content_path and not use_example_content and os.path.exists(content_path): os.remove(content_path)
        if style_path and style_choice == 'custom' and os.path.exists(style_path): os.remove(style_path)
        if 'teacher_output_path_abs' in locals() and os.path.exists(teacher_output_path_abs): os.remove(teacher_output_path_abs)
        return redirect(url_for('index'))

    # --- Prepare Results ---
    # Teacher size is complex (VGG weights + optimization state). Let's report VGG base size.
    # VGG19 size is roughly 548 MB (weights). We'll use a placeholder.
    teacher_size = "VGG19 (approx 548 MB)"
    student_size = inference_runner.student_model_size # Get from the runner instance

    # Clean up uploaded files (optional, keep for debugging if needed)
    # if content_path and not use_example_content and os.path.exists(content_path): os.remove(content_path)
    # if style_path and style_choice == 'custom' and os.path.exists(style_path): os.remove(style_path)

    end_total_time = time.time()
    print(f"Total request processing time: {end_total_time - start_total_time:.2f} seconds")

    return render_template('result.html',
        teacher_output_path=url_for('static', filename=teacher_output_path_rel),
        student_output_path=url_for('static', filename=student_output_path_rel),
        teacher_size=teacher_size,
        student_size=f"{student_size} MB" if isinstance(student_size, (int, float)) else student_size,
        teacher_time=f"{teacher_time:.2f}",
        student_time=f"{student_time:.2f}",
        teacher_loss=f"{teacher_loss:.2f}" if isinstance(teacher_loss, (int, float)) else "N/A",
        student_loss=f"Stats: Min={student_stats['min']:.2f}, Max={student_stats['max']:.2f}, Mean={student_stats['mean']:.2f}, Std={student_stats['std']:.2f}",
        teacher_iterations=teacher_iterations,
        student_iterations=student_iterations
    )

# --- Main Execution ---
if __name__ == '__main__':
    # Set TF log level to suppress warnings (optional)
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # tf.get_logger().setLevel('ERROR')
    app.run(debug=True) # debug=True is helpful for development
    
    
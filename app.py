import os
import time
import uuid # For unique filenames
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
# Import the new runner class
from inference_runner import InferenceRunner

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
STYLE_FOLDER = 'static/styles'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
# IMPORTANT: Specify the correct path to your student model
STUDENT_MODEL_PATH = 'models\student_model27.keras' # Make sure this file exists!
TARGET_SIZE = (320, 320) # Match training or desired inference size

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STYLE_FOLDER'] = STYLE_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB upload limit
app.secret_key = 'your_very_secret_key' # Change this for production!

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['STYLE_FOLDER'], exist_ok=True) # Ensure styles folder exists

# --- Initialize Inference Runner ---
# Load models once when the app starts
try:
    inference_runner = InferenceRunner(
        student_model_path=STUDENT_MODEL_PATH,
        target_size=TARGET_SIZE,
        default_teacher_iterations=50, # Default for web app
        default_student_iterations=1
    )
except Exception as e:
    print(f"FATAL: Could not initialize InferenceRunner: {e}")
    # Handle this critical error appropriately - maybe exit or disable stylization
    inference_runner = None

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_unique_filename(filename):
    """Generates a unique filename to prevent overwrites."""
    ext = filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4()}.{ext}"
    return secure_filename(unique_name) # Sanitize further

# --- Routes ---
@app.route('/')
def index():
    # List available predefined styles
    try:
        predefined_styles = [f for f in os.listdir(app.config['STYLE_FOLDER'])
                             if os.path.isfile(os.path.join(app.config['STYLE_FOLDER'], f)) and allowed_file(f)]
    except FileNotFoundError:
        predefined_styles = []
        flash("Style directory not found.", "warning")

    return render_template('index.html', predefined_styles=predefined_styles)

@app.route('/stylize', methods=['POST'])
def stylize():
    if inference_runner is None:
        flash("Stylization service is currently unavailable.", "danger")
        return redirect(url_for('index'))

    start_total_time = time.time()

    # --- Form Data Retrieval ---
    if 'content_file' not in request.files:
        flash('No content file part', 'danger')
        return redirect(url_for('index'))

    content_file = request.files['content_file']
    style_mode = request.form.get('style_mode') # 'predefined' or 'custom'
    teacher_iterations = request.form.get('teacher_iterations', type=int, default=inference_runner.default_teacher_iterations)
    student_iterations = request.form.get('student_iterations', type=int, default=inference_runner.default_student_iterations)

    # Validate iterations (optional but good practice)
    teacher_iterations = max(1, min(teacher_iterations, 500)) # Example limits
    student_iterations = max(1, min(student_iterations, 10))

    # --- Input Validation ---
    if content_file.filename == '':
        flash('No selected content file', 'danger')
        return redirect(url_for('index'))

    if not allowed_file(content_file.filename):
        flash('Invalid content file type', 'danger')
        return redirect(url_for('index'))

    # --- File Handling ---
    content_filename_orig = secure_filename(content_file.filename)
    content_filename_unique = get_unique_filename(content_filename_orig)
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename_unique)
    content_file.save(content_path)
    print(f"Saved content image to: {content_path}")

    style_path = None
    style_filename_orig = None
    style_filename_unique = None

    if style_mode == 'predefined':
        style_name = request.form.get('predefined_style_name')
        if not style_name:
            flash('No predefined style selected', 'danger')
            # Clean up uploaded content file
            if os.path.exists(content_path): os.remove(content_path)
            return redirect(url_for('index'))
        style_path = os.path.join(app.config['STYLE_FOLDER'], secure_filename(style_name))
        if not os.path.exists(style_path):
            flash(f'Selected style "{style_name}" not found on server.', 'danger')
            if os.path.exists(content_path): os.remove(content_path)
            return redirect(url_for('index'))
        style_filename_orig = style_name
        # We need a copy of the style image in results for display consistency
        style_filename_unique = get_unique_filename(style_filename_orig)
        style_display_path = os.path.join(app.config['RESULT_FOLDER'], style_filename_unique)
        try:
             # Save a display version (resized by runner)
             style_disp_img = inference_runner.get_display_image(style_path)
             if style_disp_img is not None:
                  inference_runner.save_image(style_disp_img, style_display_path)
             else: # Fallback: copy original if processing fails
                  import shutil
                  shutil.copy(style_path, style_display_path)
                  print(f"Copied style image for display: {style_display_path}")

        except Exception as e:
             print(f"Error copying/processing style image for display: {e}")
             style_display_path = None # Indicate error


    elif style_mode == 'custom':
        if 'custom_style_file' not in request.files:
            flash('No custom style file part', 'danger')
            if os.path.exists(content_path): os.remove(content_path)
            return redirect(url_for('index'))

        custom_style_file = request.files['custom_style_file']
        if custom_style_file.filename == '':
            flash('No selected custom style file', 'danger')
            if os.path.exists(content_path): os.remove(content_path)
            return redirect(url_for('index'))

        if not allowed_file(custom_style_file.filename):
            flash('Invalid custom style file type', 'danger')
            if os.path.exists(content_path): os.remove(content_path)
            return redirect(url_for('index'))

        style_filename_orig = secure_filename(custom_style_file.filename)
        style_filename_unique = get_unique_filename(style_filename_orig)
        # Save custom style temporarily in uploads, or directly process if possible
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename_unique)
        custom_style_file.save(style_path)
        print(f"Saved custom style image to: {style_path}")
         # Also save a display version to results
        style_display_path = os.path.join(app.config['RESULT_FOLDER'], style_filename_unique)
        try:
             style_disp_img = inference_runner.get_display_image(style_path)
             if style_disp_img is not None:
                  inference_runner.save_image(style_disp_img, style_display_path)
             else: # Fallback copy
                  import shutil
                  shutil.copy(style_path, style_display_path)
                  print(f"Copied style image for display: {style_display_path}")
        except Exception as e:
             print(f"Error copying/processing style image for display: {e}")
             style_display_path = None

    else:
        flash('Invalid style mode selected', 'danger')
        if os.path.exists(content_path): os.remove(content_path)
        return redirect(url_for('index'))

    if not style_path: # Should not happen if logic above is correct, but safety check
        flash('Style image path could not be determined.', 'danger')
        if os.path.exists(content_path): os.remove(content_path)
        # Clean up potential custom style upload
        if style_mode == 'custom' and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], style_filename_unique)):
             os.remove(os.path.join(app.config['UPLOAD_FOLDER'], style_filename_unique))
        return redirect(url_for('index'))


    # --- Run Inference ---
    teacher_output_img = None
    teacher_time = 0
    student_output_img = None
    student_time = 0
    student_stats = None

    # Run Teacher
    try:
        teacher_output_img, teacher_time = inference_runner.run_teacher(
            content_path, style_path, iterations=teacher_iterations
        )
    except Exception as e:
        print(f"Error during Teacher inference: {e}")
        flash(f"Error during Teacher processing: {e}", "warning")

    # Run Student
    # Determine if student needs the style image based on its loaded architecture
    student_style_path_arg = style_path if inference_runner.student_expects_list_input else None
    try:
        student_output_img, student_time, student_stats = inference_runner.run_student(
            content_path, style_image_path=student_style_path_arg, iterations=student_iterations
        )
    except Exception as e:
        print(f"Error during Student inference: {e}")
        flash(f"Error during Student processing: {e}", "warning")


    # --- Save Results ---
    results_data = {
        'content_image_url': None,
        'style_image_url': None,
        'teacher_output_url': None,
        'student_output_url': None,
        'teacher_time': f"{teacher_time:.2f}" if teacher_time else "N/A",
        'student_time': f"{student_time:.2f}" if student_time else "N/A",
        'teacher_iterations': teacher_iterations,
        'student_iterations': student_iterations,
        'error': None
    }

    # Save display version of content image
    content_display_filename = f"content_{content_filename_unique}"
    content_display_path = os.path.join(app.config['RESULT_FOLDER'], content_display_filename)
    try:
        content_disp_img = inference_runner.get_display_image(content_path)
        if content_disp_img is not None:
            if inference_runner.save_image(content_disp_img, content_display_path):
                results_data['content_image_url'] = url_for('static', filename=f'results/{content_display_filename}')
        else:
            print(f"Could not process content image {content_path} for display.")
            # Fallback: Maybe try copying original? Less ideal due to size/format.
    except Exception as e:
        print(f"Error processing content image for display: {e}")

    # Style image URL (already processed/copied above)
    if style_display_path and os.path.exists(style_display_path):
         results_data['style_image_url'] = url_for('static', filename=f'results/{style_filename_unique}')
    else:
         # Try to use original style path if display version failed
         if style_mode == 'predefined':
              results_data['style_image_url'] = url_for('static', filename=f'styles/{style_filename_orig}')
         # Custom style URL is harder if saving failed

    # Save Teacher output
    if teacher_output_img is not None:
        teacher_filename = f"teacher_{content_filename_unique}"
        teacher_save_path = os.path.join(app.config['RESULT_FOLDER'], teacher_filename)
        if inference_runner.save_image(teacher_output_img, teacher_save_path):
            results_data['teacher_output_url'] = url_for('static', filename=f'results/{teacher_filename}')
        else:
            results_data['error'] = (results_data.get('error') or "") + " Failed to save Teacher output."

    # Save Student output
    if student_output_img is not None:
        student_filename = f"student_{content_filename_unique}"
        student_save_path = os.path.join(app.config['RESULT_FOLDER'], student_filename)
        if inference_runner.save_image(student_output_img, student_save_path):
            results_data['student_output_url'] = url_for('static', filename=f'results/{student_filename}')
        else:
            results_data['error'] = (results_data.get('error') or "") + " Failed to save Student output."

    # --- Cleanup (optional: remove original uploads) ---
    # You might want to keep originals for debugging or reprocessing
    # if os.path.exists(content_path):
    #     os.remove(content_path)
    # if style_mode == 'custom' and style_filename_unique and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], style_filename_unique)):
    #      os.remove(os.path.join(app.config['UPLOAD_FOLDER'], style_filename_unique)) # Remove temp custom style

    print(f"Total request processing time: {time.time() - start_total_time:.2f} seconds")
    return render_template('result.html', **results_data)


if __name__ == '__main__':
    if inference_runner is None:
         print("\nWARNING: InferenceRunner failed to initialize. The application might not function correctly.\n")
    # Set debug=False for production
    app.run(debug=True, host='0.0.0.0') # Listen on all interfaces for easier access in network
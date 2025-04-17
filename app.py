import os
import time
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from style_transfer_logic import StyleTransferInference, deprocess_image
import math 

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_dev_12345') # Use environment variable or default

# --- Configuration ---
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['STYLE_FOLDER'] = 'static/styles'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['CONTENT_EXAMPLES_FOLDER'] = 'static/content_examples'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# --- Model Configuration ---
STUDENT_MODEL_PATH = os.path.join('models', 'student_model27.keras') # Ensure this path is correct

# --- Ensure directories exist ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['STYLE_FOLDER'], exist_ok=True)
os.makedirs(app.config['CONTENT_EXAMPLES_FOLDER'], exist_ok=True)


# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_predefined_styles():
    """Gets list of styles with filenames and display names."""
    styles = []
    style_folder = app.config['STYLE_FOLDER']
    if os.path.exists(style_folder):
        for filename in os.listdir(style_folder):
            if allowed_file(filename) and os.path.isfile(os.path.join(style_folder, filename)):
                 # Create display name: remove ext, replace underscores/hyphens, title case
                 display_name = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
                 styles.append({'filename': filename, 'display_name': display_name})
    return sorted(styles, key=lambda x: x['display_name']) # Sort alphabetically by display name

def format_loss(loss_value):
    """Formats loss value, handling potential NaN or Inf."""
    if loss_value is None:
        return "N/A"
    try:
        f_loss = float(loss_value)
        if math.isnan(f_loss) or math.isinf(f_loss):
            return "Invalid" # Indicates an issue during calculation
        return f"{f_loss:.2f}"
    except (ValueError, TypeError):
        return "N/A" # Not a number

def format_stats(stats_dict):
    """Formats the student stats dictionary into a display string and a tooltip string."""
    if not isinstance(stats_dict, dict):
        return "N/A", "Statistics not available."
    try:
        # Detailed tooltip string
        tooltip = (f"Min: {stats_dict.get('min', 0):.2f}, "
                   f"Max: {stats_dict.get('max', 0):.2f}, "
                   f"Mean: {stats_dict.get('mean', 0):.2f}, "
                   f"Std Dev: {stats_dict.get('std', 0):.2f}")
        # Simple display string for the badge (can customize)
        display = "Details" # Concise label for the badge
        return display, tooltip
    except (TypeError, ValueError, KeyError):
         # Catch potential errors during formatting or key access
         return "Stats Error", "Error formatting statistics."


# --- Global Inference Runner ---
try:
    inference_runner = StyleTransferInference(student_model_path=STUDENT_MODEL_PATH)
    print("Style Transfer Inference runner initialized successfully.")
except Exception as e:
    inference_runner = None
    print(f"FATAL: Could not initialize StyleTransferInference: {e}")

# --- Routes ---
@app.route('/')
def index():
    """Renders the main upload page."""
    predefined_styles = get_predefined_styles()
    # Define the example content file expected
    example_content_file = "example_content.jpg" # Ensure this file exists in CONTENT_EXAMPLES_FOLDER
    example_content_path_abs = os.path.join(app.config['CONTENT_EXAMPLES_FOLDER'], example_content_file)
    has_example = os.path.exists(example_content_path_abs)

    return render_template('index.html',
                           predefined_styles=predefined_styles,
                           has_example=has_example,
                           example_content_file=example_content_file)

@app.route('/stylize', methods=['POST'])
def stylize():
    """Handles the image stylization request."""
    if inference_runner is None:
        flash("Error: Style transfer service is currently unavailable. Please try again later.", "danger")
        return redirect(url_for('index'))

    start_total_time = time.time()

    content_file = request.files.get('content_image')
    style_choice = request.form.get('style_choice')
    predefined_style_filename = request.form.get('predefined_style')
    custom_style_file = request.files.get('custom_style_image')
    try:
        teacher_iterations = int(request.form.get('teacher_iterations', 50))
        student_iterations = int(request.form.get('student_iterations', 1))
        teacher_iterations = max(5, min(teacher_iterations, 500))
        student_iterations = max(1, min(student_iterations, 5))  
    except ValueError:
        flash("Invalid iteration count provided. Using defaults.", "warning")
        teacher_iterations = 10
        student_iterations = 1

    # Check if using example content
    use_example_content = request.form.get('use_example_content') == 'true'
    example_content_file_from_form = request.form.get('example_content_filename')

    # --- Validate Inputs & Prepare Paths ---
    content_path_abs = None       # Absolute path to the content image used
    style_path_abs = None         # Absolute path to the style image used
    content_filename_secure = None # Secured base filename for content
    style_filename_secure = None   # Secured base filename for style
    original_content_path_rel = None # Relative path (from static/) for content template
    original_style_path_rel = None   # Relative path (from static/) for style template
    style_name_for_result = "N/A"    # Display name for style in result

    unique_id = str(uuid.uuid4())[:8] # Unique ID for this stylization request

    # --- 1. Content Image Handling ---
    if use_example_content and example_content_file_from_form:
        # Using example content
        content_path_abs = os.path.join(app.config['CONTENT_EXAMPLES_FOLDER'], example_content_file_from_form)
        if not os.path.exists(content_path_abs):
             flash(f"Error: Example content file '{example_content_file_from_form}' not found.", "danger")
             return redirect(url_for('index'))
        # Secure filename just in case, though it's predefined
        content_filename_secure = f"example_{secure_filename(example_content_file_from_form)}"
        # Relative path from 'static' folder for template URL
        original_content_path_rel = os.path.join(os.path.basename(app.config['CONTENT_EXAMPLES_FOLDER']), example_content_file_from_form).replace("\\", "/")
        print(f"Using example content: {content_path_abs}")
    elif content_file and content_file.filename and allowed_file(content_file.filename):
        # Using uploaded content
        content_filename_secure = f"content_{unique_id}_{secure_filename(content_file.filename)}"
        content_path_abs = os.path.join(app.config['UPLOAD_FOLDER'], content_filename_secure)
        try:
            content_file.save(content_path_abs)
            # Relative path from 'static' folder for template URL
            original_content_path_rel = os.path.join(os.path.basename(app.config['UPLOAD_FOLDER']), content_filename_secure).replace("\\", "/")
            print(f"Saved uploaded content: {content_path_abs}")
        except Exception as e:
            flash(f"Error saving uploaded content file: {e}", "danger")
            print(f"Error saving content file: {e}")
            return redirect(url_for('index'))
    else:
        # No valid content source provided
        if not use_example_content: # Only flash error if not trying to use example
             flash("Please select or upload a content image.", "danger")
        # Redirect regardless
        return redirect(url_for('index'))

    # --- 2. Style Image Handling ---
    if style_choice == 'predefined' and predefined_style_filename:
        style_path_abs = os.path.join(app.config['STYLE_FOLDER'], predefined_style_filename)
        if not os.path.exists(style_path_abs):
             flash(f"Error: Selected style '{predefined_style_filename}' not found.", "danger")
             # Clean up uploaded content file if it exists and isn't an example
             if content_path_abs and not use_example_content and os.path.exists(content_path_abs):
                 try: os.remove(content_path_abs)
                 except OSError: pass # Ignore errors during cleanup
             return redirect(url_for('index'))
        style_filename_secure = secure_filename(predefined_style_filename)
        # Relative path from 'static' folder for template URL
        original_style_path_rel = os.path.join(os.path.basename(app.config['STYLE_FOLDER']), style_filename_secure).replace("\\", "/")
        # Get display name for result
        style_name_for_result = os.path.splitext(style_filename_secure)[0].replace('_', ' ').replace('-', ' ').title()
        print(f"Using predefined style: {style_path_abs}")

    elif style_choice == 'custom' and custom_style_file and custom_style_file.filename and allowed_file(custom_style_file.filename):
        # Using custom uploaded style
        style_filename_secure = f"style_{unique_id}_{secure_filename(custom_style_file.filename)}"
        style_path_abs = os.path.join(app.config['UPLOAD_FOLDER'], style_filename_secure)
        try:
            custom_style_file.save(style_path_abs)
             # Relative path from 'static' folder for template URL
            original_style_path_rel = os.path.join(os.path.basename(app.config['UPLOAD_FOLDER']), style_filename_secure).replace("\\", "/")
            style_name_for_result = "Custom Uploaded Style"
            print(f"Saved uploaded style: {style_path_abs}")
        except Exception as e:
            flash(f"Error saving custom style file: {e}", "danger")
            print(f"Error saving custom style file: {e}")
            # Clean up uploaded content file if it exists and isn't an example
            if content_path_abs and not use_example_content and os.path.exists(content_path_abs):
                 try: os.remove(content_path_abs)
                 except OSError: pass
            return redirect(url_for('index'))
    else:
        # No valid style source provided
        if style_choice == 'predefined':
            flash("Please select a style from the gallery.", "danger")
        elif style_choice == 'custom':
             if not custom_style_file or not custom_style_file.filename:
                 flash("Please upload a custom style image.", "danger")
             else: # File provided but not allowed type
                 flash("Invalid file type for custom style. Allowed: JPG, PNG, GIF.", "danger")
        else:
            flash("Invalid style selection.", "danger") # Should not happen

        # Clean up uploaded content file if it exists and isn't an example
        if content_path_abs and not use_example_content and os.path.exists(content_path_abs):
             try: os.remove(content_path_abs)
             except OSError: pass
        return redirect(url_for('index'))

    # --- Perform Inference ---
    teacher_output_path_rel = None
    student_output_path_rel = None
    teacher_output_path_abs = None # Store absolute path for potential cleanup if student fails

    # --- 3. Teacher Model Inference ---
    try:
        print(f"Running Teacher Inference (Iterations: {teacher_iterations})...")
        teacher_output_img, teacher_time, teacher_loss = inference_runner.run_teacher(
            content_path_abs, style_path_abs, iterations=teacher_iterations
        )

        if teacher_output_img is None:
            raise ValueError("Teacher model returned None (no image data).")

        base_output_name = content_filename_secure.replace('content_', '').replace('example_', '')
        # Remove original extension if present, add .png
        base_output_name = os.path.splitext(base_output_name)[0]
        teacher_output_filename = f"teacher_{unique_id}_{base_output_name}.png" # Save as PNG

        teacher_output_path_abs = os.path.join(app.config['RESULT_FOLDER'], teacher_output_filename)
        Image.fromarray(teacher_output_img).save(teacher_output_path_abs)
        # Relative path from 'static' folder for use in url_for
        teacher_output_path_rel = os.path.join(os.path.basename(app.config['RESULT_FOLDER']), teacher_output_filename).replace("\\", "/")
        print(f"Teacher output saved to: {teacher_output_path_abs}")

    except Exception as e:
        flash(f"Error during Teacher (Quality) model processing: {e}", "danger")
        print(f"Teacher Inference Error: {e}")
        return redirect(url_for('index')) 


    # --- 4. Student Model Inference ---
    try:
        print(f"Running Student Inference (Iterations: {student_iterations})...")

        student_output_img, student_time, student_stats = inference_runner.run_student(
            content_path_abs, style_path_abs, iterations=student_iterations
        )

        if student_output_img is None:
             raise ValueError("Student model returned None (no image data).")

        # Create output filename based on content image name
        base_output_name = content_filename_secure.replace('content_', '').replace('example_', '')
        base_output_name = os.path.splitext(base_output_name)[0]
        student_output_filename = f"student_{unique_id}_{base_output_name}.png" # Save as PNG

        student_output_path_abs = os.path.join(app.config['RESULT_FOLDER'], student_output_filename)
        Image.fromarray(student_output_img).save(student_output_path_abs)
         # Relative path from 'static' folder for use in url_for
        student_output_path_rel = os.path.join(os.path.basename(app.config['RESULT_FOLDER']), student_output_filename).replace("\\", "/")
        print(f"Student output saved to: {student_output_path_abs}")

    except Exception as e:
        flash(f"Error during Student (Fast) model processing: {e}", "danger")
        print(f"Student Inference Error: {e}")
        pass 


    # --- Prepare Results for Template ---
    teacher_size = "VGG19 Base 548MB" # Simplified representation
    student_size_mb = inference_runner.student_model_size # Get from the runner instance

    student_stats_display, student_stats_tooltip = format_stats(student_stats if 'student_stats' in locals() else None)


    end_total_time = time.time()
    print(f"Total request processing time: {end_total_time - start_total_time:.2f} seconds")

    # --- Render Result Page ---
    # Pass relative paths (from static/) to url_for in the template
    return render_template('result.html',
        # Original Images (Ensure paths are valid relative paths or None)
        original_content_path=url_for('static', filename=original_content_path_rel) if original_content_path_rel else '#',
        original_style_path=url_for('static', filename=original_style_path_rel) if original_style_path_rel else '#',
        style_name=style_name_for_result,
        # Teacher Results (Check if paths exist before creating URL)
        teacher_output_path=url_for('static', filename=teacher_output_path_rel) if teacher_output_path_rel else '#',
        teacher_size=teacher_size,
        teacher_time=f"{teacher_time:.2f}" if 'teacher_time' in locals() else "N/A",
        teacher_iterations=teacher_iterations,
        teacher_loss=format_loss(teacher_loss) if 'teacher_loss' in locals() else "N/A",
        # Student Results (Check if paths exist before creating URL)
        student_output_path=url_for('static', filename=student_output_path_rel) if student_output_path_rel else '#',
        student_size=f"{student_size_mb:.1f} MB" if isinstance(student_size_mb, (int, float)) else student_size_mb,
        student_time=f"{student_time:.2f}" if 'student_time' in locals() else "N/A",
        student_iterations=student_iterations,
        student_loss_display=student_stats_display,
        student_loss_tooltip=student_stats_tooltip,
    )

# --- Main Execution ---
if __name__ == '__main__':
    placeholder_path = os.path.join('static', 'img_placeholder.png')
    if not os.path.exists(placeholder_path):
        try:
            img = Image.new('RGB', (150, 150), color = (50, 50, 55))
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
            except IOError:
                font = ImageFont.load_default()
            text = "Preview N/A"
            if hasattr(draw, 'textbbox'):
                 bbox = draw.textbbox((0, 0), text, font=font)
                 text_width = bbox[2] - bbox[0]
                 text_height = bbox[3] - bbox[1]
            else:
                 text_width, text_height = draw.textsize(text, font=font) # Deprecated

            text_x = (img.width - text_width) // 2
            text_y = (img.height - text_height) // 2
            draw.text((text_x, text_y), text, fill=(150, 150, 150), font=font)
            img.save(placeholder_path)
            print(f"Created placeholder image at {placeholder_path}")
        except Exception as e:
            print(f"Warning: Could not create placeholder image: {e}")

    app.run(debug=True, host='0.0.0.0', port=80)
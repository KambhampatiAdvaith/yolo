import os
import glob
import subprocess # To run ffmpeg commands
from ultralytics import YOLO
import shutil # For cleaning up directories

# --- Configuration ---
# Part (a): Image folder segmentation
IMAGE_INPUT_FOLDER = "my_local_images"  # Create this folder and put some .jpg/.png images in it
IMAGE_OUTPUT_FOLDER = "segmented_local_images" # Where segmented images will be saved

# Part (b & c): Video processing
# IMPORTANT: Find a SHORT (10-30 seconds), safe-for-work video. Long videos take lots of time/space.
VIDEO_URL = "https://youtube.com/shorts/X-tOlHJpiMc?si=aNIzzLLdd-yHAyFm"
# If VIDEO_URL is blank or download fails, the script will look for DOWNLOADED_VIDEO_NAME locally.
DOWNLOADED_VIDEO_NAME = "2099536-hd_1920_1080_30fps.mp4" # Name of video if downloaded or placed manually

RAW_FRAMES_FOLDER = "video_frames_raw"
SEGMENTED_FRAMES_FOLDER = "video_frames_segmented"
# FPS_EXTRACTION: How many frames per second to extract from the video.
# Start with 1 or 2. Higher values = more frames = longer processing.
FPS_EXTRACTION = 1

# Part (d): Output video
OUTPUT_VIDEO_NAME = "segmented_output_video.mp4"
# OUTPUT_VIDEO_FPS: FPS for the final segmented video.
# Often good to match FPS_EXTRACTION or the original video's FPS.
OUTPUT_VIDEO_FPS = FPS_EXTRACTION # Or a value like 15, 24, 30

# YOLO Model
MODEL_NAME = 'yolov8n-seg.pt' # Or yolov8s-seg.pt for better quality but slower

# --- Helper Functions ---

def create_folder_if_not_exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    # print(f"Folder '{folder_path}' ensured.") # Less verbose

def cleanup_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        # print(f"Cleaned up folder: {folder_path}") # Less verbose
    create_folder_if_not_exists(folder_path)

def download_video(url, output_path):
    print(f"Attempting to download video from: {url}")
    try:
        # Use a format specifier that tries to get a pre-merged mp4 or merges if possible
        subprocess.run([
            'yt-dlp',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/mp4', # Tries to get best mp4
            '-o', output_path,
            url
        ], check=True, capture_output=True, text=True) # capture_output and text=True for better error messages
        print(f"Video downloaded successfully to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video with yt-dlp: {e}")
        print(f"yt-dlp stdout: {e.stdout}")
        print(f"yt-dlp stderr: {e.stderr}")
        print("Ensure yt-dlp is installed and ffmpeg is in PATH (for merging formats).")
        print("You can also manually download the video and place it as 'input_video.mp4'.")
        return False
    except FileNotFoundError:
        print("Error: yt-dlp command not found. Is it installed and in your PATH?")
        return False

def extract_frames(video_path, output_folder, fps):
    print(f"Extracting frames from '{video_path}' to '{output_folder}' at {fps} FPS...")
    cleanup_folder(output_folder)
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps}',
        os.path.join(output_folder, 'frame_%05d.png') # %05d for up to 99999 frames, good practice
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Frames extracted successfully to {output_folder}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames with ffmpeg: {e}")
        print(f"ffmpeg stdout: {e.stdout}")
        print(f"ffmpeg stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("CRITICAL Error: ffmpeg command not found. Is it installed and in your PATH?")
        return False

def segment_image_list(image_files_list, output_save_dir_base, model):
    """ Segments a list of image files and saves them directly. """
    print(f"Segmenting {len(image_files_list)} images...")
    create_folder_if_not_exists(output_save_dir_base)

    for image_file in sorted(image_files_list): # sorted() is important for video frames
        # print(f"Processing: {image_file}") # Less verbose during bulk processing
        results = model.predict(source=image_file, save=False, verbose=False, stream=False)

        if results and results[0].masks is not None:
            base_name = os.path.basename(image_file)
            save_path = os.path.join(output_save_dir_base, base_name)
            results[0].save(filename=save_path)
            # print(f"Segmented image saved to: {save_path}") # Less verbose
        else:
            print(f"Warning: No segmentation results for {image_file}, or an error occurred.")
    print(f"Finished segmenting images. Output in: {output_save_dir_base}")


def create_video_from_frames(frames_folder, output_video_path, fps):
    print(f"Creating video from frames in '{frames_folder}' to '{output_video_path}' at {fps} FPS...")
    frames_pattern = os.path.join(frames_folder, 'frame_%05d.png')
    command = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', frames_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y', # Overwrite output
        output_video_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Video created successfully: {output_video_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video with ffmpeg: {e}")
        print(f"ffmpeg stdout: {e.stdout}")
        print(f"ffmpeg stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("CRITICAL Error: ffmpeg command not found. Is it installed and in your PATH?")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Loading YOLO model: {MODEL_NAME}")
    yolo_model = YOLO(MODEL_NAME)
    print("Model loaded.")

    # --- Part (a): Segment images from a local folder ---
    print("\n--- Starting Part (a): Segmenting images from local folder ---")
    create_folder_if_not_exists(IMAGE_INPUT_FOLDER)
    local_images = glob.glob(os.path.join(IMAGE_INPUT_FOLDER, '*.jpg')) + \
                   glob.glob(os.path.join(IMAGE_INPUT_FOLDER, '*.png')) + \
                   glob.glob(os.path.join(IMAGE_INPUT_FOLDER, '*.jpeg'))
    if local_images:
        cleanup_folder(IMAGE_OUTPUT_FOLDER) # Clean before processing
        segment_image_list(local_images, IMAGE_OUTPUT_FOLDER, yolo_model)
    else:
        print(f"No images found in {IMAGE_INPUT_FOLDER}. Please add some images.")
    print("--- Part (a) finished. ---")


    # --- Part (b), (c), (d): Video Processing ---
    print("\n--- Starting Part (b, c, d): Video Processing ---")
    video_ready = False
    if VIDEO_URL and VIDEO_URL != "YOUR_YOUTUBE_VIDEO_URL_HERE_OR_LEAVE_BLANK_TO_USE_LOCAL_VIDEO":
        if download_video(VIDEO_URL, DOWNLOADED_VIDEO_NAME):
            video_ready = True
    elif os.path.exists(DOWNLOADED_VIDEO_NAME):
        print(f"Using existing local video: {DOWNLOADED_VIDEO_NAME}")
        video_ready = True
    else:
        print(f"No video URL provided and '{DOWNLOADED_VIDEO_NAME}' not found locally.")
        print("Skipping video processing. Please set VIDEO_URL or place a video named 'input_video.mp4' in the script's directory.")

    if video_ready:
        if extract_frames(DOWNLOADED_VIDEO_NAME, RAW_FRAMES_FOLDER, FPS_EXTRACTION):
            extracted_frame_files = glob.glob(os.path.join(RAW_FRAMES_FOLDER, '*.png'))
            if extracted_frame_files:
                cleanup_folder(SEGMENTED_FRAMES_FOLDER) # Clean before processing
                segment_image_list(extracted_frame_files, SEGMENTED_FRAMES_FOLDER, yolo_model)
                create_video_from_frames(SEGMENTED_FRAMES_FOLDER, OUTPUT_VIDEO_NAME, OUTPUT_VIDEO_FPS)
            else:
                print(f"No frames found in {RAW_FRAMES_FOLDER} after extraction attempt.")
        else:
            print("Skipping further video processing due to frame extraction error.")
    else:
        print("Video not available for processing.")

    print("\n--- All tasks potentially finished. Check output folders. ---")
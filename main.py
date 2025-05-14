import ffmpeg
from PIL import Image, ImageEnhance
from faster_whisper import WhisperModel
import os
import glob
import subprocess
import re
from tqdm import tqdm
import shutil
import concurrent.futures
import threading
from io import StringIO
import sys
import random

device = "cpu"
model_size="tiny.en"
model = WhisperModel(model_size, device=device)

def apply_saturation(input_image_path, output_image_path, saturation=1.6):
    img = Image.open(input_image_path).convert("RGB")
    enhancer = ImageEnhance.Color(img)
    img_enhanced = enhancer.enhance(saturation)
    img_enhanced.save(output_image_path)

def generate_subtitles(audio_path):
    # Probe audio duration for progress bar
    probe = ffmpeg.probe(audio_path)
    total_duration = float(probe['format']['duration'])

    segments_gen, _ = model.transcribe(audio_path)
    subtitles = []
    last_end = 0
    with tqdm(total=total_duration, desc="Generating subtitles", unit="s", dynamic_ncols=True) as pbar:
        for seg in segments_gen:
            start = seg.start
            end = seg.end
            text = seg.text.strip().replace('\n', ' ')
            subtitles.append((start, end, text))
            pbar.update(round(max(0, end - last_end), 2))
            last_end = end
    print("Subtitles generation completed.\n")
    return subtitles

def write_srt(subtitles, srt_path):
    def format_time(seconds):
        ms = int((seconds - int(seconds)) * 1000)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(subtitles, 1):
            f.write(f"{i}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n")

def escape_ffmpeg_path(path):
    # Only escape backslashes for ffmpeg on Windows
    return path.replace('\\', '\\\\')

def create_video(image_path, audio_path, srt_path, output_path, duration=None):
    if duration is None:
        probe = ffmpeg.probe(audio_path)
        duration = float(probe['format']['duration'])

    cmd = [
        'ffmpeg',
        '-y',
        '-loop', '1',
        '-i', image_path,
        '-i', audio_path,
        '-vf', f'scale=1920:1080,subtitles={srt_path}',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ]

    print("Starting ffmpeg export...")
    # print("FFmpeg command:", " ".join(f'"{c}"' if ' ' in str(c) else str(c) for c in cmd))
    pbar = tqdm(total=duration, unit='s', desc='Rendering', dynamic_ncols=True)
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)

    time_pattern = re.compile(r'time=(\d+):(\d+):(\d+).(\d+)')
    last_seconds = 0
    stderr_lines = []

    for line in process.stderr:
        stderr_lines.append(line)
        match = time_pattern.search(line)
        if match:
            h, m, s, ms = map(int, match.groups())
            seconds = h * 3600 + m * 60 + s + ms / 100.0
            pbar.update(seconds - last_seconds)
            last_seconds = seconds

    process.wait()
    pbar.n = duration
    pbar.refresh()
    pbar.close()

    if process.returncode != 0:
        print("FFmpeg failed! Error output:")
        print("".join(stderr_lines))
    else:
        print("Export finished.")

# Thread-safe console output management
console_lock = threading.Lock()

def process_project(project_name, project_path, output_root):
    # Redirect stdout for this thread to capture progress
    thread_stdout = StringIO()
    original_stdout = sys.stdout
    sys.stdout = thread_stdout
    
    try:
        # Find image and audio file (supports common formats)
        image_files = glob.glob(os.path.join(project_path, "*.png")) + \
                    glob.glob(os.path.join(project_path, "*.jpg")) + \
                    glob.glob(os.path.join(project_path, "*.jpeg"))
        audio_files = glob.glob(os.path.join(project_path, "*.mp3")) + \
                    glob.glob(os.path.join(project_path, "*.wav")) + \
                    glob.glob(os.path.join(project_path, "*.m4a"))

        if not image_files or not audio_files:
            with console_lock:
                print(f"Skipping {project_name}: missing image or audio.")
            return

        # Setup paths with random numbers for concurrent processing
        random_id = str(random.randint(10000, 99999))
        image_path = os.path.abspath(image_files[0])
        audio_path = os.path.abspath(audio_files[0])
        saturated_image_path = os.path.abspath(f"saturated_{project_name}_{random_id}.png")
        srt_path = f"subtitles_{project_name}_{random_id}.srt"
        temp_output_video = f"{project_name}_{random_id}.mp4"  # Output in base folder
        final_output_video = os.path.join(output_root, f"{project_name}.mp4")

        with console_lock:
            print(f"\n[{project_name}] Starting processing")
        
        # Process steps
        apply_saturation(image_path, saturated_image_path)
        with console_lock:
            print(f"[{project_name}] Generating subtitles...")
        subtitles = generate_subtitles(audio_path)
        write_srt(subtitles, srt_path)
        create_video(saturated_image_path, audio_path, srt_path, temp_output_video)
        
        # Move to final destination
        os.makedirs(os.path.dirname(final_output_video), exist_ok=True)
        shutil.move(temp_output_video, final_output_video)
        
        # Clean up temporary files
        try:
            os.remove(saturated_image_path)
            os.remove(srt_path)
        except Exception as e:
            with console_lock:
                print(f"[{project_name}] Warning: Could not delete temp files: {e}")
        
        with console_lock:
            print(f"[{project_name}] Processing completed successfully")
        return True
    except Exception as e:
        with console_lock:
            print(f"[{project_name}] Error: {str(e)}")
        return False
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        
        # Print any captured output with project name prefix
        with console_lock:
            for line in thread_stdout.getvalue().splitlines():
                if line.strip():  # Only print non-empty lines
                    print(f"[{project_name}] {line}")

if __name__ == "__main__":
    input_root = os.path.abspath("input_files")
    output_root = os.path.abspath("output_files")
    os.makedirs(output_root, exist_ok=True)
    
    # Get list of project folders
    projects = []
    for project_name in os.listdir(input_root):
        project_path = os.path.join(input_root, project_name)
        if os.path.isdir(project_path):
            projects.append((project_name, project_path))
    
    print(f"Found {len(projects)} projects to process")
    
    # Process up to 5 projects concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(process_project, name, path, output_root): name 
            for name, path in projects
        }
        
        # As each project completes
        for future in concurrent.futures.as_completed(futures):
            project_name = futures[future]
            try:
                success = future.result()
                if success:
                    print(f"{project_name} completed")
                else:
                    print(f"{project_name} failed")
            except Exception as e:
                print(f"{project_name} raised an exception: {e}")
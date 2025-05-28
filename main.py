import os
import ffmpeg
from PIL import Image, ImageEnhance
from faster_whisper import WhisperModel
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
import ctranslate2
import json

ctranslate2.set_log_level(40)

with open("config.json", "r") as f:
    config = json.load(f)
    device_for_subtitile_generation = config.get("device_for_subtitile_generation", "cpu")
    if device_for_subtitile_generation != "auto" and device_for_subtitile_generation != "cpu" and device_for_subtitile_generation != "cuda":
        print(f"Invalid device_for_subtitile_generation value: {device_for_subtitile_generation}\nChoose between\n1. cpu\n2. cuda(for nvidia gpus)\n3. auto \nDefaulting to 'cpu'.")
        device_for_subtitile_generation = "cpu"
    image_saturation = config["image_saturation"]
    subtitle_colour_in_BGR_HEX_format = config["subtitle_colour_in_BGR_HEX_format"].replace("#", "")
    subtitle_outline_colour_in_BGR_HEX_format = config["subtitle_outline_colour_in_BGR_HEX_format"].replace("#", "")
    subtitle_font_size = config["subtitle_font_size"]
    subtitle_font = config["subtitle_font"]
    no_of_concurrent_generations = config["no_of_concurrent_generations"]
    no_of_chars_per_line = config["no_of_chars_per_line"]
    fade_in_duration = config["fade_in_duration_in_seconds"]
    merge_audio_files_in_each_project = config["merge_audio_files_in_each_project"]
    vertical_margin = int(config["subtitle_vertical_position_in_pixels"])
    max_subtitle_time = config.get("max_subtitle_time_in_secs", None)
    playback_speed = float(config["playback_speed"])
    if playback_speed == 0:
        playback_speed = 1.0
    if max_subtitle_time == 0:
        max_subtitle_time = None
    

model_size="tiny.en"
model = WhisperModel(model_size, device=device_for_subtitile_generation,compute_type="float32")

def apply_saturation(input_image_path, output_image_path, saturation=image_saturation):
    img = Image.open(input_image_path).convert("RGB")
    enhancer = ImageEnhance.Color(img)
    img_enhanced = enhancer.enhance(saturation)
    img_enhanced.save(output_image_path)

def generate_subtitles(audio_path, max_chars_per_segment=no_of_chars_per_line,max_subtitle_time=None):
    
    probe = ffmpeg.probe(audio_path)
    if max_subtitle_time is not None:
        print(f"Max subtitle time is set to {max_subtitle_time} seconds.")
    total_duration = float(probe['format']['duration']) if not max_subtitle_time else max_subtitle_time
    segments_gen, _ = model.transcribe(audio_path)
    raw_subtitles = []
    final_subtitles = []
    last_end = 0
    with tqdm(total=total_duration, desc="Generating subtitles", unit="s", dynamic_ncols=True) as pbar:
        for seg in segments_gen:
            start = seg.start
            end = seg.end
            # Skip segments that start after the limit
            if max_subtitle_time is not None and start >= max_subtitle_time:
                break
            # Trim segments that end after the limit
            if max_subtitle_time is not None and end > max_subtitle_time:
                end = max_subtitle_time
            text = seg.text.strip().replace('\n', ' ')
            raw_subtitles.append((start, end, text))
            pbar.update(round(max(0, end - last_end), 2))
            last_end = end
            
        for start, end, text in raw_subtitles:
            segment_duration = end - start
            
            if len(text) <= max_chars_per_segment:
                final_subtitles.append((start, end, text))
                continue
                
            words = text.split()
            current_segment = []
            current_length = 0
            current_start = start
            total_chars = len(text)
            
            for word in words:
                if current_length + len(word) + 1 > max_chars_per_segment:
                    segment_text = ' '.join(current_segment)
                    chars_ratio = len(segment_text) / total_chars
                    time_for_segment = segment_duration * chars_ratio
                    current_end = current_start + time_for_segment
                    
                    final_subtitles.append((current_start, current_end, segment_text))
                    
                    current_segment = [word]
                    current_length = len(word)
                    current_start = current_end
                else:
                    current_segment.append(word)
                    current_length += len(word) + 1 
            
            if current_segment:
                segment_text = ' '.join(current_segment)
                final_subtitles.append((current_start, end, segment_text))
    
    print("Subtitles generation completed.\n")
    return final_subtitles

def write_srt(subtitles, srt_path, speed=1.0):
    def format_time(seconds):
        ms = int((seconds - int(seconds)) * 1000)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(subtitles, 1):
            # Adjust timestamps based on playback speed
            adjusted_start = start / speed
            adjusted_end = end / speed
            f.write(f"{i}\n{format_time(adjusted_start)} --> {format_time(adjusted_end)}\n{text}\n\n")


def create_video(image_path, audio_path, srt_path, output_path, duration=None, speed=1.0):
    if duration is None:
        probe = ffmpeg.probe(audio_path)
        duration = float(probe['format']['duration'])
    
    # Calculate adjusted duration based on speed
    adjusted_duration = duration / speed
    
    # Generate atempo filters for speeds outside 0.5-2.0 range
    def generate_atempo_filters(target_speed):
        if target_speed == 1.0:
            return None
        
        filters = []
        remaining_speed = target_speed
        
        while remaining_speed > 2.0:
            filters.append("atempo=2.0")
            remaining_speed /= 2.0
        
        while remaining_speed < 0.5:
            filters.append("atempo=0.5")
            remaining_speed /= 0.5
        
        if remaining_speed != 1.0:
            filters.append(f"atempo={remaining_speed}")
        
        return ",".join(filters)
    
    # Set up video/audio filters for speed adjustment
    vf = f'fade=t=in:st=0:d={fade_in_duration},setpts={1/speed}*PTS,scale=1920:1080,subtitles={srt_path}:force_style=\'FontName={subtitle_font},FontSize={subtitle_font_size},PrimaryColour=&H{subtitle_colour_in_BGR_HEX_format}&,OutlineColour=&H{subtitle_outline_colour_in_BGR_HEX_format}&,Bold=0,Alignment=2,MarginV={vertical_margin}\''
    
    # Generate audio tempo filter chain
    af = generate_atempo_filters(speed)
    
    cmd = [
        'bin\\ffmpeg',
        '-y',
        '-loop', '1',
        '-i', image_path,
        '-i', audio_path,
        '-vf', vf
    ]
    
    # Add audio filter if needed
    if af:
        cmd.extend(['-af', af])
    
    cmd.extend([
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ])

    print("Starting ffmpeg export...")
    pbar = tqdm(total=adjusted_duration, unit='s', desc='Rendering', dynamic_ncols=True)
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)

    time_pattern = re.compile(r'time=(\d+):(\d+):(\d+).(\d+)')
    last_seconds = 0
    stderr_lines = []
    if process.stderr is None:
        print("Error: ffmpeg process has no stderr.")
        return
    for line in process.stderr:
        stderr_lines.append(line)
        match = time_pattern.search(line)
        if match:
            h, m, s, ms = map(int, match.groups())
            seconds = h * 3600 + m * 60 + s + ms / 100.0
            pbar.update(seconds - last_seconds)
            last_seconds = seconds

    process.wait()
    pbar.n = adjusted_duration
    pbar.refresh()
    pbar.close()

    if process.returncode != 0:
        print("FFmpeg failed! Error output:")
        print("".join(stderr_lines))
    else:
        print("Export finished.")

console_lock = threading.Lock()

def process_project(project_name, project_path, output_root):
    thread_stdout = StringIO()
    original_stdout = sys.stdout
    sys.stdout = thread_stdout
    
    try:
        # Search for images in the main folder
        image_files = glob.glob(os.path.join(project_path, "*.png")) + \
                      glob.glob(os.path.join(project_path, "*.jpg")) + \
                      glob.glob(os.path.join(project_path, "*.jpeg"))
        
        # If no images found, search recursively in subfolders
        if not image_files:
            image_files = glob.glob(os.path.join(project_path, "**", "*.png"), recursive=True) + \
                          glob.glob(os.path.join(project_path, "**", "*.jpg"), recursive=True) + \
                          glob.glob(os.path.join(project_path, "**", "*.jpeg"), recursive=True)

        audio_files = glob.glob(os.path.join(project_path, "*.mp3")) + \
                      glob.glob(os.path.join(project_path, "*.wav")) + \
                      glob.glob(os.path.join(project_path, "*.m4a"))

        if not image_files or not audio_files:
            with console_lock:
                print(f"Skipping {project_name}: missing image or audio.")
            return

        random_id = str(random.randint(10000, 99999))
        image_path = os.path.abspath(image_files[0])
        audio_path = os.path.abspath(audio_files[0])
        saturated_image_path = os.path.abspath(f"saturated_{project_name}_{random_id}.png")
        srt_path = f"subtitles_{project_name}_{random_id}.srt"
        temp_output_video = f"{project_name}_{random_id}.mp4"
        final_output_video = os.path.join(output_root, f"{project_name}.mp4")

        with console_lock:
            print(f"\n[{project_name}] Starting processing")
        
        apply_saturation(image_path, saturated_image_path)
        with console_lock:
            print(f"[{project_name}] Generating subtitles...")
        subtitles = generate_subtitles(audio_path, max_subtitle_time=max_subtitle_time)
        write_srt(subtitles, srt_path, speed=playback_speed)
        create_video(saturated_image_path, audio_path, srt_path, temp_output_video, speed=playback_speed)
        
        os.makedirs(os.path.dirname(final_output_video), exist_ok=True)
        shutil.move(temp_output_video, final_output_video)
        
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
        sys.stdout = original_stdout
        
        with console_lock:
            for line in thread_stdout.getvalue().splitlines():
                if line.strip(): 
                    print(f"[{project_name}] {line}")

if __name__ == "__main__":
    input_root = os.path.abspath("input_files")
    output_root = os.path.abspath("output_files")
    os.makedirs(output_root, exist_ok=True)
    
    projects = []
    for project_name in os.listdir(input_root):
        project_path = os.path.join(input_root, project_name)
        if os.path.isdir(project_path):
            projects.append((project_name, project_path))

            if merge_audio_files_in_each_project:
                audio_files = glob.glob(os.path.join(project_path, "*.mp3"))
                def extract_first_number(filename):
                    filename = os.path.basename(filename)
                    match = re.search(r'(\d+)', filename)
                    return int(match.group(1)) if match else float('inf')
                audio_files.sort(key=extract_first_number)
                # print(f"Sorted audio files: {audio_files}")
                if len(audio_files) > 1:
                    merged_audio_name = f"{project_name}.mp3"
                    merged_audio_path = os.path.join(project_path, f"{merged_audio_name}.mp3")
                    cmd = f"""cd "{project_path}" && ..\\..\\bin\\ffmpeg -y -i "concat:{'|'.join(audio_files)}" -acodec copy "{merged_audio_name}" """
                    subprocess.run(cmd, shell=True,stdout=subprocess.DEVNULL,
                                    # stderr=subprocess.DEVNULL,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
                    #clean_up the original audio files
                    for audio_file in audio_files:
                        if audio_file != merged_audio_path:
                            try:
                                os.remove(audio_file)
                            except Exception as e:
                                print(f"Warning: Could not delete {audio_file}: {e}")
                    print(f"Merged audio files into {merged_audio_path}")
    
    print(f"Found {len(projects)} projects to process")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=no_of_concurrent_generations) as executor:
        futures = {
            executor.submit(process_project, name, path, output_root): name 
            for name, path in projects
        }
        
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
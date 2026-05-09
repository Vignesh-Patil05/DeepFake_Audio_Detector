import os
import random
from moviepy import VideoFileClip

# Configuration using relative paths
BASE_DIR = os.getcwd()
INPUT_ROOT = os.path.join(BASE_DIR, 'datasets', 'FakeAVCeleb_v1.2')
OUTPUT_ROOT = os.path.join(BASE_DIR, 'datasets', 'FakeAVCeleb_audio')
# The .list files will be generated in the root directory as seen in your sidebar
LIST_DIR = BASE_DIR

# Create output directory for audio if it does not exist
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

all_data = []

print("Starting audio extraction and list generation...")

# Check if input directory exists
if not os.path.exists(INPUT_ROOT):
    print(f"Error: Could not find dataset at {INPUT_ROOT}")
    exit()

# Walk through the dataset structure
for folder in os.listdir(INPUT_ROOT):
    folder_path = os.path.join(INPUT_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue
    
    # Labeling logic based on folder names in FakeAVCeleb:
    # 1 for Fake Audio folders, 0 for Real Audio folders
    label = 1 if "FakeAudio" in folder else 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                
                # Maintain internal folder structure in the output
                rel_path = os.path.relpath(root, INPUT_ROOT)
                target_dir = os.path.join(OUTPUT_ROOT, rel_path)
                
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                
                audio_filename = file.replace(".mp4", ".wav")
                audio_path = os.path.join(target_dir, audio_filename)
                
                # Extract audio if it does not already exist
                if not os.path.exists(audio_path):
                    try:
                        clip = VideoFileClip(video_path)
                        # Convert to 16kHz Mono (standard for audio deepfake detection)
                        clip.audio.write_audiofile(
                            audio_path, 
                            fps=16000, 
                            nbytes=2, 
                            codec='pcm_s16le', 
                            ffmpeg_params=["-ac", "1"]
                        )
                        clip.close()
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        continue
                
                # Create entry for the .list files
                # Path is relative to the datasets/FakeAVCeleb_audio folder
                list_entry = f"{rel_path.replace(os.sep, '/')}/{audio_filename} {label}"
                all_data.append(list_entry)

# Shuffle and split data: 80 percent for training, 20 percent for testing
random.shuffle(all_data)
split_idx = int(len(all_data) * 0.8)
train_data = all_data[:split_idx]
test_data = all_data[split_idx:]

# Save the list files to the project root
with open(os.path.join(LIST_DIR, 'train.list'), 'w') as f:
    f.write("\n".join(train_data))

with open(os.path.join(LIST_DIR, 'test.list'), 'w') as f:
    f.write("\n".join(test_data))

print("Process complete.")
print(f"Audio files saved to: {OUTPUT_ROOT}")
print(f"Generated train.list and test.list in: {LIST_DIR}")
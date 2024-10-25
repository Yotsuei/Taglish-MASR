# data_preprocess.py
import os
import re
import pandas as pd
import librosa
import soundfile as sf

def clean_text(text):
    """Cleans text data by lowercasing and removing unwanted characters."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s']", '', text)
    return text.strip()

def preprocess_audio(audio_path, output_path, sample_rate=16000):
    """Loads, resamples, and saves audio at the target sample rate."""
    y, _ = librosa.load(audio_path, sr=sample_rate)
    sf.write(output_path, y, sample_rate)
    print(f"Processed audio saved to {output_path}")

def load_data(tsv_file, audio_dir, output_dir, max_samples=100):
    """Loads and preprocesses audio and transcripts from dataset."""
    audio_files = []
    transcripts = []
    count = 0

    try:
        print("Loading dataset...\n\n" + "=" * 50 + "\n")
        df = pd.read_csv(tsv_file, sep='\t').sample(frac=1).reset_index(drop=True)

        for index, row in df.iterrows():
            audio_file = row['path']
            if not audio_file.endswith(".mp3"):
                audio_file += ".mp3"
            transcript = row['sentence']

            # Clean the transcript
            cleaned_transcript = clean_text(transcript)
            transcripts.append(cleaned_transcript)

            # Preprocess audio
            input_audio_path = os.path.join(audio_dir, audio_file)
            output_audio_path = os.path.join(output_dir, f"{os.path.splitext(audio_file)[0]}.wav")
            os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
            preprocess_audio(input_audio_path, output_audio_path)
            audio_files.append(output_audio_path)

            count += 1
            if count >= max_samples:
                print(f"Finished loading and processing {count} audio files and transcripts.\n\n" + "=" * 50 + "\n")
                break

        return audio_files, transcripts
    except Exception as e:
        print(f"Error loading and processing data: {e}\n")
        return [], []

if __name__ == '__main__':
    tsv_file = 'data/training/train.tsv'  # Replace with actual path to TSV file
    audio_dir = 'data/training/'          # Directory containing the original audio files
    output_dir = 'data/training_processed/'  # Directory to save processed audio
    audio_files, transcripts = load_data(tsv_file, audio_dir, output_dir)

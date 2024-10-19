import os
import torchaudio
import pandas as pd
from transformers import Wav2Vec2ForCTC, WavLMForCTC, WhisperForConditionalGeneration, Wav2Vec2Processor, WhisperProcessor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import torch
import jiwer
import warnings
import numpy as np

# Ignore specific warnings
warnings.filterwarnings("ignore", message=".*transcription using a multilingual Whisper will default to language detection.*")
warnings.filterwarnings("ignore", message=".*Passing a tuple of `past_key_values` is deprecated.*")
warnings.filterwarnings("ignore", message=".*The attention mask is not set and cannot be inferred from input.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Define model and processor parameters
models = {
    "wavlm": {
        "model": None,
        "processor": None,
    },
    "whisper": {
        "model": None,
        "processor": None,
    },
    "wav2vec2": {
        "model": None,
        "processor": None,
    }
}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")  # Print the device being used
print("\n" + "=" * 50 + "\n")  # Divider

# Load models and processors
def load_models():
    try:
        print("Loading models and processors...\n")
        for model_name in models.keys():
            print(f"Loading {model_name}...\n")
            try:
                if model_name == "wav2vec2":
                    models[model_name]["model"] = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)
                    models[model_name]["processor"] = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
                elif model_name == "wavlm":
                    models[model_name]["model"] = WavLMForCTC.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-large").to(device)
                    models[model_name]["processor"] = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-large")
                elif model_name == "whisper":
                    models[model_name]["model"] = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").to(device)
                    models[model_name]["processor"] = WhisperProcessor.from_pretrained("openai/whisper-large")
                print(f"\n{model_name} loaded successfully.\n")
            except Exception as e:
                print(f"Error loading {model_name}: {e}\n")
        print("All models and processors loaded.\n\n" + "=" * 50 + "\n")
    except Exception as e:
        print(f"Error loading models: {e}\n")


# Load Common Voice dataset (TSV and mp3)
def load_common_voice_data(tsv_file, audio_dir, max_samples=5):
    audio_files = []
    transcripts = []
    count = 0  # Initialize a counter for processed files

    try:
        print("Loading dataset...\n\n" + "=" * 50 + "\n")

        # Step 1: Load TSV file
        df = pd.read_csv(tsv_file, sep='\t')

        # Shuffle the DataFrame to get random samples
        df = df.sample(frac=1).reset_index(drop=True)

        # Step 2: Process each row in the TSV file
        for index, row in df.iterrows():
            audio_file = row['path']
            if not audio_file.endswith(".mp3"):
                audio_file += ".mp3"

            transcript = row['sentence']  # Extract transcript from the 'sentence' column

            # Append to the list of files and transcripts
            audio_files.append(os.path.join(audio_dir, audio_file))
            transcripts.append(transcript)
            count += 1

            # Stop once max_samples have been loaded
            if count >= max_samples:
                print(f"Finished loading {count} audio files and transcripts from dataset.\n\n" + "=" * 50 + "\n")
                break

        return audio_files, transcripts
    except Exception as e:
        print(f"Error loading Common Voice data: {e}\n")
        return [], []  # Return empty lists on error

# Function to calculate evaluation metrics
def calculate_metrics(reference, hypothesis):
    # Normalize to lowercase
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    
    # Tokenize the transcriptions
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    # Calculate WER and CER
    wer = jiwer.wer(reference, hypothesis)
    cer = jiwer.cer(reference, hypothesis)
    
    # Calculate precision, recall, and F1 score
    true_positives = sum(1 for word in hypothesis_words if word in reference_words)
    false_positives = len(hypothesis_words) - true_positives
    false_negatives = len(reference_words) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate accuracy based on the number of correct words
    correct_predictions = sum(1 for i in range(min(len(reference_words), len(hypothesis_words))) if reference_words[i] == hypothesis_words[i])
    accuracy = correct_predictions / len(reference_words) if reference_words else 0

    return {
        "wer": wer,
        "cer": cer,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy
    }

# Pad or truncate audio input to a target length
def pad_or_truncate(array, target_length):
    current_length = array.shape[1]
    if current_length > target_length:
        return array[:, :target_length]
    elif current_length < target_length:
        pad_width = ((0, 0), (0, target_length - current_length))
        return np.pad(array, pad_width, mode='constant')
    else:
        return array

# Function to evaluate a model on Common Voice data
def evaluate_model(model, processor, audio_files, transcripts):
    results = {}
    total_loss = 0
    last_transcription = ""  # Store the last transcription
    last_audio_file = ""  # Store the last audio file

    # Initialize cumulative sums for metrics
    total_metrics = {
        "wer": 0,
        "cer": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "accuracy": 0
    }
    num_samples = len(audio_files)

    try:
        print(f"Evaluating {model.__class__.__name__}...\n\n" + "=" * 50 + "\n")

        for i, audio_file in enumerate(audio_files):
            try:
                # Load mp3 audio file
                audio, sample_rate = torchaudio.load(audio_file)

                # Resample if necessary
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    audio = resampler(audio)

                # Preprocess audio and move to device
                if isinstance(model, WhisperForConditionalGeneration):
                    input_features = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(device)
                else:
                    inputs = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                    input_values = inputs.input_values.to(device)

                    # Ensure consistent input size
                    target_length = 200000  # Adjust as needed
                    input_values = torch.tensor(pad_or_truncate(input_values.cpu().numpy(), target_length)).to(device)

                # Forward pass
                with torch.no_grad():
                    if isinstance(model, WhisperForConditionalGeneration):
                        output = model.generate(input_features, language='en')
                        transcription = processor.batch_decode(output, skip_special_tokens=True)[0]
                    else:
                        output = model(input_values)
                        logits = output.logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = processor.batch_decode(predicted_ids)[0]

                    loss = 0  # Loss not computed during inference
                    total_loss += loss

                # Calculate metrics
                metrics = calculate_metrics(transcripts[i], transcription)

                # Accumulate metrics
                for key in total_metrics:
                    total_metrics[key] += metrics[key]

                # Store the last audio file and its transcription
                last_audio_file = audio_file
                last_transcription = transcription

            except Exception as e:
                print(f"Error evaluating file {audio_file}: {e}\n\n" + "=" * 50 + "\n")
        
        print(f"Finished evaluating {model.__class__.__name__}.\nResults for the last audio file '{last_audio_file}': '{last_transcription}'\n\n" + "=" * 50 + "\n")
        
        # Calculate averages
        if num_samples > 0:
            avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}
            results = avg_metrics
        else:
            results = {key: 0 for key in total_metrics}  # Default to zeros if no samples were processed
    except Exception as e:
        print(f"Error evaluating model: {e}\n")

    results['last_transcription'] = last_transcription
    results['last_audio_file'] = last_audio_file
    return results

# Main function to run evaluation
# Main function to run evaluation
if __name__ == "__main__":
    load_models()  # Load models and processors
    audio_dir = "data/common-voice/clips"  # Set your audio directory path
    tsv_file = "data/common-voice/validated.tsv"  # Set your TSV file path
    audio_files, transcripts = load_common_voice_data(tsv_file, audio_dir, max_samples=5)

    # Evaluate each model
    results = {}
    for model_name in models.keys():
        results[model_name] = evaluate_model(models[model_name]["model"], models[model_name]["processor"], audio_files, transcripts)

    # Print results in the desired format
    for model_name, metrics in results.items():
        print(f"Model: {model_name}")
        if metrics:  # Check if metrics is not empty
            print(f"WER: {metrics.get('wer', 0):.4f}")  # Use .get to avoid KeyError
            print(f"CER: {metrics.get('cer', 0):.4f}")
            print(f"Precision: {metrics.get('precision', 0):.4f}")
            print(f"Recall: {metrics.get('recall', 0):.4f}")
            print(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
            print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"Last Transcription: '{metrics.get('last_transcription', 'N/A')}'")  # Print last transcription
            print(f"Last Audio File: '{metrics.get('last_audio_file', 'N/A')}'")  # Print last audio file
            print("\n" + "=" * 50 + "\n")
        else:
            print("Metrics could not be calculated.\n")

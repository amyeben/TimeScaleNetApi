from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import librosa
import os
import uvicorn
import uuid
import glob
import yt_dlp
import ffmpeg

app = FastAPI()

# Configuration CORS (autorise votre frontend, ici localhost:3000)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"  # à ajuster selon vos besoins
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paramètres communs
SAMPLE_RATE = 22050
N_MELS = 128

# Pour UrbanSound8K, le modèle attend 87 time steps
MAX_TIME_STEPS_US8K = 87  
# Pour ESC10, le modèle attend 200 time steps
MAX_TIME_STEPS_ESC10 = 200  

# ------------------------------------------------------------------------------
# Fonctions de prétraitement paramétrables
# ------------------------------------------------------------------------------

def get_mel_spectrogram(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        n_fft = min(2048, len(audio))
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=n_fft,
            hop_length=512
        )
        log_mel_spectrogram = np.log(mel_spectrogram + 1e-9)
        return log_mel_spectrogram.T  # forme (time_steps, n_mels)
    except Exception as e:
        print(f"Erreur lors du traitement du fichier {file_path}: {e}")
        return None

def pad_or_truncate(spectrogram, max_time_steps):
    if spectrogram.shape[0] > max_time_steps:
        return spectrogram[:max_time_steps, :]
    else:
        return np.pad(spectrogram, ((0, max_time_steps - spectrogram.shape[0]), (0, 0)), mode="constant")

def preprocess_audio(file_path, max_time_steps):
    mel_spec = get_mel_spectrogram(file_path)
    if mel_spec is None:
        return None
    mel_spec = pad_or_truncate(mel_spec, max_time_steps)
    mean = np.mean(mel_spec)
    std = np.std(mel_spec) + 1e-9
    mel_spec = (mel_spec - mean) / std
    mel_spec = np.expand_dims(mel_spec, axis=-1)  # ajoute le canal
    mel_spec = np.expand_dims(mel_spec, axis=0)     # ajoute la dimension batch
    return mel_spec

# ------------------------------------------------------------------------------
# Chargement des modèles et définition des labels
# ------------------------------------------------------------------------------

# Modèle ESC10
MODEL_PATH_ESC10 = "timescalenet_model.h5"  # chemin vers le modèle ESC10
model_esc10 = tf.keras.models.load_model(MODEL_PATH_ESC10)
LABELS_ESC10 = ['chainsaw', 'crackling_fire', 'dog', 'rain', 'sea_waves', 
                'clock_tick', 'crying_baby', 'helicopter', 'rooster', 'sneezing']

# Modèle UrbanSound8K
MODEL_PATH_US8K = "timescalenet_urbansound8k.h5"  # chemin vers le modèle UrbanSound8K
model_urbansound = tf.keras.models.load_model(MODEL_PATH_US8K)
LABELS_US8K = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
               "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

# ------------------------------------------------------------------------------
# Fonction commune de prédiction depuis un fichier
# ------------------------------------------------------------------------------

def predict_from_file(file_location, model, labels, max_time_steps):
    input_data = preprocess_audio(file_location, max_time_steps)
    if input_data is None:
        raise HTTPException(status_code=400, detail="Impossible de traiter le fichier audio.")
    predictions = model.predict(input_data)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = labels[predicted_index]
    confidence = round(float(predictions[predicted_index]) * 100, 2)
    all_confidences = {label: f"{round(float(pred) * 100, 2)}%" for label, pred in zip(labels, predictions)}
    return {
        "label": predicted_label,
        "confidence": f"{confidence}%",
        "all_confidences": all_confidences
    }

# ------------------------------------------------------------------------------
# Endpoints pour upload de fichier
# ------------------------------------------------------------------------------

@app.post("/predict/esc10")
async def predict_esc10(file: UploadFile = File(...)):
    file_location = "temp_audio.wav"
    try:
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        return predict_from_file(file_location, model_esc10, LABELS_ESC10, MAX_TIME_STEPS_ESC10)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

@app.post("/predict/urbansound")
async def predict_urbansound(file: UploadFile = File(...)):
    file_location = "temp_audio.wav"
    try:
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        return predict_from_file(file_location, model_urbansound, LABELS_US8K, MAX_TIME_STEPS_US8K)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

# ------------------------------------------------------------------------------
# Endpoints pour prédiction via YouTube
# ------------------------------------------------------------------------------

def download_youtube_audio(url):
    temp_id = str(uuid.uuid4())
    mp3_path = f"temp_audio_{temp_id}.mp3"
    wav_path = f"temp_audio_{temp_id}.wav"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': mp3_path.replace('.mp3', ''),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'keepvideo': False
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if not os.path.exists(mp3_path):
            print(f"Erreur : Le fichier MP3 {mp3_path} n'a pas été téléchargé.")
            return None
        (ffmpeg.input(mp3_path)
         .output(wav_path, format='wav')
         .run(overwrite_output=True, quiet=True))
        os.remove(mp3_path)
        return wav_path
    except Exception as e:
        print(f"Erreur lors du téléchargement/conversion YouTube: {e}")
        return None

def cleanup_temp_files():
    for file in glob.glob("temp_audio_*"):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Impossible de supprimer {file}: {e}")

@app.post("/predict_youtube/esc10")
async def predict_youtube_esc10(url: str):
    file_location = None
    try:
        cleanup_temp_files()
        file_location = download_youtube_audio(url)
        if file_location is None:
            raise HTTPException(status_code=400, detail="Échec du téléchargement de l'audio.")
        return predict_from_file(file_location, model_esc10, LABELS_ESC10, MAX_TIME_STEPS_ESC10)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_files()

@app.post("/predict_youtube/urbansound")
async def predict_youtube_urbansound(url: str):
    file_location = None
    try:
        cleanup_temp_files()
        file_location = download_youtube_audio(url)
        if file_location is None:
            raise HTTPException(status_code=400, detail="Échec du téléchargement de l'audio.")
        return predict_from_file(file_location, model_urbansound, LABELS_US8K, MAX_TIME_STEPS_US8K)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_files()

# ------------------------------------------------------------------------------
# Lancer l'API
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

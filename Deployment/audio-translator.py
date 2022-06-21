from sklearn.preprocessing import LabelEncoder
import os
import librosa
import numpy as np
from scipy.io import wavfile
import warnings
from deep_translator import GoogleTranslator
from google.cloud import storage, aiplatform
import pickle 
import sys

warnings.filterwarnings("ignore")

def preprocess_audio(audio_file):
    samples, sample_rate = librosa.load(audio_file, sr = 16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    wave = np.array(samples).reshape(-1,8000,1)
    return wave

def predict_and_translate(input_audio,tf_model,encode,target_lang):
    transform_audio = preprocess_audio(input_audio)
    predict_label = tf_model.predict(transform_audio).argmax(axis=-1)
    result = encode.inverse_transform(predict_label)
    print("Audio input:",result[0])
    try: 
        # Using google translator from deep translator to translate the audio text
        translated = GoogleTranslator(source='auto', target=target_lang.lower()).translate(result[0])
        print("Translation Language:",target_lang)
        print("Translation result:",translated)
    except:
        print("Unsupported Language")

def main():
    target_lang=sys.argv[1]
    input_audio_dir = "data/audio_for_translation"
    filenames = os.listdir("data/audio_for_translation")
    print("**Audio-Translator** Input",len(filenames),"audio files.")
    print("**Audio-Translator** Connect to Tensorflow model endpoint.")
    endpoint = aiplatform.Endpoint(
        endpoint_name="projects/689322264179/locations/us-central1/endpoints/7542289126729449472")
    print("**Audio-Translator** Load encoder.")
    encoder=pickle.load(open('encoder.pkl','rb'))
    for audio in filenames:
        processed_audio = preprocess_audio(input_audio_dir+'/'+audio)
        print("**Audio-Translator** Processing file:", input_audio_dir+'/'+audio)
        ep_response = endpoint.predict(instances=processed_audio.astype("float").tolist())
        text_label = np.argmax(ep_response.predictions[0])
        text=encoder.inverse_transform([text_label])
        print("**Audio-Translator** Input text:", text[0])
        print("**Audio-Translator** Translate to:", target_lang)
        try: 
            # Using google translator from deep translator to translate the audio text
            translated = GoogleTranslator(source='auto', target=target_lang.lower()).translate(text[0])
            print("**Audio-Translator** Translation result:",translated)
        except:
            print("Unsupported Language")
    print("**Audio-Translator** Translation complete.")
    return
    
if __name__ == "__main__":
    main()
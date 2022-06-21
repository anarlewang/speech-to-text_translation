#!/bin/sh
echo "**Audio-Translator** Installing required packages..."
sudo apt-get install libsndfile1-dev
pip3 install google-cloud-aiplatform --upgrade --user
pip install -r requirements.txt
echo "**Audio-Translator** Installation complete, ready to run."

echo "**Audio-Translator** Download encoder from GCP bucket..."
gsutil -m cp gs://audio-bucket20200620/encoder.pkl .
echo "**Audio-Translator** Download input audio from GCP bucket..."
gsutil -m cp -r gs://audio-bucket20200620/audio_for_translation ./data
python3 audio-translator.py "$@"

import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from typing import List
from soundfile import read as read_soundfile


class AudioSegment:
    def __init__(self, path):
        self.path = path
        self.audio, self.sample_rate = librosa.load(path)
    
    def __str__(self) -> str:
        return self.path


class Transcriber:
    def __init__(self, model_path: str, processor_path: str):
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)
    
    def __call__(self, audio_segment: AudioSegment) -> List[str]:
        return self.model(self.processor(audio_segment.audio))


Transcriber(model_path="facebook/wav2vec2-base-960h", processor_path="facebook/wav2vec2-base-960h")
# download sample audio using librosa
sample_audio = AudioSegment(r"C:\Users\melgi\portfolio\speech\WinstonChurchillSpeech.wav")
print(sample_audio.audio)
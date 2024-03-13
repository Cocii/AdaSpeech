import wave

def get_sample_rate(wav_file):
    with wave.open(wav_file, 'rb') as wav:
        sample_rate = wav.getframerate()
    return sample_rate

wav_file = '/data/speech_data/gigaspeech/test/en/POD0000000004_S0000128.wav'

sample_rate = get_sample_rate(wav_file)
print(f"采样率: {sample_rate} Hz")

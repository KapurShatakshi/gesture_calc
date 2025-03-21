import pyaudio
import numpy as np

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Voice effect (chipmunk effect by increasing pitch)
def change_pitch(data, rate):
    data_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    data_np = np.interp(np.arange(0, len(data_np), 0.5), np.arange(0, len(data_np)), data_np)
    return data_np.astype(np.int16).tobytes()

# Start audio stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

print("Voice Changer Running... (Press Ctrl+C to stop)")
try:
    while True:
        data = stream.read(CHUNK)
        modified_data = change_pitch(data, RATE)
        stream.write(modified_data)
except KeyboardInterrupt:
    print("Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()

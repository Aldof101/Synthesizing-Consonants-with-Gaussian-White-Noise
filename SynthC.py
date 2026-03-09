import os
import random
import wave
import struct
import math
from array import array

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SAMPLE_RATE = 44100
DURATION = 0.2
ATTACK = DURATION * 0.5        # 50% attack
DECAY = DURATION * 0.5          # 50% decay
AMPLITUDE = 10 ** (-16 / 20)    # -16 dB FS linear factor

REF_DIR = r"E:\桌面\Nonbio\無中之人音源子音作成 ver1.0\C_Japanese"
OUTPUT_DIR = r"E:\桌面\Nonbio\自动子音\test"
FILES = {
    's-': os.path.join(REF_DIR, 's-.wav'),
    'sh-': os.path.join(REF_DIR, 'sh-.wav')
}

# Goertzel settings: analyse a segment from the middle of the reference
ANALYSIS_DURATION = 0.05        # target length (will be shortened if file is too small)
CANDIDATE_FREQS = list(range(1000, 8500, 250))   # search from 1 kHz to 8 kHz
Q_FACTOR = 5.0                  # quality factor for the band‑pass filter

# ----------------------------------------------------------------------
# Helper: read a WAV file and return samples as floats (-1..1)
# ----------------------------------------------------------------------
def read_wave_floats(path):
    with wave.open(path, 'rb') as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    if nchannels != 1 or sampwidth != 2 or framerate != SAMPLE_RATE:
        raise ValueError(f"File {path} must be mono, 16-bit, {SAMPLE_RATE} Hz")

    # Convert bytes to signed 16‑bit ints, then to float
    ints = array('h', frames)          # 'h' = signed short (2 bytes)
    floats = [x / 32768.0 for x in ints]   # range -1..1
    return floats

# ----------------------------------------------------------------------
# Goertzel algorithm: compute energy at a single frequency
# ----------------------------------------------------------------------
def goertzel(samples, target_freq, fs):
    """
    Return the squared magnitude (energy) of the signal at target_freq.
    """
    k = int(0.5 + len(samples) * target_freq / fs)
    omega = 2.0 * math.pi * k / len(samples)
    coeff = 2.0 * math.cos(omega)

    q0, q1, q2 = 0.0, 0.0, 0.0
    for s in samples:
        q0 = coeff * q1 - q2 + s
        q2 = q1
        q1 = q0

    real = q1 - q2 * math.cos(omega)
    imag = q2 * math.sin(omega)
    return real*real + imag*imag

# ----------------------------------------------------------------------
# Find the frequency with the highest energy in a robust segment
# ----------------------------------------------------------------------
def find_peak_frequency(file_path):
    samples = read_wave_floats(file_path)
    total_samples = len(samples)

    # Determine a safe analysis segment from the middle of the file
    # Use at most half of the file, and at most the requested ANALYSIS_DURATION
    max_analysis_samples = int(ANALYSIS_DURATION * SAMPLE_RATE)
    analysis_samples = min(max_analysis_samples, total_samples // 2)
    if analysis_samples < 10:  # too few samples – fallback to whole file
        analysis_samples = total_samples

    start_sample = (total_samples - analysis_samples) // 2
    segment = samples[start_sample:start_sample + analysis_samples]

    best_freq = None
    best_energy = -1.0
    for freq in CANDIDATE_FREQS:
        energy = goertzel(segment, freq, SAMPLE_RATE)
        if energy > best_energy:
            best_energy = energy
            best_freq = freq
    return best_freq

# ----------------------------------------------------------------------
# Design a second‑order band‑pass filter (RBJ cookbook)
# ----------------------------------------------------------------------
def design_bandpass(f0, fs, Q):
    """
    Returns filter coefficients (b0, b1, b2, a0, a1, a2) for a
    constant‑peak‑gain band‑pass filter (peak gain = 0 dB at f0).
    """
    w0 = 2.0 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2.0 * Q)

    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * math.cos(w0)
    a2 = 1.0 - alpha

    # Normalise to a0 = 1
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    return b0, b1, b2, a1, a2

# ----------------------------------------------------------------------
# Apply a biquad filter to a signal (direct form II transposed)
# ----------------------------------------------------------------------
def biquad_filter(signal, b0, b1, b2, a1, a2):
    out = [0.0] * len(signal)
    z1 = z2 = 0.0
    for i, x in enumerate(signal):
        y = b0 * x + z1
        out[i] = y
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
    return out

# ----------------------------------------------------------------------
# Generate noise, filter, apply envelope and gain
# ----------------------------------------------------------------------
def synthesize_from_filter(center_freq, output_path):
    num_samples = int(SAMPLE_RATE * DURATION)
    envelope = [0.0] * num_samples

    # Pre‑compute triangular envelope
    for i in range(num_samples):
        t = i / SAMPLE_RATE
        if t < ATTACK:
            envelope[i] = t / ATTACK
        elif t > DURATION - DECAY:
            envelope[i] = (DURATION - t) / DECAY
        else:
            envelope[i] = 1.0

    # Generate white Gaussian noise
    noise = [random.gauss(0.0, 1.0) for _ in range(num_samples)]

    # Design filter
    b0, b1, b2, a1, a2 = design_bandpass(center_freq, SAMPLE_RATE, Q_FACTOR)

    # Apply filter
    filtered = biquad_filter(noise, b0, b1, b2, a1, a2)

    # Apply envelope and amplitude scaling
    for i in range(num_samples):
        filtered[i] *= envelope[i] * AMPLITUDE

    # Convert to 16‑bit int
    int_samples = [max(-32768, min(32767, int(s * 32767))) for s in filtered]

    # Write WAV
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.setcomptype('NONE', 'not compressed')
        packed = b''.join(struct.pack('<h', s) for s in int_samples)
        wf.writeframes(packed)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for sound_name, ref_path in FILES.items():
        print(f"Processing {sound_name} from {ref_path} ...")

        # Find the dominant frequency in the reference
        peak_freq = find_peak_frequency(ref_path)
        print(f"  Detected peak near {peak_freq} Hz")

        # Synthesise using that frequency
        out_path = os.path.join(OUTPUT_DIR, f"{sound_name}.wav")
        synthesize_from_filter(peak_freq, out_path)
        print(f"  Saved to {out_path}")

if __name__ == "__main__":
    main()
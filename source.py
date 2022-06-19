# -*- coding: utf-8 -*-
"""
Created on Sat May 14 17:51:00 2022.

@author: Anastasios Stefanos Anagnostou
         Spyridon Papadopoulos
"""
import matplotlib.pyplot as plt
import time
import numpy as np
import soundfile

# ============================================================================
# DATA FOR PROCESSING
begin = time.time()


music_signal, music_srate = soundfile.read("music.wav")
music_signal = np.transpose(music_signal)
if music_signal.ndim == 2:
    music_signal = (music_signal[1]+music_signal[0])/2
music_signal = music_signal / abs(max(music_signal))

music_length = len(music_signal)
N = 512
MUL_N = [N*x for x in list(range(len(music_signal)//N+1))]
NUM_WINDOWS = int(np.ceil(len(music_signal)/N))
PN = 90.302  # dB
M = 32
B = 16  # number of bits used for the encoding of each signal sample
R = 2**B  # number of intensity levels of the original signal
# central frequency of each of M=32 filters
Fk = [(2*k-1)*music_srate/4/M for k in range(1, M+1)]

# END OF DATA FOR PROCESSING
# ============================================================================

# ============================================================================
# HELPFUL FUNCTIONS
# START OF PART 1 FUNCTIONS


def ath(freq):
    """Calculate the Absolute threshold of hearing."""
    return (3.64*(freq/1000)**(-0.8) -
            6.5 * np.exp((-0.6)*(freq/1000-3.3)**2)
            + (0.001)*(freq/1000)**4)


def bark(freq):
    """Convert Hz to Bark."""
    return 13*np.arctan(0.00076*freq)+3.5*np.arctan((freq/7500)**2)


def itof(k, samples=N, s_rate=music_srate):
    """Convert discrete frequency to natural frequency."""
    return s_rate * (k+1) // samples


def halfit(insig):
    """Half the insig."""
    return insig[0:len(insig)//2]


def power_spec(insig, points):
    """Calculate the power spectrum of insig using "points" points."""
    return halfit(PN+10*np.log10(
        abs(np.fft.fft(insig*np.hanning(len(insig)), points))**2))


def mask_band(k):
    """
    Mask band.

    Return the distances away from "k"
    that should be checked in order to decide if there is
    a mask in "k" or not.
    """
    if 2 < k < 63:
        return [2]
    if 63 <= k < 127:
        return [2, 3]
    if 127 <= k < 250:
        return [2, 3, 4, 5, 6]


def ismask(power_spectrum, k):
    """Check if point is mask."""
    return ((not (k < 3 or k > 249)) and
            (power_spectrum[k] > power_spectrum[k-1] and
             power_spectrum[k] > power_spectrum[k+1] and
             (not (False in [power_spectrum[k] > power_spectrum[k+pos]+7 and
                             power_spectrum[k] > power_spectrum[k-pos]+7
                             for pos in mask_band(k)]))))


def find_mask_positions(power_spectrum):
    """Find positions of masks."""
    return [x for x in range(len(power_spectrum)) if ismask(power_spectrum, x)]


def mask_power(power_spectrum, k):
    """
    Mask power.

    Returns the power of the mask in "k".
    """
    if not ismask(power_spectrum, k):
        return 0
    return (10*np.log10(10**(0.1*power_spectrum[k-1]) +
                        10**(0.1*power_spectrum[k]) +
                        10**(0.1*power_spectrum[k+1])))


def imt(pos_i, pos_j, masks, flag):
    """
    Individual masking thresholds.

    Returns the amount of covering in point i from the tone or
    noise mask in point j.
    """
    # freq_j = itof(pos_j)
    if barks[pos_i]-barks[pos_j] > 8 or barks[pos_i]-barks[pos_j] < -3:
        return 0

    def sf(pos_i, pos_j, masks):
        """
        Help function.

        Minimum power level that neighbouring frequencies must have,
        so that both of them are perceptible by a human.
        """
        # freq_i = itof(pos_i)
        # freq_j = itof(pos_j)
        delta_bark = barks[pos_i]-barks[pos_j]
        if delta_bark >= 8 or delta_bark < -3:
            return 0
        if -3 <= delta_bark < -1:
            return 17*delta_bark-0.4*masks[pos_j]+11
        if -1 <= delta_bark < 0:
            return delta_bark*(0.4*masks[pos_j]+6)
        if 0 <= delta_bark < 1:
            return -17*delta_bark
        return delta_bark*(0.15*masks[pos_j]-17)-0.15*masks[pos_j]
    if flag == "TM":
        return masks[pos_j]-0.275*barks[pos_j]+sf(pos_i, pos_j, masks)-6.025
    return masks[pos_j]-0.175*barks[pos_j]+sf(pos_i, pos_j, masks)-2.025


def gbm(k, tone_thresholds, noise_thresholds):
    """Calculate the Global Masking Threshold."""
    return 10*np.log10(
        10**(0.1*aths[k]) +
        sum([10**(0.1*(tone_thresholds[k][q]))
             for q in range(len(tone_thresholds[k]))]) +
        sum([10**(0.1*(noise_thresholds[k][m]))
             for m in range(len(noise_thresholds[k]))]))

# END OF PART 1 FUNCTIONS

# PART 2 FUNCTIONS


def downsample(insig, m=M):
    """Keep every m-th point of insig."""
    return insig[::m]


def mdct(in_sig, k, m=M, flag="analysis"):
    """Perform the Modified Discrete Cosine Transform."""
    n = np.linspace(0, 2*m-1, 2*m)
    hk = (np.sin((n+0.5)*np.pi/(2*m)) *
          ((2/m)**(1/2)) *
          (np.cos((2*n+m+1)*(2*k+1)*np.pi/(4*m))))
    if flag == "synthesis":
        # gk = hk*(2*m-1-n)
        gk = hk[::-1]
        return np.convolve(in_sig, gk)
    elif flag == "analysis":
        return np.convolve(in_sig, hk)
    print("wrong flag: either 'analysis' or 'synthesis'")


def bitsk(thresholds, i, j):
    """Calculate required bits for quantization."""
    if thresholds[i][j]:
        return int(np.ceil(np.log2(R/min(thresholds[i][j]))-1))
    return 0


def quantize(insig, bits, flag="adaptive"):
    """Perform quantization.

    Return the quantization level of each sample.
    """
    def find_best_match(sample, values):
        """Return value from values that minimizes difference from sample."""
        result = min([abs(value - sample) for value in values])
        if result <= 0:
            return -(result-sample)
        return result+sample
    if flag == "adaptive":
        return [find_best_match(sample,
                                np.linspace(min(insig), max(insig), 2**bits))
                for sample in insig]
    if flag == "8bit":
        return [find_best_match(sample, np.linspace(-1, 1, 2**8))
                for sample in insig]
    print("Flag either 'adaptive' or '8bit'")


def oversample(insig, m=M):
    """
    Oversample.

    Keep every m-th sample of insig and stuff the blanks with zeroes.
    """
    result = [0 for _ in range(len(insig)*m)]
    for s in range(len(insig)*m):
        if s % m == 0:
            result[s] = insig[s//m]
    return result


def process(windows, spec_thresh, flag="adaptive"):
    """Process the given windows and perform adaptive quantization."""
    mdct_convolutions = [[mdct(window, k) for k in range(M)]
                         for window in windows]
    mdct_downsampled = [[downsample(conv, M) for conv in convols]
                        for convols in mdct_convolutions]
    if (flag == "adaptive"):
        valid_thresholds = [[[spec_thresh[s][f] for f in domains[k]]
                             for k in range(M)] for s in range(NUM_WINDOWS)]
        Bk = [[bitsk(valid_thresholds, s, k) for k in range(M)]
              for s in range(NUM_WINDOWS)]
        quantized = [[quantize(mdct_downsampled[s][k], Bk[s][k], flag)
                      for k in range(M)] for s in range(NUM_WINDOWS)]
        oversampled = [[oversample(quantized[s][k]) for k in range(M)]
                       for s in range(NUM_WINDOWS)]
        return [[mdct(oversampled[s][k], k, M, "synthesis")
                 for k in range(M)] for s in range(NUM_WINDOWS)]
    elif (flag == "8bit"):
        quantized = [[quantize(mdct_downsampled[s][k], 8, flag)
                      for k in range(M)] for s in range(NUM_WINDOWS)]
        oversampled = [[oversample(quantized[s][k]) for k in range(M)]
                       for s in range(NUM_WINDOWS)]
        return [[mdct(oversampled[s][k], k, M, "synthesis")
                 for k in range(M)] for s in range(NUM_WINDOWS)]
    print("Flag either 'adaptive' or '8bit'")


def overlap_add(a, b, ai=0, bi=0):
    """Add signals with specified overlap."""
    assert ai >= 0
    assert bi >= 0
    al = len(a)
    bl = len(b)
    cl = max(ai+al, bi+bl)
    c = np.zeros(cl)
    c[ai: ai+al] += a
    c[bi: bi+bl] += b
    return c


def reconstruct(windows, filename, srate=44100):
    """Reconstruct the final signal using given windows and write to wav."""
    added = [np.zeros(len(windows[0][0])) for window in windows]
    for index, window in enumerate(windows):
        for filtered in window:
            added[index] = list(np.array(filtered)+np.array(added[index]))
    added = np.array(added)
    length = len(added[0])
    dif = length-N
    result = added[0]
    for i in range(1, NUM_WINDOWS):
        result = overlap_add(result, added[i], bi=len(result)-dif)
    soundfile.write(filename, result/abs(max(result)), srate)
    return result


def squared_error(sig1, sig2):
    """Calculate squared error between two signals."""
    return np.square(np.array(sig1)-np.array(sig2))

# END OF PART 2 FUNCTIONS

# END OF HELPFUL FUNCTIONS
# ============================================================================

# ============================================================================
# These arrays are helpful for speeding up some computations


itofr = [itof(k) for k in range(N//2)]  # index to natural frequency

aths = [ath(freq) for freq in itofr]    # absolute thresholds of hearing

barks = [bark(freq) for freq in itofr]  # index to bark frequency

domains = [[f for f in range(N//2)
            if ((2*k-1)*music_srate/4/M - music_srate/4/M <=
                itofr[f]
                <= (2*k-1)*music_srate/4/M + music_srate/4/M)]
           for k in range(1, M+1)]  # frequency domains for the filter in 2.3

# end of helpful arrays
# ============================================================================

# ============================================================================
# START OF PART 1


windowed_music_signals = [music_signal[x:x+N] for x in MUL_N]
windowed_music_signals[NUM_WINDOWS-1] = np.append(
    windowed_music_signals[NUM_WINDOWS-1],
    np.zeros(N-len(windowed_music_signals[NUM_WINDOWS-1])))

fig = 0
plt.figure(fig)
plt.plot(np.linspace(0, N/music_srate, N), windowed_music_signals[780])
plt.title("Music signal window 780")
plt.xlabel("seconds")
fig += 1

# power spectrum for each window of the music signal
power_spectra_music = [power_spec(music_signal[x:x+N], N) for x in MUL_N]

plt.figure(fig)
plt.plot(itofr, power_spectra_music[780])
plt.title("Power spectrum of window 780")
plt.xlabel("Hz")
plt.ylabel("dB SPL")
fig += 1

# positions of masks in power spectrum of each window
mask_positions = [find_mask_positions(spectrum)
                  for spectrum in power_spectra_music]
# power of each mask in power spectrum of each window
power_mask_positions = [[mask_power(spectrum, position)
                         for position in range(len(spectrum))]
                        for spectrum in power_spectra_music]

P_NM = np.load("P_NM.npy")
P_TMc = np.load("P_TMc.npy")
P_NMc = np.load("P_NMc.npy")

transposeP_TMc = np.transpose(P_TMc)
J_TM = [[j for j, toneMask in enumerate(transposeP_TMc[s]) if toneMask > 0]
        for s in range(NUM_WINDOWS)]    # indexes of tone masks

spectrarum_masks = [[row[s] for row in P_TMc] for s in range(NUM_WINDOWS)]
T_TM = [[[imt(i, j, spectrarum_masks[s], "TM") for j in J_TM[s]]
        for i in range(N//2)]
        for s in range(NUM_WINDOWS)]    # individual masking thresholds

plt.figure(fig)
plt.plot(barks, transposeP_TMc[780], barks, aths)
plt.title("Power of Tone Masks for window 780 and ATH")
plt.xlabel("frequency (bark)")
plt.ylabel("dB SPL")
fig += 1

transposeP_NMc = np.transpose(P_NMc)
J_NM = [[j for j, noiseMask in enumerate(transposeP_NMc[s]) if noiseMask > 0]
        for s in range(NUM_WINDOWS)]    # indexes of noise masks

plt.figure(fig)
plt.plot(barks, transposeP_TMc[780], barks, transposeP_NMc[780], barks, aths)
plt.title("Power of Tone Masks for window 780 and ATH")
plt.xlabel("frequency (bark)")
plt.ylabel("dB SPL")
fig += 1

spectrarum_masks = [[row[s] for row in P_NMc] for s in range(NUM_WINDOWS)]
T_NM = [[[imt(i, j, spectrarum_masks[s], "NM") for j in J_NM[s]]
        for i in range(N//2)]
        for s in range(NUM_WINDOWS)]    # individual masking thresholds

plt.figure(fig)
plt.plot(barks, T_TM[780], barks, aths)
plt.title("Tone Masks for window 780 and absolute threshold of hearing")
plt.xlabel("frequency (bark)")
plt.ylabel("dB SPL")
fig += 1

plt.figure(fig)
plt.plot(barks, T_TM[780], barks, T_NM[780], barks, aths)
plt.title("Tone and Noise Masks for window 780 and absolute threshold of hearing")
plt.xlabel("frequency (bark)")
plt.ylabel("dB SPL")
fig += 1

# global masking thresholds
spectrarum_thresholds = [[gbm(i, T_TM[s], T_NM[s])
                         for i in range(N//2)]
                         for s in range(NUM_WINDOWS)]

plt.figure(fig)
plt.plot(barks, spectrarum_thresholds[780], barks, aths)
plt.title("Global Masking Threshold and ATH")
plt.xlabel("frequency (bark)")
plt.ylabel("dB SPL")
fig += 1

for i in range(0, 20):
    plt.figure(fig)
    plt.plot(barks, spectrarum_thresholds[900+i])
    plt.title("Global masking threshold from window " +
              str(900) + "to " + str(919))
    plt.xlabel("frequencies (bark)")
    plt.ylabel("dB SPL")
fig += 1
for i in range(0, 20):
    plt.figure(fig)
    plt.plot(barks, spectrarum_thresholds[1000+i])
    plt.title("Global masking threshold from window " +
              str(1000) + "to " + str(1019))
    plt.xlabel("frequencies (bark)")
    plt.ylabel("dB SPL")
fig += 1

for i in range(0, 20):
    plt.figure(fig)
    plt.plot(barks, spectrarum_thresholds[200+i])
    plt.title("Global masking threshold from window " +
              str(200) + "to " + str(219))
    plt.xlabel("frequencies (bark)")
    plt.ylabel("dB SPL")
fig += 1
for i in range(0, 20):
    plt.figure(fig)
    plt.plot(barks, spectrarum_thresholds[300+i])
    plt.title("Global masking threshold from window " +
              str(300) + "to " + str(319))
    plt.xlabel("frequencies (bark)")
    plt.ylabel("dB SPL")
fig += 1

# END OF PART 1
# ============================================================================

# ============================================================================
# START OF PART 2


synthesized = process(windowed_music_signals,
                      spectrarum_thresholds, "adaptive")

synthesized_8bit = process(windowed_music_signals,
                           spectrarum_thresholds, "8bit")

result_adaptive = reconstruct(synthesized, "result_adaptive.wav")

result_8bit = reconstruct(synthesized_8bit, "result_8bit.wav")

# pad some zeros to the original music signal in order to calculate the
# mean squared error. The reason why this is necessary is that the
# the reconstructed signal contains some tiny values that could be regarded
# as zeroes, but they are not.
music_signal_altered = np.pad(
    music_signal, (2*M, len(result_adaptive)-2*M-music_length), 'constant')

squared_error_adaptive = squared_error(music_signal_altered, result_adaptive)
error_adaptive = np.array(music_signal_altered)-np.array(result_adaptive)
mse_adaptive = sum(squared_error_adaptive)/len(squared_error_adaptive)
squared_error_8bit = squared_error(music_signal_altered, result_8bit)
error_8bit = np.array(music_signal_altered)-np.array(result_8bit)
mse_8bit = sum(squared_error_8bit)/len(squared_error_8bit)

plt.figure(fig)
plt.plot(np.linspace(0, music_length-1, music_length),
         music_signal, label="music_signal")
plt.plot(np.linspace(0, len(result_8bit)-1, len(result_8bit)),
         result_8bit, label="result_8bit")
plt.title("Original music signal and reconstructed signal using 8-bit quantizer")
plt.xlabel("samples")
plt.ylabel("dB SPL")
plt.legend(loc="upper right")
fig += 1

plt.figure(fig)
plt.plot(np.linspace(0, music_length-1, music_length),
         music_signal, label="music_signal")
plt.plot(np.linspace(0, len(result_adaptive)-1, len(result_adaptive)),
         result_adaptive, label="result_adaptive")
plt.title("Original music signal and reconstructed signal using adaptive quantizer")
plt.xlabel("samples")
plt.ylabel("dB SPL")
plt.legend(loc="upper right")
fig += 1

plt.figure(fig)
plt.plot(np.linspace(0, len(result_8bit)-1, len(result_8bit)),
         error_8bit, label="error_8bit")
plt.plot(np.linspace(0, len(result_adaptive)-1, len(result_adaptive)),
         error_adaptive, label="error_adaptive")
plt.title("8-bit uniform quantizer error and adaptive quantizer error")
plt.xlabel("samples")
plt.ylabel("dB SPL")
plt.legend(loc="upper right")
fig += 1

end = time.time()
print(end-begin)

# END OF PART 2
# =============================================================================

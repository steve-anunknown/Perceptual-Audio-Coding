# -*- coding: utf-8 -*-
"""
Created on Sat May 14 17:51:00 2022

@author: Anastasios Stefanos Anagnostou
         Spyridon Papadopoulos
"""
# Step 1.0
import time
import matplotlib.pyplot as plt
import numpy as np
import soundfile

start = time.time()

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

# Helpful functions


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
                        10**(0.1*power_spectrum[k+1]))**2)


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


def downsample(insig, m):
    """Keep every m-th point of insig."""
    return insig[::m]


def mdct(in_sig, k, m=M, flag="analysis"):
    """Perform the Modified Discrete Cosine Transform."""
    n = np.linspace(0, 2*m-1, 2*m)
    hk = (np.sin((n+0.5)*np.pi/2/m) *
          ((2/M)**(1/2)) *
          (np.cos((2*n+m+1)*(2*k+1)*np.pi/(4*m))))
    if flag == "synthesis":
        gk = hk*(2*m-1-n)
        return np.convolve(in_sig, gk)
    elif flag == "analysis":
        return np.convolve(in_sig, hk)
    print("wrong flag: either 'analysis' or 'synthesis'")


def bitsk(thresholds, i, j):
    """Calculate required bits for quantization."""
    if thresholds[i][j]:
        return int(np.log2(R/min(thresholds[i][j]))-1)
    return 0


def stepk(insig, bits):
    """Calculate the quantization step of insig using "bits" bits."""
    return (max(insig)-min(insig))/(2**(bits+1))


def quantize(insig, bits):
    """Perform quantization.

    Return the quantization level of each sample.
    """
    levels = 2**bits
    sigmax = max(insig)
    sigmin = min(insig)
    sigrange = sigmax-sigmin
    return [int(np.round((sample-sigmin)/sigrange*(levels-1)))
            for sample in insig]


def dequantize(insig, first_level, step, bits):
    """Decode a quantized signal."""
    # first_level and step are numpy 16 bit floats

    # step*max(insig) is basically the max value of
    # the input signal before quantization.
    # first_level is the minimum value of the input
    # signal before quantization.
    # Their difference gives the range of the input
    # signal before quantization.
    if bits == 0:
        return insig
    sigrange = step*max(insig)-first_level
    levels = 2**bits
    return [sample/(levels-1)*(sigrange)+first_level
            for sample in insig]


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

# end of helpful functions

# These arrays are helpful for speeding up some computations


itofr = [itof(k) for k in range(N//2)]

aths = [ath(freq) for freq in itofr]

barks = [bark(freq) for freq in itofr]

# end of helpful arrays


power_spectra_music = [power_spec(music_signal[x:x+N], N) for x in MUL_N]

mask_positions = [find_mask_positions(spectrum)
                  for spectrum in power_spectra_music]
power_mask_positions = [[] for element in power_spectra_music]
for index, spectrum in enumerate(power_spectra_music):
    for position, sample in enumerate(spectrum):
        power_mask_positions[index].append(mask_power(spectrum, position))

# Load new masks because the old ones were wrong


P_NM = np.load("P_NM.npy")
P_TMc = np.load("P_TMc.npy")
P_NMc = np.load("P_NMc.npy")

end = time.time()
print(end-start)

start = time.time()
transposeP_TMc = np.transpose(P_TMc)
J_TM = [[j for j, noiseMask in enumerate(transposeP_TMc[s]) if noiseMask > 0]
        for s in range(NUM_WINDOWS)]

start1 = time.time()
spectrarum_masks = [[row[s] for row in P_TMc] for s in range(NUM_WINDOWS)]
T_TM = [[[imt(i, j, spectrarum_masks[s], "TM") for j in J_TM[s]]
        for i in range(N//2)]
        for s in range(NUM_WINDOWS)]

end1 = time.time()
print(end1-start1)

transposeP_NMc = np.transpose(P_NMc)
J_NM = [[j for j, noiseMask in enumerate(transposeP_NMc[s]) if noiseMask > 0]
        for s in range(NUM_WINDOWS)]

start1 = time.time()
spectrarum_masks = [[row[s] for row in P_NMc] for s in range(NUM_WINDOWS)]
T_NM = [[[imt(i, j, spectrarum_masks[s], "NM") for j in J_NM[s]]
        for i in range(N//2)]
        for s in range(NUM_WINDOWS)]

end1 = time.time()
print(end1-start1)
end = time.time()
print(end-start)

start = time.time()

spectrarum_thresholds = [[gbm(i, T_TM[s], T_NM[s])
                         for i in range(N//2)]
                         for s in range(NUM_WINDOWS)]
end = time.time()
print(end-start)

start = time.time()

windowed_music_signals = [music_signal[x:x+N] for x in MUL_N]
mdct_convolutions = [[mdct(window, k)
                     for k in range(M)]
                     for window in windowed_music_signals]

mdct_downsampled = [[downsample(conv, M)
                    for conv in convols]
                    for convols in mdct_convolutions]

end = time.time()
print(end-start)

B = 16  # number of bits used for the encoding of each signal sample
R = 2**B  # number of intensity levels of the original signal
# central frequency of each of M=32 filters
Fk = [(2*k-1)*music_srate*np.pi/2/M for k in range(1, M+1)]

start = time.time()

domains = [[f for f in range(N//2)
            if ((2*k-1)*music_srate*np.pi/2/M - music_srate*np.pi/2/M <=
                itofr[f]
                <= (2*k-1)*music_srate*np.pi/2/M + music_srate*np.pi/2/M)]
           for k in range(1, M+1)]

valid_thresholds = [[[spectrarum_thresholds[s][f]
                      for f in domains[k]]
                     for k in range(M)]
                    for s in range(NUM_WINDOWS)]

end = time.time()
print(end-start)
Bk = [[bitsk(valid_thresholds, s, k) for k in range(M)]
      for s in range(NUM_WINDOWS)]

Dk = [[stepk(mdct_downsampled[s][k], Bk[s][k])
      for k in range(M)] for s in range(NUM_WINDOWS)]

quantized = [[quantize(mdct_downsampled[s][k], Bk[s][k])
              for k in range(M)] for s in range(NUM_WINDOWS)]


quantized_8bit = [[quantize(mdct_downsampled[s][k], 8)
                   for k in range(M)] for s in range(NUM_WINDOWS)]

first_levels = [[min(mdct_downsampled[s][k])
                 for k in range(M)]
                for s in range(NUM_WINDOWS)]

# =============================================================================
# time_axis = np.linspace(0,N/music_srate,N)
# fig=0
# plt.figure(fig)
# plt.plot(time_axis,windowed_music_signals[1])
# plt.plot(time_axis,quantize(windowed_music_signals[1],Dk[1][0]))
# fig+=1
# plt.figure(fig)
# plt.plot(time_axis,windowed_music_signals[1])
# plt.plot(time_axis,quantize(windowed_music_signals[1],
#         (max(windowed_music_signals[1])-min(windowed_music_signals[1]))/(2**5)))
# fig+=1
# =============================================================================

dequantized = [[dequantize(quantized[s][k],
                           first_levels[s][k].astype("float16"),
                           Dk[s][k].astype("float16"),
                           Bk[s][k]) for k in range(M)]
               for s in range(NUM_WINDOWS)]

oversampled = [[oversample(dequantized[s][k]) for k in range(M)]
               for s in range(NUM_WINDOWS)]

synthesized = [[mdct(oversampled[s][k], k, M, "synthesis")
                for k in range(M)] for s in range(NUM_WINDOWS)]

# =============================================================================
# fig = 0
# for i in range(0, 20):
#     plt.figure(fig)
#     plt.plot(barks, spectrarum_thresholds[i])
# fig += 1
# for i in range(0, 20):
#     plt.figure(fig)
#     plt.plot(barks, spectrarum_thresholds[100+i])
# fig += 1
# for i in range(0, 20):
#     plt.figure(fig)
#     plt.plot(barks, spectrarum_thresholds[200+i])
# fig += 1
# for i in range(0, 20):
#     plt.figure(fig)
#     plt.plot(barks, spectrarum_thresholds[300+i])
# fig += 1
# for i in range(0, 20):
#     plt.figure(fig)
#     plt.plot(barks, spectrarum_thresholds[400+i])
# fig += 1
# for i in range(0, 20):
#     plt.figure(fig)
#     plt.plot(barks, spectrarum_thresholds[500+i])
# fig += 1
# for i in range(0, 20):
#     plt.figure(fig)
#     plt.plot(barks, spectrarum_thresholds[600+i])
# fig += 1
# for i in range(0, 20):
#     plt.figure(fig)
#     plt.plot(barks, spectrarum_thresholds[700+i])
# fig += 1
# for i in range(0, 20):
#     plt.figure(fig)
#     plt.plot(barks, spectrarum_thresholds[800+i])
# fig += 1
# for i in range(0, 20):
#     plt.figure(fig)
#     plt.plot(barks, spectrarum_thresholds[900+i])
# fig += 1
# for i in range(0, 20):
#     plt.figure(fig)
#     plt.plot(barks, spectrarum_thresholds[1000+i])
# fig += 1
# =============================================================================

import numpy as np


class BPM:
    """Namespace for `sig.bpm.<method>()` -- every method here is an
    independent way to estimate BPM from the same underlying Signal, so they
    can be called side by side and cross-checked against each other."""

    def __init__(self, sig):
        self._sig = sig

    def fft(self, ):
        """BPM from the raw FFT amplitude spectrum (amp**2 as power)."""
        sig = self._sig
        f, _, amp = sig.fft()
        _, bpm = sig.peak_bpm(f, amp ** 2, )
        return bpm

    def welch(self,):
        """BPM from the Welch power spectral density (low-variance)."""
        sig = self._sig
        f, pxx = sig.welch()
        _, bpm = sig.peak_bpm(f, pxx, )
        return bpm


    def wavelet(self, ):
        """BPM from the median wavelet-ridge frequency over time."""
        sig = self._sig
        _, ridge_freq = sig.wavelet_ridge()
        return float(np.median(ridge_freq) * 60)

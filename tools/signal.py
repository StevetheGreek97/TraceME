import warnings

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal as sps

from .bpm import BPM


class Signal:

    def __init__(self, signal, fps=60, band=(2.0, 10.0)):
        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 1:
            raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
        if signal.size == 0:
            raise ValueError("signal must not be empty")
        if not np.all(np.isfinite(signal)):
            raise ValueError("signal contains NaN or infinite values")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be finite and positive, got {fps}")
        if len(band) != 2 or not (0 < band[0] < band[1] < fps / 2):
            raise ValueError(
                f"band must be a (low, high) tuple with 0 < low < high < fps/2 ({fps / 2}), got {band}"
            )
        if np.ptp(signal) == 0:
            raise ValueError(f"signal is constant (zero variance), value={signal[0]}")

        self.signal = signal
        self.fps = fps
        self.band = band
        self.bpm = BPM(self)

        if self._signal_lost(signal):
            self.trim(end=self.dropout(), inplace=True)

        if self.duration < 1.0:
            raise ValueError(f"Less than 1 second of valid data. Duration: {self.duration:.2f} seconds.")


    def _signal_lost(self, signal):
        return -1 in signal

    def __repr__(self):
        return f"Signal(n={len(self)}, fps={self.fps}, band={self.band})"

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, key):
        """sig[i] -> the raw sample. sig[a:b] -> a new Signal over that
        slice, same fps/band -- e.g. sig[:sig.lost_frame()] to trim to
        the prefix before the signal goes flat."""
        result = self.signal[key]
        if isinstance(key, slice):
            return Signal(result, fps=self.fps, band=self.band)
        return result

    @property
    def duration(self):
        """Length of the signal in seconds."""
        return len(self) / self.fps

    def dropout(self):
        return np.flatnonzero(self.signal == -1)[0] if (self.signal == -1).any() else None

    def trim(self, start=0, end=None, inplace=False):
        end = len(self) if end is None else end
        if inplace:
            if start > 0 or end < len(self):
                warnings.warn(
                    f"Signal trimmed from {len(self)} to {end - start} frames "
                    f"(start={start}, end={end})."
                )
            self.signal = self.signal[start:end]
            return self
        return self[start:end]

    def detrend(self, inplace=False):
        """Remove linear trend from the signal."""
        x = self.signal
        if inplace:
            self.signal = sps.detrend(x)
            return self.signal
        return sps.detrend(x)

    def bandpass(self, band=None, inplace=False):
        """Bandpass filter the signal.
        returns the filtered signal.
        """
        band = self.band if band is None else band
        x = self.signal
        sos = sps.butter(4, band, btype='band', fs=self.fps, output='sos')
        if inplace:
            self.signal = sps.sosfilt(sos, x)
            return self.signal
        return sps.sosfilt(sos, x)

    def fft(self):
        """Compute the FFT of the signal and return frequency bins and amplitude spectrum."""
        x = self.signal
        n = len(x)
        f = np.fft.rfftfreq(n, d=1/self.fps)
        X = np.fft.rfft(x)
        amp = np.abs(X) / n
        return f, X, amp

    def welch(self, nperseg=None):
        """Compute the Welch power spectral density estimate of the signal."""
        x = self.signal
        f, pxx = sps.welch(x, fs=self.fps, nperseg=nperseg or min(256, len(x)))
        return f, pxx

    def periodogram(self, x=None, window="hann"):
        """Single-segment power spectral density (the whole signal as one
        window, no Welch-style segment-averaging). Higher variance than
        welch() but full frequency resolution -- useful for short clips
        where welch()'s segmentation would blur the peak."""
        x = self.signal if x is None else x
        f, pxx = sps.periodogram(x, fs=self.fps, window=window)
        return f, pxx

    def cwt(self,  wavelet=None, bandwidth=1.5, center_freq=1.0, num_freqs=128):
        """Compute the continuous wavelet transform (scalogram) of the signal.

        bandwidth: time/frequency resolution trade-off (smaller = sharper in
            time, larger = sharper in frequency). Ignored if `wavelet` is given.
        Returns frequency (Hz), time (s), and power |coef|^2 with shape
        (num_freqs, len(x)) so power[i] is the power-over-time trace at
        freqs[i].
        """
     
        wavelet_name = wavelet or f"cmor{bandwidth}-{center_freq}"
        lo, hi = self.band
        freqs = np.geomspace(lo * 0.5, hi * 1.5, num_freqs)
        dt = 1 / self.fps
        scales = pywt.frequency2scale(wavelet_name, freqs * dt)
        coefs, freqs_out = pywt.cwt(self.signal, scales, wavelet_name, sampling_period=dt)
        power = np.abs(coefs) ** 2
        times = np.arange(len(self.signal)) / self.fps
        return freqs_out, times, power

    def peak_bpm(self, f, pxx,):
        """Find the peak frequency in the given band and convert to BPM."""
        band = self.band 
        mask = (f >= band[0]) & (f <= band[1])
        f_band, p_band = f[mask], pxx[mask]
        if len(p_band) == 0:
            raise ValueError("No frequencies in the specified band.")
        idx_peak = np.argmax(p_band)
        f_peak = f_band[idx_peak]
        bpm = f_peak * 60
        return f_peak, bpm

    def energy_concentration(self, win_s=1.0):
        """Fraction of the signal's total energy contained in its single
        most energetic win_s-second window.

        Near win_s / duration when energy is spread evenly across the
        clip (a sustained rhythm or sustained noise). Near 1.0 when
        nearly all energy sits inside one brief burst and the rest of
        the clip is near-silent -- e.g. a transient from a lost tracking
        mask. Run this as a gate before trusting any spectral metric
        (snr, peak_prominence, spectral_entropy_score): a burst spreads
        power across all frequencies, so those metrics can't tell "20s
        of noise" apart from "one glitch and 19s of silence" -- this
        can.
        """
        e = self.signal ** 2
        total = e.sum()
        if total == 0:
            return 0.0
        w = max(1, min(int(round(win_s * self.fps)), len(e)))
        win_energy = np.convolve(e, np.ones(w), mode="valid")
        return float(win_energy.max() / total)

    def snr(self, f_peak=None, tol_hz=0.2, f_max=None):
        
        band = self.band

        f, _, amp = self.fft()
        pxx = amp ** 2
  

        if f_peak is None:
            f_peak, _ = self.peak_bpm(f, pxx)
        f_max = band[1] * 3 if f_max is None else f_max

        df = f[1] - f[0]
        tol = max(tol_hz, 2 * df)
        mask_signal = (f >= f_peak - tol) & (f <= f_peak + tol)

        mask_window = (f >= band[0]) & (f <= f_max)
        mask_signal &= mask_window
        mask_noise = mask_window & ~mask_signal
        power_signal = pxx[mask_signal].sum()
        power_noise = pxx[mask_noise].sum()
        if power_noise == 0:
            return np.inf if power_signal > 0 else np.nan
        if power_signal == 0:
            return -np.inf
        snr_db = 10 * np.log10(power_signal / power_noise)
        return snr_db

    def peak_prominence(self, f_peak=None, f_max=None):
        """Topographic prominence (scipy.signal.peak_prominences) of the
        detected BPM peak within [band[0], f_max]: how far you'd have to
        descend from the peak before reaching either a taller point or the
        edge of the window. Complementary to snr() -- SNR compares total
        power near the peak against power everywhere else in the band;
        prominence instead asks how much the peak specifically rises above
        its immediate local baseline, so it isn't thrown off by a small
        amount of distant noise power the way a power-ratio metric can be.

        Returns the prominence value in the same power units as the chosen
        spectral estimate (not dB) -- compare it to the peak's own height
        (pxx at f_peak) for a sense of scale, or use peak_prominence_ratio()
        for that normalized comparison directly.
        """
        band = self.band

        f, _, amp = self.fft()
        pxx = amp ** 2

        if f_peak is None:
            f_peak, _ = self.peak_bpm(f, pxx)
        f_max = band[1] * 3 if f_max is None else f_max

        mask = (f >= band[0]) & (f <= f_max)
        f_win, p_win = f[mask], pxx[mask]
        idx_peak = int(np.argmin(np.abs(f_win - f_peak)))
        prominences, _, _ = sps.peak_prominences(p_win, [idx_peak])
        return float(prominences[0])

    def peak_prominence_ratio(self, f_peak=None, f_max=None):
        """peak_prominence() normalized by the peak's own height: how much
        of the peak's height survives down to its local baseline, as a
        fraction from 0 to 1. Close to 1 means the peak descends almost
        all the way to its local floor on both sides -- a real, locally
        clean peak even if it's sitting on top of an elevated broadband
        background that drags snr() down. Close to 0 means the peak barely
        rises above its immediate neighbors."""
        band = self.band

        f, _, amp = self.fft()
        pxx = amp ** 2

        if f_peak is None:
            f_peak, _ = self.peak_bpm(f, pxx)

        height = pxx[np.argmin(np.abs(f - f_peak))]
        if height == 0:
            return 0.0
        return self.peak_prominence(f_peak=f_peak, f_max=f_max) / height

    def autocorr_peak_ratio(self):
        """Ratio of the largest autocorrelation peak within the plausible
        beat-period range (derived from `band`) to the zero-lag
        autocorrelation (total energy).

        Purely time-domain -- doesn't depend on FFT resolution or the
        band's frequency-domain shape, so it's an independent cross-check
        on periodicity rather than another spectral-domain restatement of
        snr()/spectral_entropy_score().

        Near 1: the signal looks almost the same one beat-period later
        (strongly periodic). Near 0: no meaningful self-similarity at any
        plausible beat lag (noise-like)."""
        x = self.signal - np.mean(self.signal)
        ac = sps.correlate(x, x, mode="full")
        ac = ac[len(ac) // 2:]  # lags 0..n-1

        zero_lag = ac[0]
        if zero_lag == 0:
            return 0.0

        lo_lag = int(np.floor(self.fps / self.band[1]))
        hi_lag = min(int(np.ceil(self.fps / self.band[0])), len(ac) - 1)
        if lo_lag >= hi_lag:
            return 0.0

        peak = ac[lo_lag:hi_lag + 1].max()
        return float(peak / zero_lag)

    def mean_peak_ratio(self, f_peak=None):
        """Mean FFT amplitude across `band`, divided by the peak's own
        amplitude -- the inverse of the peak-to-average ratio (crest
        factor). A simpler cousin of snr(): same "how much does the peak
        stand out from the background" question, but with a plain mean
        instead of snr()'s signal-window-vs-noise-window power split, so
        expect it to correlate with snr() rather than add an independent
        axis.

        Close to 0: peak towers over the average level (clean). Close to
        1: peak is barely above the average level (noisy/flat)."""
        band = self.band
        f, _, amp = self.fft()

        if f_peak is None:
            f_peak, _ = self.peak_bpm(f, amp ** 2)

        peak_amp = amp[np.argmin(np.abs(f - f_peak))]
        if peak_amp == 0:
            return np.nan

        mask = (f >= band[0]) & (f <= band[1])
        return float(amp[mask].mean() / peak_amp)

    def wavelet_ridge(self):
        """Per-time-step dominant frequency within `band` from the wavelet
        scalogram (the 'ridge'). Returns (times_s, ridge_freq_hz)."""
        band = self.band 
        freqs, times, power = self.cwt()
        mask = (freqs >= band[0]) & (freqs <= band[1])
        f_band = freqs[mask]
        p_band = power[mask, :]
        ridge_idx = np.argmax(p_band, axis=0)
        ridge_freq = f_band[ridge_idx]
        return times, ridge_freq

    def _entropy_trace(self, band, bandwidth, num_freqs):
        """Raw Shannon entropy (bits) of the scalogram's power distribution
        at each time step, plus the maximum possible entropy for this grid.

        Evaluated over the wavelet grid's full span (band widened by the
        same margin cwt() uses) rather than clipped to `band`, so a peak
        sitting near a band edge keeps all of its wavelet's power. Clipping
        to `band` truncates that spread -- a 9.5 Hz tone in a 2-10 Hz band
        loses ~38% of its power out the top -- which concentrates whatever
        survives and makes an edge-adjacent peak score as artificially
        clean."""
        freqs, times, power = self.cwt(bandwidth=bandwidth, num_freqs=num_freqs)
        mask = (freqs >= band[0] * 0.5) & (freqs <= band[1] * 1.5)
        p_band = power[mask, :]
        n_bins = p_band.shape[0]

        totals = p_band.sum(axis=0)
        probs = np.divide(p_band, totals, out=np.zeros_like(p_band), where=totals > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(probs > 0, probs * np.log2(probs), 0.0)
        h = -terms.sum(axis=0)
        h[totals == 0] = np.log2(n_bins)  # no power anywhere -> maximally messy
        return times, h, np.log2(n_bins)

    def _tone_entropy(self, band, bandwidth, num_freqs):
        """Entropy that a single pure tone produces on this exact wavelet
        grid -- the concentration floor, used to normalize
        spectral_entropy().

        A wavelet has finite bandwidth, so it can never place a tone in one
        bin: at the defaults a pure tone still occupies ~25 of 75 bins.
        Normalizing by log2(n_bins) alone therefore never approaches 0, and
        worse, the floor drifts with `num_freqs` (0.62 at 32 bins, 0.81 at
        512) -- an internal sampling knob leaking into the score. Measuring
        the floor on the same grid cancels both."""
        key = (tuple(band), bandwidth, num_freqs, len(self), self.fps)
        cache = getattr(self, "_tone_entropy_cache", None)
        if cache is None:
            cache = self._tone_entropy_cache = {}
        if key not in cache:
            f0 = float(np.sqrt(band[0] * band[1]))  # geometric centre of the band
            t = np.arange(len(self)) / self.fps
            probe = Signal(np.sin(2 * np.pi * f0 * t), fps=self.fps, band=band)
            _, h, _ = probe._entropy_trace(band, bandwidth, num_freqs)
            cache[key] = float(np.median(h))
        return cache[key]

    def spectral_entropy(self, band=None, bandwidth=1.5, num_freqs=128):
        """Per-time-step spectral entropy within `band`, normalized to
        [0, 1]: 0 means power is as concentrated as this wavelet can
        resolve at that instant (one dominant tone -- a 'clean' scalogram),
        1 means power is spread perfectly evenly across every frequency
        (indistinguishable from noise -- a 'messy' scalogram).

        The scale is anchored at both ends -- 0 to the entropy of a pure
        tone on this grid, 1 to the uniform distribution -- so the score
        depends on the signal rather than on `num_freqs`.

        Returns (times_s, entropy_per_time).
        """
        band = self.band if band is None else band
        times, h, h_max = self._entropy_trace(band, bandwidth, num_freqs)
        h_tone = self._tone_entropy(band, bandwidth, num_freqs)
        entropy = (h - h_tone) / (h_max - h_tone)
        return times, np.clip(entropy, 0.0, 1.0)

    def spectral_entropy_score(self, band=None, bandwidth=1.5, num_freqs=128):
        """Single spectral-entropy summary for the whole clip -- the median
        of spectral_entropy()'s trace. Near 0 = one clear dominant
        frequency throughout (clean); near 1 = power stays smeared across
        many frequencies the whole time (messy/noise-like)."""
        band = self.band if band is None else band

        _, entropy = self.spectral_entropy(band=band, bandwidth=bandwidth,
                                           num_freqs=num_freqs)
        return float(np.median(entropy))

    @staticmethod
    def longest_good_run(good):
        """(start, end) of the longest contiguous True run in a boolean mask."""
        good = np.append(good, False)
        best = (0, 0)
        start = None
        for i, g in enumerate(good):
            if g and start is None:
                start = i
            elif not g and start is not None:
                if i - start > best[1] - best[0]:
                    best = (start, i)
                start = None
        return best

    def good_ridge_segment(self, k=6, inplace=False):
        """Start/end frame indices of the longest stretch where the wavelet
        ridge stays within k robust-sigma of its stable value -- catches a
        bad stretch anywhere in the clip, not just a trailing breakdown.

        inplace=True crops self.signal to that stretch (via trim()) and
        returns self instead of the (start, end) indices."""
        _, ridge = self.wavelet_ridge()
        med = np.median(ridge)
        mad = np.median(np.abs(ridge - med)) * 1.4826
        good = np.abs(ridge - med) <= k * mad if mad > 0 else np.ones(len(ridge), dtype=bool)
        start, end = self.longest_good_run(good)
        if inplace:
            return self.trim(start, end, inplace=True)
        return start, end


    def plot_power(self, db=True, show_snr_windows=True,
                   tol_hz=0.2, f_max=None, ax=None):
        """Line plot of the power spectrum with the detected peak marked.

        method: "welch" (default, smoother), "periodogram", or "fft" --
            same estimators the metrics use, so pass method="fft" to see
            exactly the spectrum snr() scores.
        db: plot power in dB (10*log10). Power spectra span orders of
            magnitude; on a linear axis everything but the peak flattens
            into the floor.
        show_snr_windows: shade the signal window (peak +/- tol) and the
            noise window ([band[0], f_max] minus the signal window) that
            snr() compares, so a low score can be read off the plot.
        """
        band = self.band
 

        f, _, amp = self.fft()
        pxx = amp ** 2
        
        f_peak, bpm = self.peak_bpm(f, pxx)
        f_max = band[1] * 3 if f_max is None else f_max

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        y = 10 * np.log10(pxx + 1e-20) if db else pxx
        keep = f > 0  # drop DC so the log axis isn't pinned to leakage at 0 Hz
        ax.plot(f[keep], y[keep], lw=1.5, color="tab:blue")

        if show_snr_windows:
            df = f[1] - f[0]
            tol = max(tol_hz, 2 * df)
            ax.axvspan(max(f_peak - tol, band[0]), min(f_peak + tol, f_max),
                       color="tab:orange", alpha=0.25,
                       label="signal window (snr)")
        ax.axvspan(*band, color="green", alpha=0.08, label="search band")
        ax.axvline(f_peak, color="r", ls="--", lw=1,
                   label=f"peak {f_peak:.2f} Hz ({bpm:.0f} BPM)")

        ax.set_xlim(0, band[1] +5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)" if db else "Power")
        ax.set_title(f"Power spectrum")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.legend()
        return ax

    def plot_fft(self,ax=None):
        """Stem plot of the raw FFT amplitude spectrum, peak marked."""
        band = self.band
        f_hz, _, amp = self.fft()
        f_peak, bpm = self.peak_bpm(f_hz, amp ** 2)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        ax.stem(f_hz[1:], amp[1:], basefmt="None")
        lo, hi = band
        ax.set_xlim(0, hi * 1.5)
        ax.axvline(f_peak, color="r", ls="--", label=f"peak {f_peak:.2f} Hz ({bpm:.0f} BPM)")
        ax.axvspan(lo, hi, color="green", alpha=0.08, label="search band")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        return ax

    def plot_wavelet(self, ax=None,  bandwidth=1.5,
                      num_freqs=128, dynamic_range_db=30):
        """
        Wavelet scalogram (time-frequency, better multi-resolution than spectrogram).

        bandwidth: window size control (see cwt docstring). Smaller = sharper
            in time (better for spotting brief bursts), larger = sharper in
            frequency (better for reading off a stable rate). Try 0.5-1 to
            zoom in on burst timing, 3-5 for a cleaner-looking ridge.
        dynamic_range_db: how many dB below the clip's peak power to still show
            color for. The full dB range (peak to noise floor) is usually
            100+ dB, which washes out real ridges into a gray-green blur.
            Restricting to the top `dynamic_range_db` dB makes a genuine
            dominant ridge stand out; lower this (e.g. 15-20) for more
            contrast, raise it (e.g. 50+) to see more of the quiet background.
        """
        band = self.band 
        freqs, times, power = self.cwt( bandwidth=bandwidth,
                                        num_freqs=num_freqs)
        power_db = 10 * np.log10(power + 1e-20)

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))

        vmax = power_db.max()
        vmin = vmax - dynamic_range_db
        pcm = ax.pcolormesh(times, freqs, power_db, shading="gouraud",
                             vmin=vmin, vmax=vmax)
        ax.axhspan(*band, color="white", alpha=0.15, label="search band")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Wavelet Scalogram")
        ax.set_ylim(0, freqs.max())
        plt.colorbar(pcm, ax=ax, label="Power (dB, relative to peak)")
        return ax

    def plot(self):
        plt.figure(figsize=(12, 4))
        plt.plot(self.signal)
        plt.title("Signal")
        plt.xlabel("Frame")
        plt.ylabel("Amplitude")
        plt.show()

import os
import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy.fft import fft, fftfreq, rfft, rfftfreq

import plotly.subplots as plysub
import plotly.graph_objs as go
import plotly.offline as plyoff

import sounddevice as sd # used to play sd.play(data, sample_rateOSError: PortAudio library not found)



def compute_spectra_by_second(data:np.ndarray, sample_rate:int,
                              apply_window:bool=True, n_bins:int=100,
                              fmin:float=20.0, fmax:float=None,
                              binning:str='log') -> pd.DataFrame:
    """
    Compute per-second magnitude spectra using rFFT and optionally
    bin frequencies.

    Returns a pandas.DataFrame where each row is a second (0,1,2,...) and
    columns are either raw frequency bins (Hz) or aggregated frequency bins
    depending on `n_bins`.

    Args:
        data: np.array, mono or (n_samples, n_channels)
        sample_rate: int
        apply_window: bool, whether to apply a Hann window to each 1s segment
        n_bins: int or None, number of frequency bins to aggregate into. If None,
            returns full rFFT frequency resolution.
        fmin: float, minimum frequency (Hz) for binning (ignored if n_bins is None)
        fmax: float or None, maximum frequency (Hz) for binning; defaults to
            min(20000, nyquist)
        binning: 'log' or 'linear' spacing for bins

    Returns:
        pandas.DataFrame: rows=seconds, cols=frequencies (Hz) or bin centers
    """

    arr = np.asarray(data)
    # mix to mono if needed
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype(float)
    else:
        arr = arr.astype(float)

    sec_len = int(sample_rate)
    n_full_seconds = len(arr) // sec_len
    if n_full_seconds == 0:
        # pad to one second
        padded = np.pad(arr, (0, max(0, sec_len - len(arr))))
        n_full_seconds = 1
    else:
        padded = arr[:n_full_seconds * sec_len]

    # frequency bins for a 1-second rFFT
    freqs = rfftfreq(sec_len, d=1.0 / sample_rate)
    nyquist = sample_rate / 2.0

    rows = []
    for s in range(n_full_seconds):
        seg = padded[s * sec_len:(s + 1) * sec_len].astype(float)
        if apply_window:
            seg = seg * np.hanning(len(seg))
        yf = rfft(seg)
        mag = (2.0 / len(seg)) * np.abs(yf)
        rows.append(mag)

    spec = np.vstack(rows)  # shape (n_seconds, n_freq_bins)

    # If no binning requested, return full-resolution DataFrame
    if n_bins is None:
        df = pd.DataFrame(spec, columns=np.round(freqs, 3))
        df.index.name = 'second'
        return df

    # Determine fmax default
    if fmax is None:
        fmax = min(20000.0, nyquist)

    # sanitize fmin/fmax
    fmin = max(0.0, float(fmin))
    fmax = min(float(fmax), nyquist)
    if fmax <= fmin:
        raise ValueError('fmax must be greater than fmin')

    # create bin edges
    if binning == 'log':
        # avoid log(0)
        low = max(fmin, freqs[1] if freqs[0] == 0.0 else freqs[0])
        edges = np.logspace(np.log10(low), np.log10(fmax), n_bins + 1)
        # optionally include DC (0 Hz) as its own bin if fmin==0
        if fmin == 0.0 and freqs[0] == 0.0:
            edges[0] = 0.0
    else:
        edges = np.linspace(fmin, fmax, n_bins + 1)

    # digitize the positive-frequency bins into our edges
    # freqs are >= 0, ensure we only consider freqs within [fmin, fmax]
    freq_vals = freqs
    # create an index mapping each freq to a bin (1..n_bins), frequencies outside
    # the edges will get 0 or n_bins+1 from digitize; we'll ignore those
    inds = np.digitize(freq_vals, edges)  # 0..n_bins+1

    # aggregate magnitudes into bins by summing magnitudes inside each bin
    binned = np.zeros((spec.shape[0], n_bins), dtype=float)
    for b in range(1, n_bins + 1):
        mask = inds == b
        if np.any(mask):
            binned[:, b - 1] = spec[:, mask].sum(axis=1)
        else:
            binned[:, b - 1] = 0.0

    # column labels: geometric center of each bin for log spacing, arithmetic for linear
    if binning == 'log':
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        centers = 0.5 * (edges[:-1] + edges[1:])

    # round centers nicely
    centers_rounded = np.round(centers, 3)
    df = pd.DataFrame(binned, columns=centers_rounded)
    df.index.name = 'second'

    return df


def piano_note_frequencies(midi_low:int=21, midi_high:int=108) -> list[tuple[str, float, int]]:
    """
    Return a list of (note_name, frequency_hz, midi_number) for piano keys.
    Defaults to MIDI 21 (A0) through 108 (C8) — the standard 88-key piano.

    Args:
        midi_low: int, lowest MIDI note number (inclusive)
        midi_high: int, highest MIDI note number (inclusive)

    Returns:
        list of (note_name, frequency_hz, midi_number) tuples
    """

    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    notes = []
    for m in range(midi_low, midi_high + 1):
        name = names[m % 12]
        octave = (m // 12) - 1
        note_name = f"{name}{octave}"
        freq = 440.0 * (2.0 ** ((m - 69) / 12.0))
        notes.append((note_name, float(freq), int(m)))

    return notes


def compute_note_magnitudes_per_second(data:np.ndarray, sample_rate:int,
                                       notes:list[tuple[str, float, int]]=None,
                                       apply_window:bool=True, n_harmonics:int=5,
                                       harmonic_weights:list[float]=None,
                                       use_db:bool=False, db_eps:float=1e-12,
                                       db_floor:float=None) -> pd.DataFrame:
    """
    Compute per-second magnitudes for a list of target note frequencies.

    This function maps each piano note to the nearest FFT bin for a 1-second rFFT
    and (optionally) aggregates energy from the first `n_harmonics` harmonics
    for better note detection on instruments like piano.

    Returns a DataFrame with seconds as the index and one column per note name
    containing the linear magnitude (same scaling as compute_spectra_by_second uses).

    Args:
        data: np.array audio
        sample_rate: int
        notes: iterable of (note_name, freq_hz, midi) tuples. If None, uses A0..C8.
        apply_window: bool, apply Hann window to each 1s segment
        n_harmonics: int, number of harmonics to include (1 = fundamental only)
        harmonic_weights: None or iterable of length n_harmonics giving weights for
            each harmonic (1-based). If None, defaults to 1/h (inverse harmonic)
        use_db: bool, if True, convert linear magnitudes to decibels (20*log10)
        db_eps: float, small value added before log10 to avoid log(0)
        db_floor: float or None, if set, clip dB values to this minimum

    Returns:
        pandas.DataFrame: rows=seconds, cols=note names
    """

    if notes is None:
        notes = [(n, f, m) for (n, f, m) in piano_note_frequencies()]

    # mix to mono
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype(float)
    else:
        arr = arr.astype(float)

    sec_len = int(sample_rate)
    n_full_seconds = len(arr) // sec_len
    if n_full_seconds == 0:
        padded = np.pad(arr, (0, max(0, sec_len - len(arr))))
        n_full_seconds = 1
    else:
        padded = arr[:n_full_seconds * sec_len]

    freqs = rfftfreq(sec_len, d=1.0 / sample_rate)
    # build result array: rows seconds, columns notes
    mags = np.zeros((n_full_seconds, len(notes)), dtype=float)

    # map each target note to nearest FFT bin indices for harmonics (or empty if > Nyquist)
    nyquist = sample_rate / 2.0
    # default harmonic weights: inverse of harmonic number (1/h)
    if harmonic_weights is None:
        harmonic_weights = [1.0 / h for h in range(1, n_harmonics + 1)]
    else:
        harmonic_weights = list(harmonic_weights)
        if len(harmonic_weights) != n_harmonics:
            raise ValueError('harmonic_weights must have length n_harmonics')

    note_harmonic_bins = []
    for (note_name, note_freq, midi) in notes:
        bins = []
        for h in range(1, n_harmonics + 1):
            hf = note_freq * h
            if hf > nyquist:
                bins.append(-1)
            else:
                idx = int(np.argmin(np.abs(freqs - hf)))
                bins.append(idx)
        note_harmonic_bins.append(bins)

    for s in range(n_full_seconds):
        seg = padded[s * sec_len:(s + 1) * sec_len].astype(float)
        if apply_window:
            seg = seg * np.hanning(len(seg))
        yf = rfft(seg)
        mag = (2.0 / len(seg)) * np.abs(yf)
        for j, bins in enumerate(note_harmonic_bins):
            val = 0.0
            for h_idx, b in enumerate(bins):
                if b == -1:
                    continue
                val += harmonic_weights[h_idx] * mag[b]
            mags[s, j] = val

    note_names = [n for (n, f, m) in notes]
    df = pd.DataFrame(mags, columns=note_names)
    df.index.name = 'second'

    if use_db:
        # convert linear magnitude to decibels (20*log10)
        df = 20.0 * np.log10(df + db_eps)
        if db_floor is not None:
            df = df.clip(lower=float(db_floor))

    return df


def top_k_piano_notes_per_second(data:np.ndarray, sample_rate:int, k:int=10,
                                 apply_window:bool=True, use_db:bool=False,
                                 db_eps:float=1e-12, db_floor:float=None) -> pd.DataFrame:
    """
    Return a DataFrame where each row (second) lists the top-k piano notes and
    magnitudes.

    Args:
        data: np.array audio
        sample_rate: int
        k: int, number of top notes to return per second
        apply_window: bool, whether to apply a Hann window to each 1s segment
        use_db: bool, if True, convert linear magnitudes to decibels (20*log10)
        db_eps: float, small value added before log10 to avoid log(0)
        db_floor: float or None, if set, clip dB values to this minimum

    Returns:
        pandas.DataFrame: rows=seconds, Columns: note_1, mag_1, note_2, mag_2, ..., note_k, mag_k
    """

    notes = piano_note_frequencies()
    df_notes = compute_note_magnitudes_per_second(data, sample_rate, notes=notes,
                                                  apply_window=apply_window,
                                                  n_harmonics=5,
                                                  harmonic_weights=None,
                                                  use_db=use_db,
                                                  db_eps=db_eps,
                                                  db_floor=db_floor)

    n_secs = df_notes.shape[0]
    cols = []
    for i in range(1, k + 1):
        cols.extend([f'note_{i}', f'mag_{i}'])

    out = np.empty((n_secs, 2 * k), dtype=object)

    for s in range(n_secs):
        row = df_notes.values[s]
        # get indices of top-k magnitudes
        idxs = np.argsort(row)[-k:][::-1]
        for rank, idx in enumerate(idxs):
            out[s, 2 * rank] = df_notes.columns[idx]
            out[s, 2 * rank + 1] = float(row[idx])

    df_top = pd.DataFrame(out, columns=cols)
    df_top.index.name = 'second'

    return df_top


def plot_spectrogram_heatmap(df:pd.DataFrame, log_scale:bool=False,
                             clip_percentiles:tuple[int]=(1, 99)) -> go.Heatmap:
    """
    Create a Plotly heatmap for per-second piano note magnitudes from 
    top_k_piano_notes_per_second output.

    Args:
        df: pandas.DataFrame from top_k_piano_notes_per_second
        log_scale: if True, plot log(1 + magnitude) to improve visibility
        clip_percentiles: tuple (low, high) of percentiles to clip color scale for contrast

    Returns:
        plotly.go.Heatmap object
    """

    seconds = df.index.values
    k = len(df.columns) // 2  # number of notes per second
    note_cols = [f'note_{i}' for i in range(1, k + 1)]
    mag_cols = [f'mag_{i}' for i in range(1, k + 1)]
    
    # Get unique notes in order of first appearance
    unique_notes = []
    seen = set()
    for notes in df[note_cols].values:
        for note in notes:
            if note not in seen:
                unique_notes.append(note)
                seen.add(note)
    
    # Create a matrix of zeros (n_notes x n_seconds)
    z = np.zeros((len(unique_notes), len(seconds)))
    
    # Fill in the magnitudes
    for s, sec in enumerate(seconds):
        row = df.iloc[s]
        for i in range(k):
            note = row[f'note_{i+1}']
            mag = row[f'mag_{i+1}']
            note_idx = unique_notes.index(note)
            z[note_idx, s] = mag
            
    if log_scale:
        z_plot = np.log1p(z)
    else:
        z_plot = z
        
    # clip color range for better contrast
    lo, hi = np.percentile(z_plot, [clip_percentiles[0], clip_percentiles[1]])
    
    heat = go.Heatmap(
        x=seconds,
        y=unique_notes,
        z=z_plot,
        colorscale='Viridis',
        zmin=lo,
        zmax=hi,
        colorbar=dict(title='Magnitude' + (' (log1p)' if log_scale else '')),
    )

    return heat

# constants
SONG_DATA_PATH = './data/'
PLOTS_PATH = './plots/'

# load the data files
song_data_files = [f for f in os.listdir(SONG_DATA_PATH) if f.endswith('.wav')]
songs = dict()
for file in song_data_files:
    # get the filename, which is the youtube id
    song_id = file.split('.')[0]
    sample_rate, data = wavfile.read(os.path.join(SONG_DATA_PATH, file))
    print('%s data loaded with sample rate %d and data shape %s'%\
        (song_id, sample_rate, data.shape))

    # get top-10 notes per second using dB ranking
    df_top10_db = top_k_piano_notes_per_second(data, sample_rate, k=10,
                                               use_db=True, db_floor=-80)

    # plot the waveform
    fig = plysub.make_subplots(rows=2, cols=1, 
                              subplot_titles=('Audio Waveform', 'Piano Note Detection'),
                              shared_xaxes=True,
                              vertical_spacing=0.15)

    time = np.linspace(0, len(data)/sample_rate, num=len(data))
    if data.shape[1] == 2:
        # stereo encoding
        fig.add_trace(go.Scatter(x=time, y=data[:,0], mode='lines',
                           name='Left Channel', line={'color':'blue'}), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=data[:,1], mode='lines',
                           name='Right Channel', line={'color':'red'}), row=1, col=1)
    else:
        # mono encoding
        fig.add_trace(go.Scatter(x=time, y=data, mode='lines',
                           name='Audio Signal', line={'color':'blue'}), row=1, col=1)
    
    # plot the piano note detection as a heatmap
    fig.add_trace(plot_spectrogram_heatmap(df=df_top10_db, log_scale=True),
                  row=2, col=1)
    
    # finish with the plot and improve layout
    fig.update_layout(
        title_text='Audio Analysis for %s'%song_id,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=800,
        width = 1200,
    )
    
    # Update axis labels
    fig.update_xaxes(title_text='Time (seconds)', row=2, col=1)  # Only bottom plot needs x label
    fig.update_yaxes(title_text='Amplitude', row=1, col=1)
    fig.update_yaxes(title_text='Piano Note', row=2, col=1)


    plyoff.plot(fig, filename=os.path.join(PLOTS_PATH, '%s_waveform.html'%song_id),
                auto_open=False)
    # save it all
    songs[song_id] = (sample_rate, data, df_top10_db, fig)


'''
TODO
Fourier analysis
Feature Engineering
Save .csv of engineered data
'''
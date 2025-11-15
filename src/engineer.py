import os
from dotenv import dotenv_values
import numpy as np
import pandas as pd
import ipdb

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

    D4_int = 293 # We assume left hand is for A0 - C#3

    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    notes = []
    for m in range(midi_low, midi_high + 1):
        name = names[m % 12]
        octave = (m // 12) - 1
        note_name = f"{name}{octave}"
        freq = 440.0 * (2.0 ** ((m - 69) / 12.0))
        hand = 'L' if m < D4_int else 'R'
        notes.append((note_name, float(freq), int(m), hand))

    return notes


def compute_note_magnitudes_per_interval(data:np.ndarray, sample_rate:int,
                                         notes:list[tuple[str,float,int, str]],
                                         process_params:dict,
                                         harmonic_weights:list[float]=None) -> pd.DataFrame:
    """
    Compute magnitudes per time interval for a list of target note frequencies.

    This function maps each piano note to the nearest FFT bin for each interval's rFFT
    and (optionally) aggregates energy from the first `n_harmonics` harmonics
    for better note detection on instruments like piano.

    Returns a DataFrame with time intervals as the index and one column per note name
    containing the linear magnitude.

    Args:
        data: np.array audio
        sample_rate: int
        notes: iterable of (note_name, freq_hz, midi, hand) tuples.
        harmonic_weights: None or iterable of length n_harmonics giving weights for
            each harmonic (1-based). If None, defaults to 1/h (inverse harmonic)
        process_params: dict, processing parameters including:
            POINTS_PER_SECOND: int, number of time points per second
            TOP_NOTES_KEEP: int, number of top notes to keep per interval
            N_HARMONICS: int, number of harmonics to include
            USE_DB: bool, whether to use decibel scale for magnitudes           

    Returns:
        pandas.DataFrame: rows=time intervals, cols=note names
    """

    interval_seconds = 1.0 / process_params['POINTS_PER_SECOND']
    n_harmonics = process_params['N_HARMONICS']
    use_db = process_params['USE_DB']
    apply_window = process_params['HANN_WINDOW']
    db_eps = 1e-12
    db_floor = None
    
    # mix to mono
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype(float)
    else:
        arr = arr.astype(float)

    interval_len = int(sample_rate * interval_seconds)
    n_intervals = len(arr) // interval_len
    if n_intervals == 0:
        padded = np.pad(arr, (0, max(0, interval_len - len(arr))))
        n_intervals = 1
    else:
        padded = arr[:n_intervals * interval_len]

    freqs = rfftfreq(interval_len, d=1.0 / sample_rate)
    # build result array: rows=intervals, columns=notes
    mags = np.zeros((n_intervals, len(notes)), dtype=float)

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
    for (note_name, note_freq, midi, hand) in notes:
        bins = []
        for h in range(1, n_harmonics + 1):
            hf = note_freq * h
            if hf > nyquist:
                bins.append(-1)
            else:
                idx = int(np.argmin(np.abs(freqs - hf)))
                bins.append(idx)
        note_harmonic_bins.append(bins)

    for i in range(n_intervals):
        seg = padded[i * interval_len:(i + 1) * interval_len].astype(float)
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
            mags[i, j] = val

    note_names = [n for (n, f, m, h) in notes]
    df = pd.DataFrame(mags, columns=note_names)
    df.index.name = 'second'

    if use_db:
        # convert linear magnitude to decibels (20*log10)
        df = 20.0 * np.log10(df + db_eps)
        if db_floor is not None:
            df = df.clip(lower=float(db_floor))

    return df


def top_k_piano_notes_per_interval(data, sample_rate, process_params:dict) -> pd.DataFrame:
    """
    Return a DataFrame where each row (time interval) lists the top-k piano notes and
    magnitudes.

    Args:
        data: np.array audio data
        sample_rate: int, sample rate in Hz
        process_params: dict, processing parameters including:
            TOP_NOTES_KEEP: int, number of top notes to keep per interval

    Returns:
        DataFrame with columns: note_1, mag_1, note_2, mag_2, ..., note_k, mag_k
    """

    notes = piano_note_frequencies()
    df_notes = compute_note_magnitudes_per_interval(data, sample_rate,
                                                    notes=notes,
                                                    process_params=process_params,
                                                    harmonic_weights=None)

    k = process_params['TOP_NOTES_KEEP']
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


def plot_spectrogram_heatmap(df:pd.DataFrame, process_params, log_scale:bool=False,
                             clip_percentiles:tuple[int]=(1, 99)) -> go.Heatmap:
    """
    Create a Plotly heatmap for per-second piano note magnitudes from 
    top_k_piano_notes_per_second output.

    Args:
        df: pandas.DataFrame from top_k_piano_notes_per_second
        process_params: dict, processing parameters including:
            POINTS_PER_SECOND: int, number of time points per second
            TOP_NOTES_KEEP: int, number of top notes to keep per interval
            N_HARMONICS: int, number of harmonics to include
            USE_DB: bool, whether to use decibel scale for magnitudes           
        log_scale: if True, plot log(1 + magnitude) to improve visibility
        clip_percentiles: tuple (low, high) of percentiles to clip color scale for contrast

    Returns:
        plotly.go.Heatmap object
    """

    seconds = df.index.values/process_params['POINTS_PER_SECOND']
    k = len(df.columns) // 2  # number of notes per interval
    note_cols = [f'note_{i}' for i in range(1, k + 1)]

    # Get unique notes and sort them by pitch (lowest to highest)
    unique_notes = sorted(
        list(set(n for row in df[note_cols].values for n in row)),
        key=lambda x: (int(x[1:]) if len(x) == 2 else int(x[2:]),  # octave number
                      'C C# D D# E F F# G G# A A# B'.split().index(x[:-1]))  # note within octave
        )#[::-1]  # reverse to put lowest notes at bottom

    # Create a matrix of zeros (n_notes x n_intervals)
    z = np.zeros((len(unique_notes), len(seconds)))
    for s in range(len(seconds)):
        row = df.iloc[s]
        for i in range(k):
            note = row[f'note_{i+1}']
            mag = float(row[f'mag_{i+1}']) if row[f'mag_{i+1}'] is not None else 0.0
            note_idx = unique_notes.index(note)
            z[note_idx, s] = mag

    z_plot = np.log1p(z) if log_scale else z
    lo, hi = np.percentile(z_plot, [clip_percentiles[0], clip_percentiles[1]])

    heat = go.Heatmap(
        x=seconds,
        y=unique_notes,
        z=z_plot,
        colorscale='Viridis',
        zmin=lo,
        zmax=hi,
        colorbar=dict(title='Magnitude' + (' (log1p)' if log_scale else '')),
        hoverongaps=False,
        hovertemplate=(
            'Time: %{x:.2f}s<br>' +
            'Note: %{y}<br>' +
            'Magnitude: %{z:.1f}' +
            ('<br>(log scale)' if log_scale else '') +
            '<extra></extra>'
        )
    )

    return heat


def load_song_data(input_listing:str, data_path:str) -> tuple[str, dict]:
    '''
    Load song data files based on a CSV listing.

    Args:
        input_listing: str, path to CSV file with song metadata
        data_path: str, directory containing .wav files

    Returns:
        song_listing: pds.DataFrame with song metadata
        song_files: dict mapping youtube_id to (sample_rate, data) tuples
    '''

    # load the csv file and iteratively load them
    song_listing = pd.read_csv(input_listing, parse_dates=['accessed_date'])
    song_files = dict()
    for _, row in song_listing.iterrows():
        # get the data
        song_id = row['youtube_id']
        trim_start = row['trim_stt_sec']
        trim_stop = row['trim_stp_sec']

        # load the wav file
        song_file = os.path.join(data_path, song_id + '.wav')
        sample_rate, data = wavfile.read(song_file)
        print('%s data loaded with sample rate %d and data shape %s'%\
            (song_id, sample_rate, data.shape))

        # Trim audio to include only observations
        start_idx = int(max(0, trim_start) * sample_rate)
        end_idx = int(min(len(data) / sample_rate, trim_stop + 1) * sample_rate)
        if start_idx >= len(data):
            print(f"Trim range starts at {trim_start}s which is beyond file length; skipping file {song_id}")
            continue
        if end_idx <= start_idx:
            print(f"Trim range invalid after clamping; skipping file {song_id}")
            continue
        if end_idx > len(data):
            print(f"Requested end {trim_stop}s beyond file end; clipping to available length")
            end_idx = len(data)
        data = data[start_idx:end_idx]
        print(f"Trimmed {song_id} to samples {start_idx}:{end_idx} ({(end_idx-start_idx)/sample_rate:.2f}s)")

        # save it
        song_files[song_id] = (sample_rate, data, trim_start, trim_stop)

    return song_listing, song_files


def process_song_file(data:np.ndarray, sample_rate:int, song_id:str,
                      start_sec:int, stop_sec:int, process_params:dict) -> tuple[pd.DataFrame, go.Figure]:
    '''
    Process input song file data into notes and generate plots.

    Args:
        data: np.array audio data
        sample_rate: int, sample rate in Hz
        song_id: str, unique identifier for the song
        start_sec: int, starting second of the trimmed audio
        stop_sec: int, stopping second of the trimmed audio
        process_params: dict, processing parameters including:

    Returns:
        df_topN_db: DataFrame of top-K notes per interval
        fig: Plotly figure object
    '''

    # get top-K notes per quarter-second using dB ranking
    resolution_name = {1:'', 2:'Half-', 4:'Quarter-'}[process_params['POINTS_PER_SECOND']]
    df_topN_db = top_k_piano_notes_per_interval(data, sample_rate, process_params)

    # plot the waveform and a single combined note heatmap
    fig = plysub.make_subplots(rows=2, cols=1, row_heights=[0.25, 0.75],
                               shared_xaxes=True,
                               vertical_spacing=0.1,
                               subplot_titles=('Waveform',
                                               '%sSecond Resolution Top-%d Piano Notes Spectrogram'%\
                                                (resolution_name,process_params['TOP_NOTES_KEEP'])))

    time = np.linspace(0, len(data)/sample_rate, num=len(data))
    if data.ndim == 2 and data.shape[1] == 2:
        # stereo encoding: plot both channels in the waveform panel
        fig.add_trace(go.Scatter(x=time, y=data[:,0], mode='lines',
                           name='Left Channel', line={'color':'blue'},
                           hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>'),
                           row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=data[:,1], mode='lines',
                           name='Right Channel', line={'color':'red'},
                           hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>'),
                           row=1, col=1)
    else:
        # mono encoding
        fig.add_trace(go.Scatter(x=time, y=data, mode='lines',
                           name='Audio Signal', line={'color':'blue'},
                           hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>'),
                           row=1, col=1)

    # build a single heatmap from the top-k DataFrame
    heat = plot_spectrogram_heatmap(df_topN_db, process_params, log_scale=True)
    fig.add_trace(heat, row=2, col=1)

    # finish with the plot and improve layout
    fig.update_layout(
        title_text='Audio Analysis for %s (seconds %d to %d)'%\
            (song_id, start_sec, stop_sec),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=800,
        width=1200,
    )

    # Update axis labels
    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_yaxes(title_text='Amplitude', row=1, col=1)
    fig.update_yaxes(title_text='Note', row=2, col=1)

    return df_topN_db, fig


if __name__ == '__main__':
    # load environment variables from .env file
    config = dotenv_values('./settings.env')
    SONG_DATA_PATH = config['SONG_DATA_PATH']
    PLOTS_PATH = config['PLOTS_PATH']
    ENG_DATA_PATH = config['ENG_DATA_PATH']
    SONG_LISTING_FILE = config['SONG_LISTING_FILE']

    process_params={'POINTS_PER_SECOND':int(config['POINTS_PER_SECOND']),
                    'TOP_NOTES_KEEP':int(config['TOP_NOTES_KEEP']),
                    'N_HARMONICS':int(config['N_HARMONICS']),
                    'USE_DB':('True'==config['USE_DB']),
                    'HANN_WINDOW':('True'==config['HANN_WINDOW'])}
    param_str = 'pps%d_topN%d_harm%d_db%d_hann%d'%\
        (process_params['POINTS_PER_SECOND'],
         process_params['TOP_NOTES_KEEP'],
         process_params['N_HARMONICS'],
         process_params['USE_DB'],
         process_params['HANN_WINDOW'])

    # load the songs and process them to notes listing
    song_listing, song_raw_data = load_song_data(SONG_LISTING_FILE, SONG_DATA_PATH)
    for (song_id, song_data) in song_raw_data.items():
        print("Processing song %s..."%song_id)
        sample_rate, data, start_sec, end_sec = song_data
        df_topN_db, fig = process_song_file(data, sample_rate, song_id, start_sec,
                                            end_sec, process_params)
        # do more stuff

        # save the plot &  engineered data
        plyoff.plot(fig, filename=os.path.join(PLOTS_PATH, '%s_%s.html'%(song_id, param_str)),
                    auto_open=False)
        df_topN_db.to_csv(os.path.join(ENG_DATA_PATH, '%s_%s_notes.csv'%(song_id, param_str)),
                          index_label='second')

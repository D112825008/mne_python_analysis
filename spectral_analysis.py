"""
頻譜分析模組（全腦分析）

版本: 1.2 - 修正頻率解析度問題
修改內容：
  - 提高 FFT 的 n_fft 參數，使用 zero-padding 提高頻率解析度
  - 原本問題：200ms 窗口 → 64 點 FFT → 8 Hz 解析度 → Theta 和 Alpha 無法區分
  - 修正後：使用 2 秒 n_fft (1024 點) → 0.5 Hz 解析度
"""

import numpy as np
import mne
from mne.time_frequency import psd_array_welch


def compute_fft_power(epochs, fmin, fmax, tmin=None, tmax=None):
    """
    計算 FFT 功率（所有電極）
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 資料
    fmin, fmax : float
        頻率範圍
    tmin, tmax : float or None
        時間窗口（None 表示使用整個 epoch）
        
    Returns
    -------
    power : ndarray
        功率值 (n_epochs, n_channels)
    """
    # 選擇時間窗口
    if tmin is not None and tmax is not None:
        epochs_crop = epochs.copy().crop(tmin=tmin, tmax=tmax)
    else:
        epochs_crop = epochs.copy()
    
    # 取得資料
    data = epochs_crop.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs_crop.info['sfreq']
    n_times = data.shape[2]
    
    # 使用較大的 n_fft 以提高頻率解析度
    # 即使 n_fft > n_times，psd_array_welch 也會自動 zero-padding
    n_fft = n_times  # 2 秒 → 512 Hz * 2 = 1024 點 → 0.5 Hz 解析度
    
    # n_per_seg 設為實際資料長度，不分段
    n_per_seg = n_times
    n_overlap = 0
    
#    print(f"[DEBUG compute_fft_power] n_times={n_times}, n_fft={n_fft}, freq_resolution={sfreq/n_fft:.2f} Hz")
    # ===================================
    
    # 計算 PSD
    psds, freqs = psd_array_welch(
        data, sfreq=sfreq,
        fmin=0, fmax=sfreq/2,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap
    )
    
    # 平均頻率範圍內的功率
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    
    if not freq_mask.any():
        raise ValueError(f"No frequencies found in range {fmin}-{fmax} Hz")
    
    # DEBUG: 顯示選擇的頻率
#    selected_freqs = freqs[freq_mask]
#    print(f"[DEBUG compute_fft_power] {fmin}-{fmax} Hz: {len(selected_freqs)} frequency bins selected")
#    print(f"  Frequency bins: {selected_freqs[:5]} ... {selected_freqs[-5:]}" if len(selected_freqs) > 10 else f"  Frequency bins: {selected_freqs}")
    
    power = np.mean(psds[:, :, freq_mask], axis=2)  # (n_epochs, n_channels)
    
    return power


def compute_band_power_multiple_bands(epochs, freq_bands, tmin=None, tmax=None):
    """
    計算多個頻段的功率（所有電極）
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 資料
    freq_bands : dict
        頻段定義，例如 {'theta': (4, 8), 'alpha': (8, 13)}
    tmin, tmax : float or None
        時間窗口
        
    Returns
    -------
    band_powers : dict
        {頻段名稱: ndarray (n_epochs, n_channels)}
    """
   # 選擇時間窗口
    if tmin is not None and tmax is not None:
        epochs_crop = epochs.copy().crop(tmin=tmin, tmax=tmax)
    else:
        epochs_crop = epochs.copy()
    
    # 取得資料
    data = epochs_crop.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs_crop.info['sfreq']
    n_times = data.shape[2]
    
    # ===== 修正：提高頻率解析度 =====
    n_fft = int(sfreq * 2)
    n_per_seg = n_times
    n_overlap = 0
    # ===================================
    
    band_powers = {}
    
    for band_name, (fmin, fmax) in freq_bands.items():
        # 計算所有頻率
        psds, freqs = psd_array_welch(
            data, sfreq=sfreq,
            fmin=0, fmax=sfreq/2,
            n_fft=n_fft,
            n_per_seg=n_per_seg,
            n_overlap=n_overlap
        )
        
        # 手動選擇頻率範圍
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        
        if not freq_mask.any():
            raise ValueError(f"No frequencies found in range {fmin}-{fmax} Hz for band {band_name}")
        
        band_power = np.mean(psds[:, :, freq_mask], axis=2)  # (n_epochs, n_channels)
        band_powers[band_name] = band_power
    
    return band_powers



def compute_power_with_freq_baseline(epochs, fmin, fmax, 
                                     task_tmin, task_tmax,
                                     baseline_tmin, baseline_tmax,
                                     method='relative'):
    """
    計算功率並進行 frequency-domain baseline correction
    
    這是兩階段 baseline correction：
    1. Time-domain: 已在 epoch 創建時完成
    2. Frequency-domain: 此函數執行
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 資料（已經過 time-domain baseline correction）
    fmin, fmax : float
        頻率範圍（Hz）
    task_tmin, task_tmax : float
        任務窗口時間（秒）
    baseline_tmin, baseline_tmax : float
        Baseline 窗口時間（秒）
    method : str
        Baseline correction 方法：
        - 'relative': (task - baseline) / baseline
        - 'percent': (task - baseline) / baseline * 100
        - 'db': 10 * log10(task / baseline)
        - 'zscore': (task - baseline) / std(baseline)
        
    Returns
    -------
    corrected_power : ndarray
        校正後的功率 (n_epochs, n_channels)
    baseline_power : ndarray
        Baseline 功率 (n_epochs, n_channels)
    task_power : ndarray
        任務功率 (n_epochs, n_channels)
    """
#    print(f"[DEBUG compute_power_with_freq_baseline] Computing {fmin}-{fmax} Hz")
    
    # 1. 計算 baseline 期的功率
    baseline_power = compute_fft_power(epochs, fmin, fmax, 
                                      baseline_tmin, baseline_tmax)
    print(f"  Baseline power: mean={np.mean(baseline_power):.4e}, shape={baseline_power.shape}")
    
    # 2. 計算任務期的功率
    task_power = compute_fft_power(epochs, fmin, fmax,
                                   task_tmin, task_tmax)
    print(f"  Task power: mean={np.mean(task_power):.4e}, shape={task_power.shape}")
    
    # 3. Frequency-domain baseline correction
    if method == 'relative':
        # 相對變化
        corrected_power = (task_power - baseline_power) / baseline_power
    elif method == 'percent':
        # 百分比變化
        corrected_power = (task_power - baseline_power) / baseline_power * 100
    elif method == 'db':
        # dB 轉換
        corrected_power = 10 * np.log10(task_power / baseline_power)
    elif method == 'zscore':
        # Z-score 標準化
        baseline_std = np.std(baseline_power, axis=0, keepdims=True)
        corrected_power = (task_power - baseline_power) / baseline_std
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"  Corrected power ({method}): mean={np.mean(corrected_power):.4e}")
    
    return corrected_power, baseline_power, task_power


def compute_tfr_morlet(epochs, freqs, n_cycles='auto', average=False, return_itc=False):
    """
    計算時頻分析（Morlet wavelet）
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 資料
    freqs : array-like
        頻率陣列
    n_cycles : float or array-like or 'auto'
        Wavelet 週期數
    average : bool
        是否平均所有 epochs
    return_itc : bool
        是否返回 ITC
        
    Returns
    -------
    power : mne.time_frequency.AverageTFR or mne.time_frequency.EpochsTFR
        時頻功率
    itc : mne.time_frequency.AverageTFR or None
        ITC（如果 return_itc=True）
    """
    from mne.time_frequency import tfr_morlet
    
    if n_cycles == 'auto':
        n_cycles = freqs / 2.0
    
    power = tfr_morlet(
        epochs, freqs=freqs, n_cycles=n_cycles,
        use_fft=True, return_itc=return_itc,
        average=average, n_jobs=-1
    )
    
    return power


def exclude_channels_from_epochs(epochs, exclude_list=['HEOG', 'VEOG', 'A1', 'A2']):
    """
    排除特定電極（用於全腦分析）
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 資料
    exclude_list : list
        要排除的電極名稱
        
    Returns
    -------
    epochs_clean : mne.Epochs
        排除特定電極後的 epochs
    excluded_channels : list
        實際被排除的電極
    """
    # 找出要排除且確實存在的電極
    excluded_channels = [ch for ch in exclude_list if ch in epochs.ch_names]
    
    # 找出保留的電極
    keep_channels = [ch for ch in epochs.ch_names 
                    if ch not in exclude_list and 
                    epochs.get_channel_types([ch])[0] == 'eeg']
    
    if len(keep_channels) == 0:
        raise ValueError("No EEG channels left after exclusion!")
    
    epochs_clean = epochs.copy().pick_channels(keep_channels)
    
    print(f"✓ 排除電極: {excluded_channels}")
    print(f"✓ 保留 {len(keep_channels)} 個 EEG 電極用於全腦分析")
    
    return epochs_clean, excluded_channels

def compute_roi_power_with_freq_baseline(epochs, roi_channels,
                                         fmin, fmax,
                                         task_tmin, task_tmax,
                                         baseline_tmin, baseline_tmax,
                                         method='relative'):
    """
    計算指定 ROI 的功率並做 frequency-domain baseline correction，
    然後在 channel 軸做平均，輸出單一 ROI 的功率。

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 資料（已做過 time-domain baseline 的前處理）
    roi_channels : list of str
        ROI 內的電極名稱，例如 ['Fz', 'FCz', 'Cz', 'C3', 'C4']
    fmin, fmax : float
        頻率範圍（Hz）
    task_tmin, task_tmax : float
        任務時間窗口（秒）
    baseline_tmin, baseline_tmax : float
        baseline 時間窗口（秒）
    method : str
        同 compute_power_with_freq_baseline 的 method 參數

    Returns
    -------
    roi_corrected : ndarray, shape (n_epochs,)
        ROI 平均後的校正功率
    roi_task : ndarray, shape (n_epochs,)
        ROI 平均後的任務期功率
    roi_baseline : ndarray, shape (n_epochs,)
        ROI 平均後的 baseline 功率
    """
    # 只保留 ROI 中「實際存在於 epochs」的通道
    roi_channels_existing = [ch for ch in roi_channels if ch in epochs.ch_names]
    if len(roi_channels_existing) == 0:
        raise ValueError(f"ROI channels not found in epochs: {roi_channels}")

    # 複製一份 epochs，挑出 ROI channels
    epochs_roi = epochs.copy().pick_channels(roi_channels_existing)

    # 使用既有的 frequency-domain baseline 函數
    corrected_power, baseline_power, task_power = compute_power_with_freq_baseline(
        epochs_roi,
        fmin=fmin, fmax=fmax,
        task_tmin=task_tmin, task_tmax=task_tmax,
        baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
        method=method
    )
    # 目前形狀是 (n_epochs, n_channels)，在 channel 軸平均，得到每個 epoch 一個 ROI 值
    roi_corrected = np.mean(corrected_power, axis=1)   # (n_epochs,)
    roi_task = np.mean(task_power, axis=1)             # (n_epochs,)
    roi_baseline = np.mean(baseline_power, axis=1)     # (n_epochs,)

    return roi_corrected, roi_task, roi_baseline


def exclude_non_eeg_channels(epochs, exclude_list=['HEOG', 'VEOG', 'A1', 'A2']):
    """
    包一層，符合 main.py 的呼叫方式：
    - 輸入：epochs
    - 輸出：只保留 EEG、且排除指定通道的 epochs

    Parameters
    ----------
    epochs : mne.Epochs
        原始 epochs
    exclude_list : list of str
        要排除的通道名稱（預設 ['HEOG', 'VEOG', 'A1', 'A2']）

    Returns
    -------
    epochs_clean : mne.Epochs
        已排除非 EEG / 指定通道後的 epochs
    """
    epochs_clean, excluded_channels = exclude_channels_from_epochs(
        epochs, exclude_list=exclude_list
    )
    # 這裡不特別回傳 excluded_channels，因為 main.py 目前只需要乾淨的 epochs
    return epochs_clean
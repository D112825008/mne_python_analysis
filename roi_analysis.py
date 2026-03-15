"""
ROI 分析模組
定義和分析腦區 ROI（Region of Interest）

版本: 1.1 - 修正 Alpha ROI 定義（符合 ASRT 實驗需求）
修改內容：Alpha ROI 從 9 個電極改為 6 個（O1, Oz, O2, P3, Pz, P4）
"""

import numpy as np
import mne


def define_roi_channels(roi_name):
    """
    定義 ROI 電極（根據實際電極布局）
    
    實際電極列表：
    - 前額: Fp1, Fp2
    - 額葉: F7, F3, Fz, F4, F8
    - 額-顳: FT7, FT8
    - 額-中央: FC3, FCz, FC4
    - 顳葉: T3, T4, T5, T6
    - 中央: C3, Cz, C4
    - 顳-頂: TP7, TP8
    - 中央-頂: CP3, CPz, CP4
    - 頂葉: P3, Pz, P4
    - 枕葉: O1, Oz, O2
    - 參考: A1, A2
    
    Parameters
    ----------
    roi_name : str
        ROI 名稱：'theta', 'alpha', 'frontal', 'central', 'parietal', 'occipital', 'temporal'
        
    Returns
    -------
    channels : list
        電極名稱列表
    """
    roi_definitions = {
        # 實驗用主要 ROI
        'theta': ['Fz', 'FCz', 'Cz', 'C3', 'C4'],
        'alpha': ['O1', 'Oz', 'O2', 'P3', 'Pz', 'P4'],
        
        # 其他預定義 ROI
        'frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
        'frontocentral': ['FC3', 'FCz', 'FC4'],
        'central': ['C3', 'C4', 'Cz'],
        'centroparietal': ['CP3', 'CPz', 'CP4'],
        'parietal': ['P3', 'P4', 'Pz'],
        'occipital': ['O1', 'O2', 'Oz'],  # ← 修正：移除 PO3, PO4, POz
        'temporal': ['T3', 'T4', 'T5', 'T6'],
        'frontotemporal': ['FT7', 'FT8'],
        'temporoparietal': ['TP7', 'TP8'],
        
        # 組合 ROI
        'parieto_occipital': ['P3', 'Pz', 'P4', 'O1', 'Oz', 'O2'],
        'frontal_central': ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4']
    }
    
    if roi_name.lower() not in roi_definitions:
        raise ValueError(f"Unknown ROI: {roi_name}. Available: {list(roi_definitions.keys())}")
    
    return roi_definitions[roi_name.lower()]


def average_roi_epochs(epochs, roi_channels, tmin, tmax, fmin, fmax):
    """
    平均 ROI 電極並計算功率
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 資料
    roi_channels : list
        ROI 電極名稱
    tmin, tmax : float
        時間窗口
    fmin, fmax : float
        頻率範圍
        
    Returns
    -------
    roi_power : ndarray
        ROI 平均功率 (n_epochs,)
    """
    from mne.time_frequency import psd_array_welch
    
    # 選擇 ROI 電極
    available_channels = [ch for ch in roi_channels if ch in epochs.ch_names]
    if len(available_channels) == 0:
        raise ValueError(f"None of the ROI channels found in epochs: {roi_channels}")
    
    # 檢查是否有缺失的電極
    missing_channels = [ch for ch in roi_channels if ch not in epochs.ch_names]
    if missing_channels:
        print(f"⚠️ 警告：以下 ROI 電極不存在，已跳過: {missing_channels}")
        print(f"   使用電極: {available_channels}")
    
    epochs_roi = epochs.copy().pick_channels(available_channels)
    
    # 裁切時間
    epochs_roi = epochs_roi.crop(tmin=tmin, tmax=tmax)
    
    # 取得資料並平均電極
    data = epochs_roi.get_data()  # (n_epochs, n_channels, n_times)
    data_avg = np.mean(data, axis=1)  # (n_epochs, n_times)
    
    # 計算功率
    sfreq = epochs_roi.info['sfreq']
    psds, freqs = psd_array_welch(
        data_avg[:, np.newaxis, :],  # 需要 (n_epochs, 1, n_times)
        sfreq=sfreq,
        fmin=fmin, fmax=fmax,
        n_fft=int(sfreq),
        n_overlap=int(sfreq * 0.5)
    )
    
    # 平均頻率
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    roi_power = np.mean(psds[:, 0, freq_mask], axis=1)  # (n_epochs,)
    
    return roi_power


def create_virtual_channel_epochs(epochs, roi_channels, virtual_ch_name='ROI'):
    """
    創建虛擬電極（ROI 平均）
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 資料
    roi_channels : list
        ROI 電極名稱
    virtual_ch_name : str
        虛擬電極名稱
        
    Returns
    -------
    epochs_virtual : mne.Epochs
        只包含虛擬電極的 epochs
    """
    # 選擇 ROI 電極
    available_channels = [ch for ch in roi_channels if ch in epochs.ch_names]
    
    # 檢查缺失電極
    missing_channels = [ch for ch in roi_channels if ch not in epochs.ch_names]
    if missing_channels:
        print(f"⚠️ 虛擬電極：以下電極不存在，已跳過: {missing_channels}")
    
    epochs_roi = epochs.copy().pick_channels(available_channels)
    
    # 平均電極
    data = epochs_roi.get_data()  # (n_epochs, n_channels, n_times)
    data_avg = np.mean(data, axis=1, keepdims=True)  # (n_epochs, 1, n_times)
    
    # 創建新的 info
    info = mne.create_info(
        ch_names=[virtual_ch_name],
        sfreq=epochs.info['sfreq'],
        ch_types='eeg'
    )
    
    # 創建新的 epochs
    epochs_virtual = mne.EpochsArray(data_avg, info, tmin=epochs.tmin)
    
    return epochs_virtual


def compare_roi_across_conditions(epochs_dict, roi_channels, tmin, tmax, fmin, fmax):
    """
    比較不同條件下的 ROI 活動
    
    Parameters
    ----------
    epochs_dict : dict
        {condition_name: epochs}
    roi_channels : list
        ROI 電極
    tmin, tmax : float
        時間窗口
    fmin, fmax : float
        頻率範圍
        
    Returns
    -------
    results : dict
        {condition_name: power_array}
    """
    results = {}
    
    for condition, epochs in epochs_dict.items():
        power = average_roi_epochs(epochs, roi_channels, tmin, tmax, fmin, fmax)
        results[condition] = power
        print(f"{condition}: M={np.mean(power):.4e}, SD={np.std(power):.4e}")
    
    return results


def validate_electrode_availability(epochs, required_electrodes):
    """
    驗證所需電極是否都存在
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 資料
    required_electrodes : list
        需要的電極列表
        
    Returns
    -------
    available : list
        可用的電極
    missing : list
        缺失的電極
    """
    available = [ch for ch in required_electrodes if ch in epochs.ch_names]
    missing = [ch for ch in required_electrodes if ch not in epochs.ch_names]
    
    return available, missing
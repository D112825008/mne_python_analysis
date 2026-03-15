"""
訊號處理模組 - Signal Processing Module

包含EEG信號濾波與參考等處理功能。

版本: 1.1 - 修改 Average Reference 邏輯
修改內容：
  - apply_average_reference() 現在可以排除特定電極（HEOG/VEOG/A1/A2）
"""

import mne


#def apply_cz_reference(raw):
#    """
#    應用Cz (B16) offline reference。
    
#    參數:
#        raw (mne.io.Raw): 要處理的Raw物件
    
#    返回:
#        mne.io.Raw: 已應用Cz參考的Raw物件
#    """
#    print("\n進行Cz (B16) offline reference...")
    
#    raw_cz_ref = raw.copy()
    
    # 確保有Cz或B16通道
#    if 'Cz' in raw.ch_names:
#        ref_channel = 'Cz'
#    elif 'B16' in raw.ch_names:
#        ref_channel = 'B16'
#    else:
#        print("警告: 找不到Cz或是B16通道，無法進行Cz reference")
#        return raw_cz_ref
    
#    # 應用Cz參考
#    raw_cz_ref.filter(1, 40).set_eeg_reference(ref_channels=[ref_channel])
#    
#    print(f"成功應用{ref_channel}參考")
#    
#    return raw_cz_ref

def apply_linked_mastoid_reference(raw):
    """
    應用 Linked Mastoids Reference (M1/M2 或 A1/A2)
    
    自動偵測使用 M1/M2 或 A1/A2：
    - 優先使用 M1/M2
    - 若找不到，使用 A1/A2
    
    參數:
        raw (mne.io.Raw): 要處理的Raw物件
    
    返回:
        mne.io.Raw: 已應用 Linked Mastoids Reference 的Raw物件
    """
    print("\n應用 Linked Mastoids Reference...")
    raw_mastoid_ref = raw.copy()
    
    # 檢查可用的 mastoid 電極
    has_m1_m2 = 'M1' in raw_mastoid_ref.ch_names and 'M2' in raw_mastoid_ref.ch_names
    has_a1_a2 = 'A1' in raw_mastoid_ref.ch_names and 'A2' in raw_mastoid_ref.ch_names
    
    if has_m1_m2:
        ref_channels = ['M1', 'M2']
        print("✓ 使用 M1/M2 作為 Reference")
    elif has_a1_a2:
        ref_channels = ['A1', 'A2']
        print("✓ 使用 A1/A2 作為 Reference")
    else:
        print("❌ 錯誤：找不到 M1/M2 或 A1/A2 電極")
        print(f"可用電極: {raw_mastoid_ref.ch_names}")
        return raw_mastoid_ref
    
    # 應用 Linked Mastoids Reference
    raw_mastoid_ref.set_eeg_reference(ref_channels=ref_channels, projection=False)
    print(f"✓ 成功應用 {'/'.join(ref_channels)} Reference")
    
    return raw_mastoid_ref


def apply_single_electrode_reference(raw, ref_channel):
    """
    應用單一電極 Reference
    
    參數:
        raw (mne.io.Raw): 要處理的Raw物件
        ref_channel (str): 要作為 Reference 的電極名稱
    
    返回:
        mne.io.Raw: 已應用單一電極 Reference 的Raw物件
    """
    print(f"\n應用 {ref_channel} Reference...")
    raw_single_ref = raw.copy()
    
    # 檢查電極是否存在
    if ref_channel not in raw_single_ref.ch_names:
        print(f"❌ 錯誤：找不到電極 {ref_channel}")
        print(f"可用電極: {raw_single_ref.ch_names}")
        return raw_single_ref
    
    # 應用單一電極 Reference
    raw_single_ref.set_eeg_reference(ref_channels=[ref_channel], projection=False)
    print(f"✓ 成功應用 {ref_channel} Reference")
    
    return raw_single_ref


def apply_average_reference(raw, exclude_channels=['HEOG', 'VEOG', 'A1', 'A2']):
    """
    應用平均參考 (average reference)，排除特定電極。
    
    參數:
        raw (mne.io.Raw): 要處理的Raw物件
        exclude_channels (list): 要排除的電極名稱（預設：['HEOG', 'VEOG', 'A1', 'A2']）
    
    返回:
        mne.io.Raw: 已應用平均參考的Raw物件
    """
    print("\n應用 Average Reference（排除特定電極）...")
    raw_avg_ref = raw.copy()
    
    # 找出要排除且確實存在的電極
    excluded = [ch for ch in exclude_channels if ch in raw_avg_ref.ch_names]
    
    # 找出用於 average reference 的 EEG 電極
    eeg_channels = [ch for ch in raw_avg_ref.ch_names 
                   if raw_avg_ref.get_channel_types([ch])[0] == 'eeg' and 
                   ch not in exclude_channels]
    
    if len(eeg_channels) == 0:
        print("⚠️  警告：沒有可用的 EEG 電極做 average reference")
        return raw_avg_ref
    
    print(f"✓ 排除電極: {excluded}")
    print(f"✓ 使用 {len(eeg_channels)} 個 EEG 電極計算 average reference")
    
    # 應用 average reference（只使用 EEG 電極）
    raw_avg_ref.set_eeg_reference(ref_channels=eeg_channels, projection=False)
    
    print("✓ 成功應用 average reference")
    
    return raw_avg_ref


def apply_filters(raw, l_freq=1.0, h_freq=40.0, notch_freq=60):
    """
    應用高通、低通和notch濾波器。
    
    參數:
        raw (mne.io.Raw): 要處理的Raw物件
        l_freq (float): 高通濾波器頻率，預設為1.0 Hz
        h_freq (float): 低通濾波器頻率，預設為40.0 Hz
        notch_freq (int): notch濾波器頻率，預設為60 Hz (電源干擾)
    
    返回:
        mne.io.Raw: 已濾波的Raw物件
    """
    raw_filtered = raw.copy()
    
    # 應用高通/低通濾波器
    if l_freq is not None or h_freq is not None:
        print(f"應用濾波器 (l_freq={l_freq} Hz, h_freq={h_freq} Hz)...")
        raw_filtered.filter(l_freq=l_freq, h_freq=h_freq)
    
    # 應用notch濾波器
    if notch_freq is not None:
        print(f"應用notch濾波器 ({notch_freq} Hz)...")
        raw_filtered.notch_filter(freqs=[notch_freq])
    
    print("濾波完成")
    return raw_filtered


def resample_data(raw, sfreq=500):
    """
    重新採樣Raw數據。
    
    參數:
        raw (mne.io.Raw): 要重新採樣的Raw物件
        sfreq (float): 目標採樣率，預設為500 Hz
    
    返回:
        mne.io.Raw: 已重新採樣的Raw物件
    """
    if raw.info['sfreq'] != sfreq:
        print(f"將採樣率從{raw.info['sfreq']}Hz調整為{sfreq}Hz...")
        raw_resampled = raw.copy().resample(sfreq=sfreq)
        print("重新採樣完成")
        return raw_resampled
    else:
        print(f"採樣率已經是{sfreq}Hz，無需調整")
        return raw.copy()
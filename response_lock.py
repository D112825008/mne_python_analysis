"""
Response-locked 分析模組
"""

import numpy as np
import mne
from .spectral_analysis import compute_fft_power


def align_epochs_to_response(epochs_stim, response_times_ms):
    """
    將 stimulus-locked epochs 重新對齊到反應時間
    
    注意：這是概念性函數，實際應該在 raw 層級重新 epoch
    
    Parameters
    ----------
    epochs_stim : mne.Epochs
        Stimulus-locked epochs
    response_times_ms : ndarray
        反應時間（毫秒）
        
    Returns
    -------
    epochs_resp : mne.Epochs
        Response-locked epochs（需要重新從 raw 創建）
    """
    print("⚠️ 警告：請使用 create_response_locked_epochs 從 raw 創建")
    print("   此函數僅供參考")
    return None


def compute_response_locked_power(epochs_resp, roi_channels, fmin, fmax, tmin=-0.75, tmax=0.3):
    """
    計算 response-locked 功率
    
    Parameters
    ----------
    epochs_resp : mne.Epochs
        Response-locked epochs
    roi_channels : list
        ROI 電極
    fmin, fmax : float
        頻率範圍
    tmin, tmax : float
        時間窗口（相對反應）
        
    Returns
    -------
    power : ndarray
        功率值 (n_epochs,)
    """
    from .roi_analysis import average_roi_epochs
    
    power = average_roi_epochs(epochs_resp, roi_channels, tmin, tmax, fmin, fmax)
    
    return power


def validate_response_times(response_times_ms, min_rt=100, max_rt=2000):
    """
    驗證反應時間的合理性
    
    Parameters
    ----------
    response_times_ms : ndarray
        反應時間（毫秒）
    min_rt, max_rt : float
        合理的 RT 範圍（毫秒）
        
    Returns
    -------
    valid_mask : ndarray (bool)
        有效的 trials
    invalid_indices : list
        異常的 trial 索引
    """
    valid_mask = (response_times_ms >= min_rt) & (response_times_ms <= max_rt)
    invalid_indices = np.where(~valid_mask)[0]
    
    print(f"\nRT 驗證結果:")
    print(f"  有效 trials: {np.sum(valid_mask)} / {len(response_times_ms)}")
    print(f"  RT 範圍: {np.min(response_times_ms[valid_mask]):.1f} - {np.max(response_times_ms[valid_mask]):.1f} ms")
    
    if len(invalid_indices) > 0:
        print(f"  ⚠️ 異常 trials: {len(invalid_indices)}")
        print(f"     索引: {invalid_indices[:10]}{'...' if len(invalid_indices) > 10 else ''}")
    
    return valid_mask, invalid_indices
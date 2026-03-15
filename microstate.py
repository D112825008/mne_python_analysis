"""
微狀態分析模組 - Microstate Module

包含EEG微狀態分析相關功能。
"""

import mne
import matplotlib.pyplot as plt


def prepare_for_microstate(raw):
    """
    準備微狀態分析數據。
    
    參數:
        raw (mne.io.Raw): 要處理的Raw物件
    
    返回:
        mne.io.Raw: 準備好的微狀態分析數據
    """
    print("\n準備microstate data analysis...")
    
    # 複製原始數據
    raw_ms = raw.copy()
    
    # 應用2-20Hz濾波
    print("應用2-20Hz濾波...")
    raw_ms.filter(l_freq=2.0, h_freq=20.0)
    
    # 顯示濾波後的PSD以確認
    raw_ms.plot_psd(fmax=50)
    plt.show()
    
    print("Microstate數據準備完成")
    return raw_ms


def segment_into_microstates(raw_ms, n_microstates=4, random_state=42):
    """
    將EEG數據分割為微狀態。
    
    參數:
        raw_ms (mne.io.Raw): 準備好的微狀態Raw物件
        n_microstates (int): 微狀態數量，預設為4
        random_state (int): 隨機數種子，預設為42
    
    返回:
        dict: 微狀態分析結果
    
    注意: 此函數需要安裝MNE-microstates插件
    """
    try:
        # 嘗試導入MNE-microstates
        import mne_microstates
    except ImportError:
        print("需要安裝MNE-microstates插件。請執行: pip install mne-microstates")
        return None
    
    print(f"使用{n_microstates}個微狀態進行分析...")
    
    # 準備數據
    data = raw_ms.get_data(picks='eeg')
    times = raw_ms.times
    
    # 使用MNE-microstates進行分析
    microstate_model = mne_microstates.Microstates(
        n_states=n_microstates,
        random_state=random_state
    )
    
    # 擬合模型
    microstate_model.fit(data)
    
    # 獲取結果
    microstate_labels = microstate_model.labels_
    microstate_maps = microstate_model.maps_
    
    # 計算GEV (Global Explained Variance)
    gev = microstate_model.gev_
    gev_total = microstate_model.gev_total_
    
    # 顯示微狀態地形圖
    microstate_model.plot_maps(raw_ms.info)
    
    # 結果儲存
    result = {
        'model': microstate_model,
        'labels': microstate_labels,
        'maps': microstate_maps,
        'gev': gev,
        'gev_total': gev_total
    }
    
    print(f"微狀態分析完成。總GEV: {gev_total:.3f}")
    return result


def calculate_microstate_stats(microstate_result, fs=None):
    """
    計算微狀態統計資訊。
    
    參數:
        microstate_result (dict): 微狀態分析結果
        fs (float, optional): 採樣率，默認為None (從原始數據獲取)
    
    返回:
        dict: 微狀態統計資訊
    """
    if microstate_result is None:
        print("沒有可用的微狀態分析結果")
        return None
    
    import numpy as np
    
    # 獲取標籤
    labels = microstate_result['labels']
    
    # 計算各狀態持續時間和發生頻率
    unique_states = np.unique(labels)
    state_counts = {}
    state_durations = {}
    
    # 建立狀態切換矩陣
    transitions = np.zeros((len(unique_states), len(unique_states)))
    
    # 計算狀態持續時間和轉換
    current_state = labels[0]
    current_duration = 1
    
    for i in range(1, len(labels)):
        if labels[i] == current_state:
            current_duration += 1
        else:
            # 記錄狀態持續時間
            if current_state not in state_durations:
                state_durations[current_state] = []
            state_durations[current_state].append(current_duration)
            
            # 記錄狀態轉換
            prev_idx = np.where(unique_states == current_state)[0][0]
            next_idx = np.where(unique_states == labels[i])[0][0]
            transitions[prev_idx, next_idx] += 1
            
            # 重設計數器
            current_state = labels[i]
            current_duration = 1
    
    # 記錄最後一個狀態
    if current_state not in state_durations:
        state_durations[current_state] = []
    state_durations[current_state].append(current_duration)
    
    # 計算統計量
    stats = {
        'coverage': {},  # 覆蓋率
        'mean_duration': {},  # 平均持續時間
        'occurrence': {},  # 發生頻率
        'transition_matrix': transitions  # 轉換矩陣
    }
    
    # 計算各狀態統計量
    for state in unique_states:
        # 覆蓋率
        coverage = np.sum(labels == state) / len(labels)
        stats['coverage'][state] = coverage
        
        # 平均持續時間
        if state in state_durations and state_durations[state]:
            mean_duration = np.mean(state_durations[state])
            if fs is not None:
                mean_duration /= fs  # 轉換為秒
            stats['mean_duration'][state] = mean_duration
        else:
            stats['mean_duration'][state] = 0
        
        # 發生頻率
        if state in state_durations:
            occurrence = len(state_durations[state])
        else:
            occurrence = 0
        stats['occurrence'][state] = occurrence
    
    return stats
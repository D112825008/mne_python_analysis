"""
通用工作流程模組 - General Workflows Module

包含非 ASRT 專用的 EEG 分析工作流程函數

所有函數均從 main.py 的選項處理邏輯中抽取
"""

import os
from pathlib import Path
import mne

# mne_python_analysis 基礎模組
from mne_python_analysis.preprocessing import (
    preprocess_raw_data_interactive,
    interactive_marking_bad_segments
)
from mne_python_analysis.ica_analysis import perform_ica
from mne_python_analysis.microstate import prepare_for_microstate
from mne_python_analysis.epochs import (
    select_epoch_mode,
    epoch_data,
    epoch_data_interactive,
    epoch_data_asrt,
    create_asrt_epochs,
    compute_psd,
    compute_tfr
)
from mne_python_analysis.utils import (
    plot_raw_data,
    plot_electrodes,
    plot_psd
)
from mne_python_analysis.data_io import save_raw_data, save_epochs

# 同一資料夾內的模組
from .prompts import ask_save_confirmation


# ============================================================
# 檢視資料
# ============================================================

def display_raw_waveform(raw, subject_id):
    """
    顯示原始 EEG 波形
    
    來源：main.py process_eeg_data() 第 2044-2045 行（選項 1）
    抽取：波形顯示邏輯
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw 資料物件
    subject_id : str
        受試者 ID
    """
    plot_raw_data(raw, title=f'{subject_id} EEG', block=True)


def display_electrode_positions(raw):
    """
    顯示電極位置圖
    
    來源：main.py process_eeg_data() 第 2047-2048 行（選項 2）
    抽取：電極位置顯示邏輯
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw 資料物件
    """
    plot_electrodes(raw, show_names=True, block=True)


def display_psd_plot(raw):
    """
    顯示功率頻譜圖 (PSD)
    
    來源：main.py process_eeg_data() 第 2050-2051 行（選項 3）
    抽取：PSD 顯示邏輯
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw 資料物件
    """
    plot_psd(raw, fmax=50, block=True)


# ============================================================
# 前處理流程
# ============================================================

def run_standard_preprocessing(raw, subject_id):
    """
    執行標準前處理流程（互動式）
    
    來源：main.py process_eeg_data() 第 2061-2067 行（選項 5）
    抽取：標準前處理執行邏輯
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw 資料物件
    subject_id : str
        受試者 ID
    
    Returns
    -------
    processed_raw : mne.io.Raw
        處理後的 Raw 物件
    proc_info : dict
        處理資訊
    """
    try:
        processed_raw, proc_info = preprocess_raw_data_interactive(raw, subject_id)
        print("✓ 標準前處理流程完成")
        return processed_raw, proc_info
    except Exception as e:
        print(f"Preprocessing 時發生錯誤: {str(e)}")
        raise


def mark_bad_segments_interactive(raw):
    """
    互動式標記壞段落
    
    來源：main.py process_eeg_data() 第 2053-2058 行（選項 4）
    抽取：標記壞段落執行邏輯
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw 資料物件
    
    Returns
    -------
    marked_raw : mne.io.Raw
        標記後的 Raw 物件
    """
    try:
        marked_raw = interactive_marking_bad_segments(raw)
        print("✓ 互動式標記壞段落完成")
        return marked_raw
    except Exception as e:
        print(f"標記時發生錯誤: {str(e)}")
        raise


# ============================================================
# 進階分析
# ============================================================

def run_ica_analysis(raw):
    """
    執行 ICA 分析
    
    來源：main.py process_eeg_data() 第 2072-2080 行（選項 6）
    抽取：ICA 分析執行邏輯
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw 資料物件
    
    Returns
    -------
    processed_raw : mne.io.Raw or None
        ICA 處理後的 Raw 物件
    ica_info : dict or None
        ICA 資訊
    """
    try:
        result = perform_ica(raw)
        if result is not None:
            processed_raw, ica_info = result
            print("✓ ICA 處理完成")
            return processed_raw, ica_info
        else:
            return None, None
    except Exception as e:
        print(f"ICA 分析時發生錯誤: {str(e)}")
        raise


def prepare_microstate_analysis(raw, subject_id):
    """
    準備 Microstate 分析
    
    來源：main.py process_eeg_data() 第 2082-2089 行（選項 7）
    抽取：Microstate 準備執行邏輯
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw 資料物件
    subject_id : str
        受試者 ID
    
    Returns
    -------
    raw_ms : mne.io.Raw
        準備好的 Microstate Raw 物件
    """
    try:
        raw_ms = prepare_for_microstate(raw.copy())
        print("✓ Microstate 準備完成")
        plot_raw_data(raw_ms, title=f'{subject_id} - Microstate', block=True)
        return raw_ms
    except Exception as e:
        print(f"Microstate 處理時發生錯誤: {str(e)}")
        raise


# ============================================================
# Epochs 分析
# ============================================================

def create_epochs_interactive(raw, subject_id, behavior_df=None, trial_classification='trigger'):
    """
    互動式建立 Epochs

    Parameters
    ----------
    raw : mne.io.Raw
    subject_id : str
    behavior_df : pd.DataFrame or None
        從 CSV 載入的行為資料（ASRT triplet 分類用）
    trial_classification : str
        'trigger' 或 'triplet'（僅 ASRT 模式有效）

    Returns
    -------
    (epochs, mode_desc) tuple
    """
    if raw is None:
        print("⚠️  尚未載入資料，請先載入 EEG 檔案")
        return None, None

    try:
        mode = select_epoch_mode()

        if mode == 'fixed':
            epochs = epoch_data_interactive(raw, subject_id)
            mode_desc = "固定時間切割 (互動式參數)"

        elif mode == 'event':
            epochs = epoch_data(raw, subject_id)
            mode_desc = "事件鎖定切割 (預設參數)"

        elif mode == 'asrt':
            # ASRT 任務專用：支援 trigger 和 triplet 兩種分類
            epochs = create_asrt_epochs(
                raw, subject_id,
                behavior_df=behavior_df,
                trial_classification=trial_classification,
            )
            if trial_classification == 'triplet':
                mode_desc = "ASRT Triplet 頻率分類 (high / low)"
            else:
                mode_desc = "ASRT 任務專用 Epoch (Regular / Random)"

        else:
            print("⚠️ 未知的 epoch 模式，取消建立 Epochs")
            return None, None

        print(f"✓ Epochs 建立完成（{mode_desc}）")
        return epochs, mode_desc

    except Exception as e:
        print(f"建立 Epochs 時發生錯誤: {str(e)}")
        raise


def create_epochs_default(raw, subject_id):
    """
    使用預設參數建立 Epochs
    
    來源：main.py process_eeg_data() 第 2126-2132 行（選項 9）
    抽取：預設 Epochs 建立邏輯
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw 資料物件
    subject_id : str
        受試者 ID
    
    Returns
    -------
    epochs : mne.Epochs
        建立的 Epochs
    """
    try:
        epochs = epoch_data(raw, subject_id)
        print("✓ Epochs 建立完成")
        return epochs
    except Exception as e:
        print(f"建立 Epochs 時發生錯誤: {str(e)}")
        raise


def display_epochs_info(epochs, subject_id):
    """
    顯示 Epochs 資訊（簡單版本，在主選單中使用）
    
    來源：從 ui/menu.py display_epochs_info_detailed 簡化而來
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 物件
    subject_id : str
        受試者 ID
    """
    if epochs is None:
        print("⚠️  尚未建立 Epochs")
        return
    
    print(f"\nEpochs 資訊:")
    print(f"  受試者: {subject_id}")
    print(f"  Epochs 數量: {len(epochs)}")
    print(f"  通道數: {len(epochs.ch_names)}")
    print(f"  採樣率: {epochs.info['sfreq']} Hz")
    print(f"  時間範圍: {epochs.tmin} ~ {epochs.tmax} s")


def display_epochs_plot(epochs):
    """
    繪製 Epochs
    
    來源：main.py process_eeg_data() 第 2134-2138 行（選項 10）
    抽取：Epochs 繪圖邏輯
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 物件
    """
    if epochs is not None:
        epochs.plot(n_channels=30, n_epochs=10, block=True)
    else:
        print("⚠️  請先建立 Epochs")


def compute_epochs_psd(epochs):
    """
    計算 Epochs PSD
    
    來源：main.py process_eeg_data() 第 2140-2148 行（選項 11）
    抽取：PSD 計算邏輯
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 物件
    
    Returns
    -------
    psd : mne.time_frequency.Spectrum or None
        PSD 物件
    """
    if epochs is not None:
        try:
            psd = compute_psd(epochs)
            print("✓ PSD 計算完成")
            return psd
        except Exception as e:
            print(f"計算 PSD 時發生錯誤: {str(e)}")
            raise
    else:
        print("⚠️  請先建立 Epochs")
        return None


def compute_epochs_tfr(epochs):
    """
    計算 Epochs 時頻分析 (TFR)
    
    來源：main.py process_eeg_data() 第 2150-2158 行（選項 12）
    抽取：TFR 計算邏輯
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 物件
    
    Returns
    -------
    tfr : mne.time_frequency.AverageTFR or None
        TFR 物件
    """
    if epochs is not None:
        try:
            tfr = compute_tfr(epochs)
            print("✓ TFR 計算完成")
            return tfr
        except Exception as e:
            print(f"計算 TFR 時發生錯誤: {str(e)}")
            raise
    else:
        print("⚠️  請先建立 Epochs")
        return None


# ============================================================
# 儲存相關
# ============================================================

def save_raw_interactive(raw, subject_id):
    """
    互動式儲存 Raw 資料
    
    來源：main.py process_eeg_data() 第 2444-2449 行（選項 17）
    抽取：Raw 儲存邏輯
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw 物件
    subject_id : str
        受試者 ID
    """
    if raw is None:
        print("⚠️  沒有 Raw 資料可以儲存")
        return
    
    if ask_save_confirmation('raw'):
        try:
            save_raw_data(raw, subject_id)
            print("✓ Raw 資料已儲存")
        except Exception as e:
            print(f"儲存 Raw 資料時發生錯誤: {str(e)}")


def save_epochs_interactive(epochs, subject_id):
    """
    互動式儲存 Epochs
    
    來源：main.py process_eeg_data() 第 2451-2455 行（選項 18）
    抽取：Epochs 儲存邏輯
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 物件
    subject_id : str
        受試者 ID
    """
    if epochs is None:
        print("⚠️  沒有 Epochs 可以儲存")
        return
    
    if ask_save_confirmation('epochs'):
        try:
            save_epochs(epochs, subject_id)
            print("✓ Epochs 已儲存")
        except Exception as e:
            print(f"儲存 Epochs 時發生錯誤: {str(e)}")
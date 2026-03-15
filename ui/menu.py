import os
from pathlib import Path

"""
選單顯示模組 - Menu Display Module

包含所有選單顯示和資訊顯示函數

所有函數均從 main.py 的 print() 邏輯中抽取
"""

import os
from pathlib import Path


def show_data_source_menu():
    """
    顯示資料來源選擇選單
    
    來源：main.py select_data_source() 第 237-250 行
    抽取：選單顯示部分
    
    Returns
    -------
    choice : str
        使用者選擇的選項 ('0'-'7')
    """
    print("\n" + "="*60)
    print("EEG 資料分析系統")
    print("="*60)
    print("\n請選擇資料來源:")
    print("1. BIDS 格式資料夾")
    print("2. CNT 檔案 (Neuroscan)")
    print("3. CDT 檔案 (Curry)")
    print("4. SET 檔案 (EEGLAB)")
    print("5. FIF 檔案 (MNE)")
    print("6. CNT 檔案 (Neuroscan 64→32)")
    print("0. 退出程式")

    choice = input("\n請輸入選項 (0-6): ").strip()
    return choice


def show_main_menu(has_raw=True, has_epochs=False, asrt_available=True):
    """
    顯示主要操作選單
    
    來源：main.py process_eeg_data() 第 1999-2041 行
    抽取：主選單顯示邏輯
    
    Parameters
    ----------
    has_raw : bool
        是否已載入 Raw 資料
    has_epochs : bool
        是否已建立 Epochs
    asrt_available : bool
        ASRT 模組是否可用
    
    Returns
    -------
    choice : str
        使用者選擇的選項
    """
    print("\n請選擇要執行的操作:")
    print("\n 提示：建議先執行標準前處理流程，再進行 ASRT 分析\n")
    
    print("【檢視資料】")
    print("  1. 顯示原始 EEG 波形")
    print("  2. 顯示電極位置圖")
    print("  3. 顯示功率頻譜圖 (PSD)")
    print("  4. 互動式標記壞段落")
    
    print("\n【標準前處理流程】")
    print("  5. 執行標準前處理流程 (互動式參數)")
    
    print("\n【進階分析】")
    print(" 6. 執行 ICA 分析")
    print(" 7. 準備 Microstate 分析")
    
    print("\n【Epochs 分析】")
    print(" 8. 建立 Epochs (互動式參數)")
    print(" 10. 顯示 Epochs")
    print(" 11. 計算 PSD (Epochs)")
    print(" 12. 計算時頻分析 (TFR)")
    
    if asrt_available:
        print("\n【ASRT 實驗分析】")
        print(" 13. ASRT 完整分析 (Stimulus + Response Lock)")
        print("     → Stimulus Lock: Alpha (200-400ms)")
        print("     → Response Lock: Theta (-300~+50ms)")
        print("     → 排除: HEOG/VEOG/A1/A2")
        print(" 14. ASRT ROI 頻譜分析 (需先建立 Epochs)")
        print(" 15. ASRT Block 比較分析 (需先建立 Epochs)")
        print(" 16. ASRT 群體分析 (Group-level ERSP)")
        print(" 20. 輔助：把 RT 加入當前 Epochs (精確對齊)")
        print("     → 根據 block 和 trial 從行為資料精確提取 RT")
        print("     → 100% 準確匹配")
        print(" 21. Response ERSP (Stimulus 對齊 + 整段平均 baseline)")
        print("     → Time domain: Stimulus → Response 對齊")
        print("     → Frequency domain: 整段 epoch 平均 baseline")

    
    print("\n【儲存與退出】")
    print(" 17. 存檔 (Raw)")
    print(" 18. 存檔 (Epochs)")
    print(" 19. 顯示處理歷程")
    print("  0. 返回主選單")
    
    choice = input("\n請輸入選項 (0-21): ").strip()
    return choice


def show_epochs_analysis_menu():
    """
    顯示 Epochs 分析子選單（當已載入 Epochs 檔案時）
    
    來源：main.py process_single_file() 第 335-343 行
    抽取：Epochs 子選單顯示邏輯
    
    Returns
    -------
    choice : str
        使用者選擇的選項
    """
    print("\n由於已載入 Epochs，可直接進行以下分析:")
    print("1. ASRT ROI 頻譜分析")
    print("2. ASRT Block 比較分析")
    print("3. 顯示 Epochs 資訊")
    print("4. EEG 視覺化（PSD/TFR/Topomap）")
    print("5. 全腦 FFT 功率分析")
    print("6. ERSP 分析（Lum et al. 2023 風格）")
    print("7. 極端值排除")
    print("0. 返回主選單")
    
    choice = input("\n請選擇: ").strip()
    return choice


def display_welcome_message(asrt_available=True):
    """
    顯示程式啟動歡迎訊息
    
    來源：main.py main() 第 2470-2486 行
    抽取：歡迎訊息顯示邏輯
    
    Parameters
    ----------
    asrt_available : bool
        ASRT 模組是否可用
    """
    print("\n" + "="*60)
    print("  EEG 資料分析系統")
    print("  MNE-Python Analysis Package v4.1")
    print("="*60)
    print("\n撰寫者: HE-JUN, CHEN (Dillian417)")
    print("指導教授: Prof. Erik Chung")
    print("實驗室: Action & Cognition Laboratory, NCU")
    print("="*60)
    
    if asrt_available:
        print("\n✓ ASRT 分析模組已載入")
    else:
        print("\n⚠️  ASRT 分析模組未完全載入")
    
    print("="*60)


def display_subject_info(subject_id, raw=None, epochs=None):
    """
    顯示當前受試者資訊
    
    來源：main.py process_eeg_data() 第 1992-1998 行
    抽取：受試者資訊顯示邏輯
    
    Parameters
    ----------
    subject_id : str
        受試者 ID
    raw : mne.io.Raw or None
        Raw 資料物件
    epochs : mne.Epochs or None
        Epochs 資料物件
    """
    print(f"\n{'='*60}")
    print(f"受試者: {subject_id}")
    print(f"{'='*60}")
    
    if raw is not None:
        print(f"Raw 資料: ✓")
        print(f"  通道數: {len(raw.ch_names)}")
        print(f"  採樣率: {raw.info['sfreq']} Hz")
    
    if epochs is not None:
        print(f"Epochs 資料: ✓")
        print(f"  Epochs 數量: {len(epochs)}")


def display_processing_history(history_list):
    """
    顯示處理歷程
    
    來源：main.py process_eeg_data() 第 2456-2459 行（選項 19）
    抽取：處理歷程顯示邏輯
    
    Parameters
    ----------
    history_list : list
        處理歷程列表
    """
    print("\n" + "="*60)
    print("處理歷程")
    print("="*60)
    if history_list:
        for i, item in enumerate(history_list, 1):
            print(f"{i}. {item}")
    else:
        print("尚未執行任何處理步驟")


def display_epochs_info_detailed(epochs, subject_id):
    """
    顯示詳細的 Epochs 資訊
    
    來源：main.py process_single_file() 第 368-377 行（選項 3）
    抽取：詳細資訊顯示邏輯
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 資料物件
    subject_id : str
        受試者 ID
    """
    print(f"\nEpochs 資訊:")
    print(f"  受試者: {subject_id}")
    print(f"  Epochs 數量: {len(epochs)}")
    print(f"  通道數: {len(epochs.ch_names)}")
    print(f"  通道名稱: {epochs.ch_names}")
    print(f"  採樣率: {epochs.info['sfreq']} Hz")
    print(f"  時間範圍: {epochs.tmin} ~ {epochs.tmax} s")
    
    if hasattr(epochs, 'metadata') and epochs.metadata is not None:
        print(f"  Metadata 欄位: {list(epochs.metadata.columns)}")
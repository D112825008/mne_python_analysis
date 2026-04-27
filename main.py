"""
EEG 主要處理程式 (Using MNE-Python version 1.8+)
                                                        
撰寫者: HE-JUN,CHEN (Dillian417) 2025
指導教授: Prof. Erik Chung from Action & Cognition Laboratory, National Central University

版本: 4.0 - 新增 ASRT 實驗分析模組
修改內容：
  - 新增 ASRT Stimulus-locked 分析
  - 新增 ASRT Response-locked 分析
  - 新增 ASRT ROI 頻譜分析
  - 新增 ASRT 統計比較功能
  - 整合降採樣優化建議
"""

import os
import re
import glob as glob_module
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import mne
import numpy as np
import pandas as pd

# 確保可以導入 mne_python_analysis 套件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 基礎模組導入
# ============================================================
from mne_python_analysis.data_io import (
    load_bids_eeg, load_eeg_file, save_raw_data, save_epochs,
    select_file_interactively
)
from mne_python_analysis.montage import setup_bids_montage, setup_standard_montage
from mne_python_analysis.utils import (
    select_subject,
    set_matplotlib_properties
)

# ============================================================
# UI 模組 (新增)
# ============================================================
from mne_python_analysis.ui.menu import (
    show_data_source_menu,
    show_main_menu,
    show_epochs_analysis_menu,
    display_welcome_message,
    display_subject_info,
    display_processing_history,
    display_epochs_info_detailed
)
from mne_python_analysis.ui.prompts import ask_file_path
from mne_python_analysis.ui.workflows import (
    display_raw_waveform,
    display_electrode_positions,
    display_psd_plot,
    run_standard_preprocessing,
    mark_bad_segments_interactive,
    run_ica_analysis,
    prepare_microstate_analysis,
    create_epochs_interactive,
    create_epochs_default,
    display_epochs_plot,
    compute_epochs_psd,
    compute_epochs_tfr,
    save_raw_interactive,
    save_epochs_interactive
)

# ============================================================
# ASRT 模組 (新增)
# ============================================================
try:
    from mne_python_analysis.asrt.workflows import (
        asrt_complete_analysis,
        asrt_roi_spectral_analysis,
        asrt_block_comparison,
        asrt_artifact_rejection
    )
    from mne_python_analysis.asrt import (
        asrt_visualization,
        asrt_wholebrain_fft_analysis,
        asrt_testing_phase_topomap,
        asrt_ersp_full_analysis
    )
    ASRT_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ASRT 分析模組未完全安裝: {e}")
    ASRT_MODULES_AVAILABLE = False

# 新增：導入 ASRT workflows
try:
    from mne_python_analysis.asrt.workflows import asrt_stimulus_to_response_full_baseline
    ASRT_WORKFLOWS_AVAILABLE = True
except ImportError:
    ASRT_WORKFLOWS_AVAILABLE = False
    
from extract_rt_precise import add_rt_to_epochs_from_behavioral
from asrt_response_ersp_from_epochs import response_ersp_from_current_epochs


# ============================================================
# 時長計算函數 (保留)
# 來源：原 main.py 94-234 行
# ============================================================

def calculate_experiment_duration(raw, manual_sfreq=None):
    """
    計算實驗總時長
    
    參數:
        raw (mne.io.Raw): Raw 資料物件
        manual_sfreq (float, optional): 手動輸入的採樣率 (Hz)
    
    返回:
        dict: 包含時長資訊的字典
    """
    # 獲取採樣率
    if manual_sfreq is not None:
        sfreq = manual_sfreq
        print(f"\n使用手動輸入的採樣率: {sfreq} Hz")
    else:
        sfreq = raw.info['sfreq']
        print(f"\n使用檔案中的採樣率: {sfreq} Hz")
    
    # 獲取總樣本數
    n_samples = raw.n_times
    
    # 計算總時長（秒）
    duration_seconds = n_samples / sfreq
    
    # 轉換成不同單位
    duration_minutes = duration_seconds / 60
    duration_hours = duration_minutes / 60
    
    # 格式化時間顯示（時:分:秒）
    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = duration_seconds % 60
    
    # 建立結果字典
    result = {
        'total_samples': n_samples,
        'sampling_rate': sfreq,
        'duration_seconds': duration_seconds,
        'duration_minutes': duration_minutes,
        'duration_hours': duration_hours,
        'formatted_time': f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    }
    
    return result


def display_duration_info(duration_info):
    """
    顯示實驗時長資訊
    
    參數:
        duration_info (dict): calculate_experiment_duration 返回的字典
    """
    print("\n" + "=" * 60)
    print("實驗時長資訊 Experiment Duration Information")
    print("=" * 60)
    print(f"\n採樣率 Sampling Rate:        {duration_info['sampling_rate']:.2f} Hz")
    print(f"總樣本數 Total Samples:        {duration_info['total_samples']:,}")
    print(f"\n實驗總時長 Total Duration:")
    print(f"  ├─ {duration_info['duration_seconds']:.2f} 秒 (seconds)")
    print(f"  ├─ {duration_info['duration_minutes']:.2f} 分鐘 (minutes)")
    print(f"  ├─ {duration_info['duration_hours']:.4f} 小時 (hours)")
    print(f"  └─ 格式化時間: {duration_info['formatted_time']} (HH:MM:SS)")
    print("=" * 60)


def ask_calculate_duration(raw):
    """
    詢問使用者是否要計算實驗時長
    
    參數:
        raw (mne.io.Raw): Raw 資料物件
    
    返回:
        dict or None: 時長資訊字典，若使用者選擇跳過則返回 None
    """
    print("\n" + "=" * 60)
    print("實驗時長計算 Experiment Duration Calculator")
    print("=" * 60)
    print("\n是否要計算實驗總時長？")
    print("Do you want to calculate the total experiment duration?")
    print("\n選項 Options:")
    print("  1. 是，使用檔案中的採樣率")
    print("     Yes, use the sampling rate from the file")
    print("  2. 是，手動輸入採樣率")
    print("     Yes, manually input the sampling rate")
    print("  3. 否，跳過")
    print("     No, skip")
    
    choice = input("\n請選擇 (1/2/3) [預設 3]: ").strip() or '3'
    
    if choice == '1':
        # 使用檔案中的採樣率
        duration_info = calculate_experiment_duration(raw)
        display_duration_info(duration_info)
        return duration_info
        
    elif choice == '2':
        # 手動輸入採樣率
        while True:
            try:
                manual_sfreq_input = input("\n請輸入採樣率 (Hz) [例如: 512, 1000, 2048]: ").strip()
                
                if manual_sfreq_input == '':
                    print("未輸入採樣率，取消計算")
                    return None
                
                manual_sfreq = float(manual_sfreq_input)
                
                if manual_sfreq <= 0:
                    print("⚠️  採樣率必須大於 0，請重新輸入")
                    continue
                
                if manual_sfreq < 100 or manual_sfreq > 10000:
                    print(f"⚠️  警告：採樣率 {manual_sfreq} Hz 似乎不在常見範圍 (100-10000 Hz)")
                    confirm = input("是否繼續？(y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                # 計算時長
                duration_info = calculate_experiment_duration(raw, manual_sfreq=manual_sfreq)
                display_duration_info(duration_info)
                
                # 顯示與檔案採樣率的差異
                file_sfreq = raw.info['sfreq']
                if abs(manual_sfreq - file_sfreq) > 0.01:
                    file_duration = raw.n_times / file_sfreq
                    print(f"\n⚠️  注意：手動輸入的採樣率 ({manual_sfreq} Hz) 與檔案中的採樣率 ({file_sfreq} Hz) 不同")
                    print(f"    時長差異: {abs(duration_info['duration_seconds'] - file_duration):.2f} 秒")
                
                return duration_info
                
            except ValueError:
                print("⚠️  輸入格式錯誤，請輸入有效的數字")
                
    else:
        # 跳過
        print("跳過實驗時長計算")
        return None



# ============================================================
# 資料來源選擇 (簡化版)
# 來源：原 main.py 235-278 行
# 修改：使用 ui.menu.show_data_source_menu() 和 ui.prompts.ask_file_path()
# ============================================================

def select_data_source():
    """選擇資料來源（簡化版）"""
    choice = show_data_source_menu()
    
    format_map = {
        '1': 'BIDS',
        '2': 'CNT',
        '3': 'CDT',
        '4': 'SET',
        '5': 'FIF',
        '6': 'CNT_64to32'
    }

    if choice == '0':
        return None, None
    elif choice in format_map:
        file_format = format_map[choice]
        path = ask_file_path(file_format)
        return path, file_format
    else:
        print("無效的選項")
        return None, None


# ============================================================
# CSV 行為資料自動搜尋
# ============================================================

def _extract_subject_keyword(filepath):
    """從檔案路徑提取受試者關鍵字（例如 'sub0002'）"""
    filename = os.path.basename(filepath)
    match = re.search(r'(sub\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    parts = os.path.splitext(filename)[0].split('_')
    return parts[0] if parts else None


def _load_behavior_csv(csv_path):
    """載入完整行為 CSV"""
    NEEDED_COLS = [
        'learning_loop.thisRepN',
        'learning_loop.thisTrialN',
        'learning_trials.thisRepN',
        'learning_trials.thisTrialN',
        'combined_testing_trials.thisTrialN',
        'motor_percept_testing_loop.thisTrialN',
        'correct_answer_index',
        'key_resp.corr',
        'key_resp.rt',
        'learning_seq_files',
        'test_seq_files',
        'learning_type',
    ]
    try:
        df = pd.read_csv(csv_path)
        missing = [c for c in NEEDED_COLS if c not in df.columns]
        if missing:
            print(f"  ⚠  CSV 缺少欄位（略過）: {missing}")
        print(f"✓ CSV 已載入：{len(df)} 筆資料，{len(df.columns)} 個欄位")
        return df
    except Exception as e:
        print(f"⚠  載入 CSV 失敗: {e}")
        return None


def _auto_search_csv(raw_filepath):
    """
    在 raw 檔案所在目錄自動搜尋對應的 CSV 行為資料。

    Returns
    -------
    pd.DataFrame or None
    """
    subject_keyword = _extract_subject_keyword(raw_filepath)
    if subject_keyword is None:
        print("⚠  無法從檔案名提取受試者關鍵字，略過 CSV 搜尋")
        return None

    csv_dir = os.path.dirname(os.path.abspath(raw_filepath))
    pattern = os.path.join(csv_dir, f'*{subject_keyword}*.csv')
    found_csvs = glob_module.glob(pattern)

    if len(found_csvs) == 0:
        print(f"⚠  找不到對應的 CSV 檔案（關鍵字: {subject_keyword}），triplet 分類功能將無法使用")
        return None

    if len(found_csvs) == 1:
        print(f"✓ 找到 CSV：{found_csvs[0]}")
        confirm = input("確認是否載入？(y/n) [y]: ").strip().lower() or 'y'
        if confirm == 'y':
            return _load_behavior_csv(found_csvs[0])
        return None

    # 找到多個
    print("找到多個 CSV 檔案，請選擇：")
    for i, f in enumerate(found_csvs):
        print(f"  {i+1}. {os.path.basename(f)}")
    while True:
        choice_str = input("請輸入編號：").strip()
        try:
            idx = int(choice_str) - 1
            if 0 <= idx < len(found_csvs):
                return _load_behavior_csv(found_csvs[idx])
            print(f"⚠  請輸入 1 到 {len(found_csvs)} 之間的數字")
        except ValueError:
            print("⚠  請輸入有效的數字")


# ============================================================
# 檔案處理 (部分簡化)
# 來源：原 main.py 279-449 行
# 修改：使用 ui.menu.show_epochs_analysis_menu() 和 display_epochs_info_detailed()
# ============================================================

def process_single_file(file_path, file_format):
    """處理單一檔案"""
    print(f"\n載入 {file_format} 檔案: {file_path}")

    # 若是 Neuroscan 64→32 模式，呼叫 data.io 的專用轉換函式
    if file_format == "CNT_64to32":
        from mne_python_analysis.data_io import load_neuroscan_64_to_32
        raw, events = load_neuroscan_64_to_32(file_path)
        data_obj = raw
        actual_format = 'CNT'
    else:
        data_obj, events, actual_format = load_eeg_file(file_path)

    if data_obj is None:
        print("檔案載入失敗")
        return

    subject_id = Path(file_path).stem
    # 實驗總時長的功能
    # 只有 Raw 資料才能計算總實驗時長
    if actual_format != 'FIF_EPOCHS' and hasattr(data_obj, 'n_times'):
        # 詢問是否要計算實驗時長
        duration_info = ask_calculate_duration(data_obj)
        
        # 如果使用者計算了時長，可以選擇儲存資訊
        if duration_info is not None:
            # 可選：詢問是否要儲存到文字檔
            save_choice = input("\n是否要將時長資訊儲存到文字檔？(y/n) [預設 n]: ").strip().lower()
            if save_choice == 'y':
                output_dir = Path(file_path).parent
                output_file = output_dir / f"{subject_id}_duration_info.txt"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 60 + "\n")
                    f.write("實驗時長資訊 Experiment Duration Information\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"\n受試者 Subject ID:           {subject_id}\n")
                    f.write(f"檔案路徑 File Path:          {file_path}\n")
                    f.write(f"\n採樣率 Sampling Rate:        {duration_info['sampling_rate']:.2f} Hz\n")
                    f.write(f"總樣本數 Total Samples:        {duration_info['total_samples']:,}\n")
                    f.write(f"\n實驗總時長 Total Duration:\n")
                    f.write(f"  ├─ {duration_info['duration_seconds']:.2f} 秒 (seconds)\n")
                    f.write(f"  ├─ {duration_info['duration_minutes']:.2f} 分鐘 (minutes)\n")
                    f.write(f"  ├─ {duration_info['duration_hours']:.4f} 小時 (hours)\n")
                    f.write(f"  └─ 格式化時間: {duration_info['formatted_time']} (HH:MM:SS)\n")
                    f.write("=" * 60 + "\n")
                
                print(f"\n✓ 時長資訊已儲存至: {output_file}")

    # 判斷載入的是 raw 還是 epochs（移到外層，不受時長計算的 if 影響）
    if actual_format == 'FIF_EPOCHS':
        # === 顯示 Epochs 資訊後進入分析選單 ===
        print("\n" + "="*60)
        print("已載入 Epochs 檔案")
        print("="*60)
        print(f"受試者: {subject_id}")
        print(f"Epochs 數量: {len(data_obj)}")
        print(f"通道數: {len(data_obj.ch_names)}")

        # 判斷 Epochs 類型（根據時間範圍）
        if hasattr(data_obj, 'tmin') and hasattr(data_obj, 'tmax'):
            if data_obj.tmin >= -0.9 and data_obj.tmax <= 1.1:
                epochs_type = "Stimulus-locked"
            elif data_obj.tmin >= -1.2 and data_obj.tmax <= 0.6:
                epochs_type = "Response-locked"
            else:
                epochs_type = "未知"
            print(f"  類型: {epochs_type}")
            print(f"  時間範圍: {data_obj.tmin:.3f} ~ {data_obj.tmax:.3f} s")

        if hasattr(data_obj, 'baseline') and data_obj.baseline is not None:
            print(f"  Baseline: {data_obj.baseline}")

        if hasattr(data_obj, 'metadata') and data_obj.metadata is not None:
            print(f"  Metadata 欄位: {list(data_obj.metadata.columns)}")
            if 'trial_type' in data_obj.metadata.columns:
                print(f"\n  Trial 統計:")
                for trial_type, count in data_obj.metadata['trial_type'].value_counts().items():
                    print(f"    {trial_type}: {count} trials")
            if 'block' in data_obj.metadata.columns:
                blocks = data_obj.metadata['block'].unique()
                print(f"  Block 範圍: {blocks.min()} ~ {blocks.max()}")

        print("\n" + "="*60)
        print("⚠️  注意：由於載入的是 Epochs 檔案（非 Raw），")
        print("    前處理、ICA 等功能無法使用")
        print("="*60)

        # 直接進入分析選單（pre-load epochs）
        process_eeg_data(subject_id, {'raw': None, 'events': events, 'epochs': data_obj}, file_path)

    else:
        # === 如果是 raw，進入正常流程 ===
        behavior_df = _auto_search_csv(file_path)
        process_eeg_data(subject_id, {'raw': data_obj, 'events': events}, file_path,
                         behavior_df=behavior_df)



# ============================================================
# BIDS 資料處理 (保留)
# 來源：原 main.py 450-477 行
# ============================================================

def process_bids_data(data_path):
    """處理 BIDS 格式資料"""
    print(f"\n載入 BIDS 資料: {data_path}")
    
    try:
        all_data = load_bids_eeg(data_path)
        print(f"\n成功載入 {len(all_data)} 位受試者的資料")
        
        while True:
            selected_subject = select_subject(all_data)
            if selected_subject is None:
                break
            
            process_eeg_data(
                selected_subject,
                all_data[selected_subject],
                data_path
            )
    
    except Exception as e:
        print(f"載入 BIDS 資料時發生錯誤: {str(e)}")




# ============================================================
# Raw 資料互動載入輔助函數
# ============================================================

def _load_raw_interactively():
    """
    當需要 Raw 資料但尚未載入時，提示使用者輸入路徑並載入。
    返回: mne.io.Raw 物件，或 None（使用者取消/載入失敗）
    """
    print("\n" + "="*60)
    print("⚠️  此功能需要 Raw 資料，目前尚未載入")
    print("="*60)
    print("  支援格式: .fif (Raw)、.cnt (Neuroscan)、.cdt (Curry)")
    print("  （留空直接 Enter 取消）")
    raw_path = input("\n  請輸入 Raw 檔案路徑: ").strip()

    if not raw_path:
        print("取消，返回選單")
        return None

    if not os.path.exists(raw_path):
        print(f"⚠️  路徑不存在: {raw_path}")
        return None

    try:
        loaded_obj, _, loaded_format = load_eeg_file(raw_path)

        if loaded_obj is None:
            print("⚠️  載入失敗")
            return None

        if loaded_format == 'FIF_EPOCHS':
            print("⚠️  偵測到 Epochs 檔案，請提供 Raw 格式的 .fif（檔名不含 _epo）")
            return None

        # 設定標準電極位置
        print("\n設定標準電極位置...")
        try:
            loaded_obj = setup_standard_montage(loaded_obj, 'biosemi64')
        except Exception as e:
            print(f"設定 montage 時發生錯誤: {str(e)}")

        print(f"\n✓ Raw 資料已載入: {Path(raw_path).name}")
        print(f"  通道數: {len(loaded_obj.ch_names)}, 採樣率: {loaded_obj.info['sfreq']} Hz")
        return loaded_obj

    except Exception as e:
        print(f"載入失敗: {str(e)}")
        return None


# ============================================================
# EEG 資料處理主函數 (完全重構)
# 來源：原 main.py 1969-2466 行 (498 行)
# 重構：使用 ui.menu, ui.workflows, asrt.workflows 模組
# ============================================================

def process_eeg_data(subject_id, subject_data, data_path=None, behavior_df=None):
    """處理選定的受試者 EEG 資料（重構版）"""
    
    # 初始化
    raw = subject_data.get('raw')

    # Montage 設定（僅在有 Raw 資料時）
    if raw is not None:
        if data_path and Path(data_path).is_dir():
            channels_tsv = os.path.join(data_path, f"{subject_id}_task-restingstate_channels.tsv")
            coordsystem_json = os.path.join(data_path, f"{subject_id}_task-restingstate_coordsystem.json")

            if os.path.exists(channels_tsv) and os.path.exists(coordsystem_json):
                raw = setup_bids_montage(raw, channels_tsv, coordsystem_json)
        else:
            print("\n設定標準電極位置...")
            try:
                raw = setup_standard_montage(raw, 'biosemi64')
            except Exception as e:
                print(f"設定 montage 時發生錯誤: {str(e)}")

    current_raw = raw
    current_epochs = subject_data.get('epochs')
    processing_history = []
    asrt_results = {}
    
    # 主迴圈
    while True:
        # 顯示受試者資訊
        display_subject_info(subject_id, current_raw, current_epochs)
        
        # 顯示選單並取得選項
        choice = show_main_menu(
            has_raw=(current_raw is not None),
            has_epochs=(current_epochs is not None),
            asrt_available=ASRT_MODULES_AVAILABLE
        )

        # === 需要 Raw 資料的選項，若未載入則提示使用者輸入路徑 ===
        _RAW_REQUIRED = {'1', '2', '3', '4', '5', '6', '7', '21'}
        if choice in _RAW_REQUIRED and current_raw is None:
            current_raw = _load_raw_interactively()
            if current_raw is None:
                continue

        # === 檢視資料 (選項 1-4) ===
        if choice == '1':
            display_raw_waveform(current_raw, subject_id)
        
        elif choice == '2':
            display_electrode_positions(current_raw)
        
        elif choice == '3':
            display_psd_plot(current_raw)
        
        elif choice == '4':
            try:
                current_raw = mark_bad_segments_interactive(current_raw)
                processing_history.append("互動式標記壞段落")
            except Exception as e:
                print(f"標記時發生錯誤: {str(e)}")
        
        # === 標準前處理 (選項 5) ===
        elif choice == '5':
            try:
                result = run_standard_preprocessing(current_raw, subject_id)
                # ========== 修正：檢查返回值 ==========
                if result is not None and result[0] is not None:
                    current_raw, proc_info = result
                    processing_history.append("標準前處理流程")
                    print("✓ 標準前處理流程完成")
                else:
                    print("⚠️  前處理流程已取消，資料未修改")
                # =======================================
            except Exception as e:
                print(f"前處理時發生錯誤: {str(e)}")
        
        # === 進階分析 (選項 6-7) ===
        elif choice == '6':
            try:
                result = run_ica_analysis(current_raw)
                if result is not None and result[0] is not None:
                    current_raw, ica_info = result
                    processing_history.append("ICA")
            except Exception as e:
                print(f"ICA 分析時發生錯誤: {str(e)}")
        
        elif choice == '11':
            try:
                raw_ms = prepare_microstate_analysis(current_raw, subject_id)
                processing_history.append("Microstate 準備")
            except Exception as e:
                print(f"Microstate 處理時發生錯誤: {str(e)}")

        # === Epochs 分析 (選項 7-10) ===
        elif choice == '7':
            try:
                # --- Trial 分類方式選擇 ---
                print("\nTrial 分類方式:")
                print("  1. Regular trial / Random trial（根據 trigger code）")
                print("  2. High-frequency / Low-frequency triplet（需要 CSV 行為資料）")

                if behavior_df is None:
                    print("  ⚠  尚未載入 CSV，只能選擇選項 1")
                    trial_classification = 'trigger'
                else:
                    cls_choice = input("請選擇 (1/2) [1]: ").strip() or '1'
                    trial_classification = 'trigger' if cls_choice == '1' else 'triplet'

                epochs_result = create_epochs_interactive(
                    current_raw, subject_id,
                    behavior_df=behavior_df,
                    trial_classification=trial_classification,
                )
                if epochs_result[0] is not None:
                    result, mode_desc = epochs_result
                    processing_history.append(f"建立 Epochs - {mode_desc}")
                    
                    # === 判斷返回值類型 ===
                    if isinstance(result, dict) and 'stimulus' in result and 'response' in result:
                        # ============================================================
                        # 雙 Epochs 模式 - 支援同步極端值排除
                        # ============================================================
                        print("\n" + "="*60)
                        print("雙 Epochs 模式（同步極端值排除）")
                        print("="*60)
                        
                        epochs_stim = result['stimulus']
                        epochs_resp = result['response']
                        resp_to_stim_map = result['resp_to_stim_map']
                        phase_tag = result['phase_tag']
                        subject_id_from_epochs = result['subject_id']
                        
                        print(f"✓ 已建立 Stimulus-locked epochs: {len(epochs_stim)} trials")
                        print(f"✓ 已建立 Response-locked epochs: {len(epochs_resp)} trials")
                        print(f"✓ Trial 對應關係: {len(resp_to_stim_map)} 對")
                        
                        # === 詢問是否做極端值排除 ===
                        print("\n" + "="*60)
                        print("極端值排除（Artifact Rejection）")
                        print("="*60)
                        print("提示：")
                        print("  - 會先對 Response epochs 做極端值排除")
                        print("  - 被排除的 trials 會自動在 Stimulus epochs 中也被排除")
                        print("  - 這樣可以確保兩個 epochs 對應相同的 trials")
                        
                        do_reject = input("\n是否要進行極端值排除？(y/n) [預設 y]: ").strip().lower()
                        
                        if do_reject != 'n':
                            print("\n選擇極端值排除方法:")
                            print("1. Flexible (彈性閾值 + Visual Inspection)")
                            print("2. Fixed (固定閾值 + Visual Inspection)")
                            print("3. Autoreject (自動排除)")
                            
                            method_choice = input("\n請選擇 (1/2/3) [預設 1]: ").strip() or '1'
                            method_map = {'1': 'flexible', '2': 'fixed', '3': 'autoreject'}
                            method = method_map.get(method_choice, 'flexible')
                            
                            try:
                                print("\n" + "="*60)
                                print("步驟 1/3: 對 Response epochs 做極端值排除")
                                print("="*60)
                                
                                # 對 Response epochs 做極端值排除
                                epochs_resp_clean, log = asrt_artifact_rejection(epochs_resp, method=method)
                                
                                if epochs_resp_clean is not None:
                                    # 找出被保留的 Response trials（用 events 比較）
                                    original_resp_samples = epochs_resp.events[:, 0]
                                    clean_resp_samples = epochs_resp_clean.events[:, 0]
                                    
                                    kept_resp_indices = []
                                    for i, sample in enumerate(original_resp_samples):
                                        if sample in clean_resp_samples:
                                            kept_resp_indices.append(i)
                                    
                                    print(f"\n✓ Response epochs 排除完成")
                                    print(f"  原始: {len(epochs_resp)} trials")
                                    print(f"  保留: {len(epochs_resp_clean)} trials")
                                    print(f"  排除: {len(epochs_resp) - len(epochs_resp_clean)} trials")
                                    print(f"  保留率: {log['retention_rate']:.1f}%")
                                    
                                    # === 步驟 2: 同步排除 Stimulus epochs ===
                                    print("\n" + "="*60)
                                    print("步驟 2/3: 同步排除 Stimulus epochs")
                                    print("="*60)
                                    
                                    # 找出對應的 Stimulus trial 索引
                                    kept_stim_indices = []
                                    for resp_idx in kept_resp_indices:
                                        if resp_idx in resp_to_stim_map:
                                            stim_idx = resp_to_stim_map[resp_idx]
                                            kept_stim_indices.append(stim_idx)
                                    
                                    kept_stim_indices = sorted(set(kept_stim_indices))
                                    
                                    print(f"  對應到 {len(kept_stim_indices)} 個 Stimulus trials")
                                    
                                    # 排除 Stimulus epochs
                                    epochs_stim_clean = epochs_stim[kept_stim_indices]
                                    
                                    print(f"\n✓ Stimulus epochs 同步排除完成")
                                    print(f"  原始: {len(epochs_stim)} trials")
                                    print(f"  保留: {len(epochs_stim_clean)} trials")
                                    print(f"  排除: {len(epochs_stim) - len(epochs_stim_clean)} trials")
                                    
                                    # === 步驟 3: 儲存兩個同步的 epochs ===
                                    print("\n" + "="*60)
                                    print("步驟 3/3: 儲存同步的 epochs")
                                    print("="*60)
                                    
#                                    import os
                                    
                                    # 儲存 Stimulus epochs
                                    default_stim_fname = f"{subject_id_from_epochs}_ASRT_stim_{phase_tag}-epo.fif"
                                    stim_fname = input(f"\n請輸入 Stimulus epochs 檔名 [預設: {default_stim_fname}]: ").strip() or default_stim_fname
                                    if stim_fname.endswith('.fif.gz'):
                                        pass
                                    elif stim_fname.endswith('.fif'):
                                        if not ('-epo.fif' in stim_fname or '_epo.fif' in stim_fname):
                                            stim_fname = stim_fname[:-4] + '-epo.fif'
                                    else:
                                        stim_fname += '-epo.fif'
                                    stim_path = os.path.join(os.getcwd(), stim_fname)
                                    epochs_stim_clean.save(stim_path, overwrite=True)
                                    print(f"✓ 已儲存 Stimulus epochs: {stim_fname}")

                                    # 儲存 Response epochs
                                    default_resp_fname = f"{subject_id_from_epochs}_ASRT_resp_{phase_tag}-epo.fif"
                                    resp_fname = input(f"\n請輸入 Response epochs 檔名 [預設: {default_resp_fname}]: ").strip() or default_resp_fname
                                    if resp_fname.endswith('.fif.gz'):
                                        pass
                                    elif resp_fname.endswith('.fif'):
                                        if not ('-epo.fif' in resp_fname or '_epo.fif' in resp_fname):
                                            resp_fname = resp_fname[:-4] + '-epo.fif'
                                    else:
                                        resp_fname += '-epo.fif'
                                    resp_path = os.path.join(os.getcwd(), resp_fname)
                                    epochs_resp_clean.save(resp_path, overwrite=True)
                                    print(f"✓ 已儲存 Response epochs: {resp_fname}")
                                    
                                    print("\n" + "="*60)
                                    print("✓ 雙 Epochs 建立完成（已同步極端值排除）")
                                    print("="*60)
                                    print(f"兩個 epochs 都保留 {len(epochs_resp_clean)} 個對應的 trials")
                                    print("可以用於後續分析：")
                                    print(f"  - {stim_fname} → 選項 15 (Stimulus ERSP)")
                                    print(f"  - {resp_fname} → 選項 16-17 (Response ERSP)")
                                    print(f"  → 完成選項 15+16 後，可用選項 18 進行 EEG-行為整合分析")
                                    
                                    # 設定 current_epochs
                                    current_epochs = epochs_resp_clean
                                    processing_history.append(f"極端值排除 ({method}) - 已同步")
                                    
                                else:
                                    print("\n⚠️  極端值排除失敗")
                                    
                            except Exception as e:
                                print(f"\n極端值排除時發生錯誤: {str(e)}")
                                import traceback
                                traceback.print_exc()
                        
                        else:
                            # 不做極端值排除，直接儲存
                            print("\n跳過極端值排除，儲存原始 epochs...")
                            
#                            import os
                            
                            # 儲存 Stimulus epochs
                            default_stim_fname = f"{subject_id_from_epochs}_ASRT_stim_{phase_tag}-epo.fif"
                            stim_fname = input(f"\n請輸入 Stimulus epochs 檔名 [預設: {default_stim_fname}]: ").strip() or default_stim_fname
                            if stim_fname.endswith('.fif.gz'):
                                pass
                            elif stim_fname.endswith('.fif'):
                                if not ('-epo.fif' in stim_fname or '_epo.fif' in stim_fname):
                                    stim_fname = stim_fname[:-4] + '-epo.fif'
                            else:
                                stim_fname += '-epo.fif'
                            stim_path = os.path.join(os.getcwd(), stim_fname)
                            epochs_stim.save(stim_path, overwrite=True)
                            print(f"✓ 已儲存 Stimulus epochs: {stim_fname}")

                            # 儲存 Response epochs
                            default_resp_fname = f"{subject_id_from_epochs}_ASRT_resp_{phase_tag}-epo.fif"
                            resp_fname = input(f"\n請輸入 Response epochs 檔名 [預設: {default_resp_fname}]: ").strip() or default_resp_fname
                            if resp_fname.endswith('.fif.gz'):
                                pass
                            elif resp_fname.endswith('.fif'):
                                if not ('-epo.fif' in resp_fname or '_epo.fif' in resp_fname):
                                    resp_fname = resp_fname[:-4] + '-epo.fif'
                            else:
                                resp_fname += '-epo.fif'
                            resp_path = os.path.join(os.getcwd(), resp_fname)
                            epochs_resp.save(resp_path, overwrite=True)
                            print(f"✓ 已儲存 Response epochs: {resp_fname}")
                            
                            current_epochs = epochs_resp
                        
                    else:
                        # ============================================================
                        # 單一 Epochs 模式（保持原邏輯）
                        # ============================================================
                        current_epochs = result
                        
                        # === 詢問是否排除極端值 ===
                        if len(current_epochs) > 0:
                            print("\n" + "="*60)
                            print("極端值排除（Artifact Rejection）")
                            print("="*60)
                            do_reject = input("\n是否要進行極端值排除？(y/n) [預設 n]: ").strip().lower()
                            
                            if do_reject == 'y':
                                print("\n選擇極端值排除方法:")
                                print("1. Flexible (彈性閾值 + Visual Inspection)")
                                print("2. Fixed (固定閾值 + Visual Inspection)")
                                print("3. Autoreject (自動排除)")
                                
                                method_choice = input("\n請選擇 (1/2/3) [預設 1]: ").strip() or '1'
                                method_map = {'1': 'flexible', '2': 'fixed', '3': 'autoreject'}
                                method = method_map.get(method_choice, 'flexible')
                                
                                try:
                                    epochs_clean, log = asrt_artifact_rejection(current_epochs, method=method)
                                    if epochs_clean is not None:
                                        current_epochs = epochs_clean
                                        processing_history.append(f"極端值排除 ({method})")
                                        print(f"\n✓ Epochs 已更新")
                                        print(f"  保留: {len(current_epochs)} epochs")
                                        print(f"  保留率: {log['retention_rate']:.1f}%")
                                except Exception as e:
                                    print(f"極端值排除時發生錯誤: {str(e)}")
                                    import traceback
                                    traceback.print_exc()
                    
            except Exception as e:
                print(f"建立 Epochs 時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()
        
        elif choice == '8':
            display_epochs_plot(current_epochs)

        elif choice == '9':
            psd = compute_epochs_psd(current_epochs)
            if psd is not None:
                processing_history.append("計算 PSD")

        elif choice == '10':
            tfr = compute_epochs_tfr(current_epochs)
            if tfr is not None:
                processing_history.append("計算 TFR")

        # === ASRT 分析 (選項 15-20) ===
        # 來源：原 main.py 2161-2424 行
        elif choice == '15':
            # ASRT ERSP 分析
            if current_epochs is None:
                print("\n⚠️  請先建立 Epochs")
                print("方法 1: 使用選項 8/9 建立 Epochs")
                print("方法 2: 載入已儲存的 Epochs 檔案（主選單選項 5）")
                continue
            
            # 檢查 Epochs 是否為空
            if len(current_epochs) == 0:
                print("\n⚠️  Epochs 是空的（所有 epochs 都被拒絕）！")
                print("可能原因：")
                print("  1. reject 參數太嚴格，所有 epochs 都被拒絕")
                print("  2. 事件檔案有問題")
                print("\n建議：")
                print("  • epochs.py 已修正（reject_criteria = None）")
                print("  • 請重新建立 Epochs")
                print("  • 或檢查事件標記是否正確")
                continue
            
            try:
                import glob as _glob

                print("\n" + "="*60)
                print("ASRT ERSP 分析（Stimulus-locked）")
                print("="*60)

                # ── 自動偵測並確認使用 Stimulus epochs ──
                def _is_stimulus_locked(ep):
                    return ep.tmin >= -0.9 and ep.tmax >= 0.8

                stim_epochs = None

                # 1. 先提取乾淨的 base ID（去掉 _resp/_stim 等 epoch 後綴）
                import re as _re
                _base_sid = _re.sub(
                    r'[_-](resp|stim|ASRT)[_-].*$', '', subject_id, flags=_re.IGNORECASE
                ).rstrip('_-') or subject_id

                # 2. 在當前目錄搜尋 stim epochs 檔案
                # 先用受試者 ID 搜尋，若失敗再用寬泛模式
                search_patterns = [
                    f"*{_base_sid}*stim*-epo.fif",
                    f"*{_base_sid}*stim*_epo.fif",
                    f"*{_base_sid}*ASRT*stim*-epo.fif",
                    f"*{_base_sid}*stim*.fif",
                    f"*stim*-epo.fif",   # 寬泛 fallback（不限受試者 ID）
                    f"*stim*_epo.fif",
                    f"*stim*.fif",
                ]
                found_files = []
                for pat in search_patterns:
                    found_files = _glob.glob(pat)
                    if found_files:
                        break

                if found_files:
                    if len(found_files) > 1:
                        print(f"\n  找到多個 Stimulus 檔案，請選擇：")
                        for i, f in enumerate(found_files):
                            print(f"    {i+1}. {f}")
                        sel = input(f"  請輸入編號 [1]: ").strip()
                        try:
                            stim_path = found_files[int(sel) - 1] if sel else found_files[0]
                        except (ValueError, IndexError):
                            stim_path = found_files[0]
                    else:
                        stim_path = found_files[0]
                        print(f"\n  自動找到 Stimulus epochs: {stim_path}")
                    print(f"  載入中...")
                    stim_epochs = mne.read_epochs(stim_path, preload=True)
                    print(f"  ✓ 載入完成：{len(stim_epochs)} 個 epochs"
                          f"（{stim_epochs.tmin:.2f} ~ {stim_epochs.tmax:.2f} s）")

                elif _is_stimulus_locked(current_epochs):
                    # 沒有找到檔案，但 current_epochs 本身是 Stimulus-locked
                    stim_epochs = current_epochs
                    print(f"\n  使用當前 Stimulus epochs（{len(stim_epochs)} trials）")

                else:
                    # 找不到檔案，且 current_epochs 是 Response-locked → 讓使用者手動輸入路徑
                    print(f"\n⚠️  在當前目錄找不到 stim epochs 檔案")
                    print(f"  當前 Epochs 為 Response-locked"
                          f"（tmin={current_epochs.tmin:.2f}, tmax={current_epochs.tmax:.2f} s）")
                    print(f"\n請手動輸入 Stimulus epochs 檔案路徑（直接 Enter 取消）：")
                    manual_path = input("  路徑: ").strip().strip('"')
                    if not manual_path:
                        print("取消")
                        continue
#                    import os as _os
                    if not _os.path.exists(manual_path):
                        print(f"❌ 找不到檔案: {manual_path}")
                        continue
                    print(f"  載入中...")
                    stim_epochs = mne.read_epochs(manual_path, preload=True)
                    print(f"  ✓ 載入完成：{len(stim_epochs)} 個 epochs"
                          f"（{stim_epochs.tmin:.2f} ~ {stim_epochs.tmax:.2f} s）")

                # ── 確認 Stimulus-locked ──
                if not _is_stimulus_locked(stim_epochs):
                    print(f"\n⚠️  載入的 epochs 不符合 Stimulus-locked 條件")
                    print(f"  tmin={stim_epochs.tmin:.2f}, tmax={stim_epochs.tmax:.2f} s")
                    print(f"  （需要 tmin ≥ -0.9 且 tmax ≥ 0.8）")
                    continue

                # ── 檢查 metadata ──
                if stim_epochs.metadata is None:
                    print("⚠️  Epochs 缺少 metadata")
                    print("ERSP 分析需要 metadata 來區分條件（Regular/Random 等）")
                    cont = input("\n仍要繼續嗎？(y/n) [n]: ").strip().lower()
                    if cont != 'y':
                        continue

                # ── 顯示 Epochs 資訊 ──
                print(f"\n✓ Epochs 資訊:")
                print(f"  總 epochs: {len(stim_epochs)}")
                print(f"  通道數: {len(stim_epochs.ch_names)}")
                print(f"  時間範圍: {stim_epochs.tmin:.2f} ~ {stim_epochs.tmax:.2f} s")
                print(f"  採樣率: {stim_epochs.info['sfreq']} Hz")
                print(f"  類型: Stimulus-locked ✓")

                # ── Baseline 設定 ──
                print("\nBaseline 設定:")
                print("  1. 只做 FD baseline（推薦）")
                print("  2. TD + FD baseline")
                bl_choice = input("請選擇 (1/2) [1]: ").strip() or '1'
                do_td_baseline = bl_choice == '2'

                print("\nBaseline 方法:")
                print("  1. 固定時間窗口（pre-stimulus blank 期）")
                print("  2. 整段 epoch 平均（Lum et al. 2023）")
                bl_method_choice = input("請選擇 (1/2) [1]: ").strip() or '1'
                baseline_method = 'whole_epoch' if bl_method_choice == '2' else 'pre_stim'

                # ── 群體分析選項 ──
                print("\n" + "─"*60)
                print("群體分析選項：")
                print("  • 選擇「是」會將此受試者的 ERSP 資料儲存到 C:\\Experiment\\Result\\h5")
                print("  • 之後可使用「選項 16: 群體分析」整合多位受試者")
                save_choice = input("是否儲存 ERSP 資料供群體分析用？(y/n) [預設 y]: ").strip().lower() or 'y'
                save_for_group = (save_choice == 'y')
                print("─"*60)

                # ── 使用已提取的 base ID 作為分析用受試者 ID ──
                analysis_sid = _base_sid

                # ── 自動執行 Learning + Testing 兩個階段 ──
                for phase in ['learning', 'testing']:
                    print(f"\n{'='*60}")
                    print(f"開始 ERSP 分析（{phase.capitalize()} 階段）...")
                    print(f"{'='*60}")
                    print(f"  受試者: {analysis_sid}")
                    print(f"  階段: {phase.capitalize()}")
                    print(f"  鎖定: Stimulus-locked")
                    print(f"  Epochs: {len(stim_epochs)}")
                    print(f"  儲存群體資料: {'是' if save_for_group else '否'}")

                    print("\n輸出資料夾:")
                    print(f"  1. Trigger 分類結果 → C:\\Experiment\\ersp_results")
                    print(f"  2. Triplet 分類結果 → C:\\Experiment\\ersp_results\\triplet")
                    dir_choice = input("請選擇 (1/2) [1]: ").strip() or '1'
                    output_dir = r'C:\Experiment\ersp_results\triplet' if dir_choice == '2' else r'C:\Experiment\ersp_results'

                    result = asrt_ersp_full_analysis(
                        stim_epochs,
                        subject_id=analysis_sid,
                        phase=phase,
                        lock_type='stimulus',
                        output_dir=output_dir,
                        save_for_group_analysis=save_for_group,
                        group_data_dir=r'C:\Experiment\Result\h5',
                        do_td_baseline=do_td_baseline,
                        baseline_method=baseline_method,
                    )

                    if result:
                        asrt_results['ersp'] = result
                        processing_history.append(f"ASRT ERSP 分析 ({phase}, stimulus)")
                        print("\n" + "="*60)
                        print(f"✓ ASRT ERSP 分析完成（{phase.capitalize()} 階段）")
                        print("="*60)
                        print(f"\n結果已儲存至: ./ersp_results")
                        if save_for_group:
                            print(f"群體分析資料已儲存至: C:\\Experiment\\Result\\h5")

                # ── 單一電極個人圖（Stimulus-locked）──
                if save_for_group:
                    print("\n" + "─"*60)
                    elec_choice15 = input("是否產生單一電極 Stimulus ERSP 比較圖？(y/n) [n]: ").strip().lower() or 'n'
                    if elec_choice15 == 'y':
                        elec_input15 = input("請輸入電極名稱（逗號分隔，例如: Fz,Cz,Pz）: ").strip()
                        if elec_input15:
                            electrodes15 = [e.strip() for e in elec_input15.split(',') if e.strip()]
                            _h5_dir15 = r'C:\Experiment\Result\h5'

                            is_triplet15 = (
                                hasattr(stim_epochs, 'metadata') and
                                stim_epochs.metadata is not None and
                                'classification' in stim_epochs.metadata.columns and
                                stim_epochs.metadata['classification'].iloc[0] == 'triplet'
                            )
                            lbl_left15  = 'high' if is_triplet15 else 'Regular'
                            lbl_right15 = 'low'  if is_triplet15 else 'Random'

                            elec_out15 = os.path.join(_h5_dir15, '..', 'single_electrode')
                            os.makedirs(elec_out15, exist_ok=True)

                            import glob as _glob15
                            import numpy as _np15
                            import matplotlib.pyplot as _plt15
                            from mne_python_analysis.group_ersp_analysis import _load_h5_single_electrode

                            h5_files_reg15 = sorted(_glob15.glob(
                                os.path.join(_h5_dir15, f'{analysis_sid}_Stimulus_*_Regular_ERSP.h5')))

                            for electrode15 in electrodes15:
                                print(f"\n  電極: {electrode15}")
                                for fp_l15 in h5_files_reg15:
                                    fp_r15 = fp_l15.replace('_Regular_', '_Random_')
                                    if not os.path.exists(fp_r15):
                                        continue
                                    try:
                                        el15, freqs15, times15 = _load_h5_single_electrode(fp_l15, electrode15)
                                        er15, _, _             = _load_h5_single_electrode(fp_r15, electrode15)
                                    except Exception as ex15:
                                        print(f"    ⚠ {os.path.basename(fp_l15)}: {ex15}")
                                        continue

                                    diff15 = el15 - er15
                                    x_min15, x_max15 = -0.5, 0.5
                                    t_mask15 = (times15 >= x_min15) & (times15 <= x_max15)
                                    combined15 = _np15.concatenate([el15[:, t_mask15].ravel(), er15[:, t_mask15].ravel()])
                                    vmax15_c = _np15.percentile(_np15.abs(combined15), 95)
                                    vmax15_d = _np15.percentile(_np15.abs(diff15[:, t_mask15].ravel()), 95)
                                    lv15_c = _np15.linspace(-vmax15_c, vmax15_c, 20)
                                    lv15_d = _np15.linspace(-vmax15_d, vmax15_d, 20)

                                    base15 = os.path.basename(fp_l15).replace('_ERSP.h5', '').replace(f'{analysis_sid}_Stimulus_', '')
                                    block_label15 = '_'.join(base15.split('_')[:-1])

                                    fig15, axes15 = _plt15.subplots(1, 3, figsize=(18, 5))
                                    for ax15, data15, title15, lv15, vm15, cbl15 in [
                                        (axes15[0], el15,   f'{lbl_left15}',                          lv15_c, vmax15_c, 'Power (dB)'),
                                        (axes15[1], er15,   f'{lbl_right15}',                         lv15_c, vmax15_c, 'Power (dB)'),
                                        (axes15[2], diff15, f'Difference ({lbl_left15} - {lbl_right15})', lv15_d, vmax15_d, 'Power Difference (dB)'),
                                    ]:
                                        im15 = ax15.contourf(times15, freqs15, data15, levels=lv15,
                                                             cmap='RdBu_r', vmin=-vm15, vmax=vm15, extend='both')
                                        ax15.axvline(0, color='black', linestyle='--', linewidth=1.5)
                                        ax15.axhline(8,  color='white', linestyle=':', linewidth=1, alpha=0.6)
                                        ax15.axhline(13, color='white', linestyle=':', linewidth=1, alpha=0.6)
                                        ax15.set_xlabel('Time (s)', fontsize=11)
                                        ax15.set_ylabel('Frequency (Hz)', fontsize=11)
                                        ax15.set_title(title15, fontsize=11, fontweight='bold')
                                        ax15.set_xlim([x_min15, x_max15])
                                        _plt15.colorbar(im15, ax=ax15, label=cbl15)

                                    fig15.suptitle(
                                        f'{analysis_sid} | Stimulus-locked | {block_label15} | Electrode: {electrode15}',
                                        fontsize=12, fontweight='bold'
                                    )
                                    _plt15.tight_layout()
                                    out_fig15 = os.path.join(elec_out15,
                                        f'{analysis_sid}_stimulus_{electrode15}_{block_label15}_comparison.png')
                                    fig15.savefig(out_fig15, dpi=300, bbox_inches='tight')
                                    _plt15.close(fig15)
                                    print(f"    ✓ 已儲存: {out_fig15}")

                            processing_history.append(f"單一電極個人圖 Stimulus（{elec_input15}）")

            except Exception as e:
                print(f"\nERSP 分析時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()
        
        elif choice == '19':
            # ASRT ROI 頻譜分析
            try:
                result = asrt_roi_spectral_analysis(current_epochs, subject_id)
                if result:
                    asrt_results['roi_spectral'] = result
                    processing_history.append("ASRT ROI 頻譜分析")
                    print("✓ ASRT ROI 分析完成")
            except Exception as e:
                print(f"ASRT ROI 分析時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()

        elif choice == '20':
            # ASRT Block 比較分析
            try:
                result = asrt_block_comparison(current_epochs, subject_id)
                if result:
                    block_powers, block_stats, learning_effect = result
                    asrt_results['block_comparison'] = {
                        'powers': block_powers,
                        'stats': block_stats,
                        'learning': learning_effect
                    }
                    processing_history.append("ASRT Block 比較分析")
                    print("✓ ASRT Block 分析完成")
            except Exception as e:
                print(f"ASRT Block 分析時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()

        elif choice == '17':
            # ASRT 群體分析 — 全自動跑完所有組合
            try:
                from mne_python_analysis.group_ersp_analysis import auto_group_ersp_analysis

                print("\n" + "="*60)
                print("ASRT 群體分析 (Group-level ERSP)")
                print("="*60)
                print("\n自動執行全部組合：")
                print("  階段  : Learning + Testing")
                print("  測試  : Motor + Perceptual  (Testing 階段)")
                print("  鎖定  : Stimulus-locked + Response-locked")
                print("  ROI   : Theta + Alpha")

                print("\n分析類型:")
                print("  1. Trigger 分類（Regular / Random）")
                print("  2. Triplet 分類（high / low）")
                analysis_type = input("請選擇 (1/2) [1]: ").strip() or '1'

                if analysis_type == '2':
                    PKL_DIR    = r'C:\Experiment\Result\h5'
                    H5_DIR     = r'C:\Experiment\Result\h5'
                    OUTPUT_DIR = r'C:\Experiment\Result\triplet\group_ersp'
                    print(f"  → Triplet 分類")
                    print(f"  → Stimulus 資料來源：{PKL_DIR}")
                    print(f"  → Response 資料來源：{H5_DIR}")
                    print(f"  → 輸出：{OUTPUT_DIR}")
                else:
                    PKL_DIR    = r'C:\Experiment\Result\h5'
                    H5_DIR     = r'C:\Experiment\Result\h5'
                    OUTPUT_DIR = r'C:\Experiment\Result\group_ersp'
                    print(f"  → Trigger 分類")
                    print(f"  → Stimulus 資料來源：{PKL_DIR}")
                    print(f"  → Response 資料來源：{H5_DIR}")
                    print(f"  → 輸出：{OUTPUT_DIR}")

                print("\n⚠  資料前置需求：")
                print("   Stimulus-locked → 先執行選項 15，分析中選「儲存供群體分析用：y」")
                print("   Response-locked → 先執行選項 16")

                # 輸入受試者 ID
                print("\n" + "─"*60)
                subject_input = input(
                    "請輸入受試者 ID（以逗號分隔，例如: sub0001,sub0002,sub0003）: "
                ).strip()

                if not subject_input:
                    print("❌ 未輸入受試者 ID，取消分析")
                    continue

                subject_ids = [s.strip() for s in subject_input.split(',') if s.strip()]
                print(f"✓ 將分析 {len(subject_ids)} 位受試者: {subject_ids}")

                # 僅詢問 Permutation Test
                print("\n統計檢定:")
                print("  Cluster-based Permutation Test（建議受試者數 ≥ 8 再開啟）")
                do_perm = input("執行 Permutation Test？(y/n) [預設 n]: ").strip().lower() or 'n'
                do_permutation = (do_perm == 'y')

                print("\n" + "─"*60)
                print(f"受試者: {len(subject_ids)} 位")
                print(f"Permutation Test: {'是' if do_permutation else '否'}")
                confirm = input("\n確定執行全部組合？(y/n) [預設 y]: ").strip().lower() or 'y'

                if confirm != 'y':
                    print("❌ 取消分析")
                    continue

                print("\nColorbar 範圍設定:")
                print("  1. 每個 Block 各自計算（預設）")
                print("  2. 同一 ROI 跨所有 Block 統一（方便比較 Early vs Late）")
                cb_choice = input("請選擇 (1/2) [1]: ").strip() or '1'
                unified_colorbar = (cb_choice == '2')

                results = auto_group_ersp_analysis(
                    subject_ids         = subject_ids,
                    pkl_dir             = PKL_DIR,
                    h5_dir              = H5_DIR,
                    output_dir          = OUTPUT_DIR,
                    do_permutation_test = do_permutation,
                    n_permutations      = 1000,
                    unified_colorbar    = unified_colorbar,
                    display_label1      = 'low'  if analysis_type == '2' else None,
                    display_label2      = 'high' if analysis_type == '2' else None,
                )

                n_done = sum(1 for v in results.values() if v)
                processing_history.append(
                    f"ASRT 群體分析（全自動，{len(subject_ids)} 位受試者，{n_done} 個組合完成）"
                )

                # ── 單一電極群體分析 ──
                print("\n" + "─"*60)
                elec_choice = input("是否進行單一電極群體分析？(y/n) [n]: ").strip().lower() or 'n'
                if elec_choice == 'y':
                    elec_input = input("請輸入電極名稱（逗號分隔，例如: Fz,Cz,Pz）: ").strip()
                    if elec_input:
                        electrodes = [e.strip() for e in elec_input.split(',') if e.strip()]
                        elec_out = os.path.join(OUTPUT_DIR, 'single_electrode')

                        if analysis_type == '2':
                            lbl_left, lbl_right = 'high', 'low'
                        else:
                            lbl_left, lbl_right = 'Regular', 'Random'

                        from mne_python_analysis.group_ersp_analysis import run_single_electrode_group_analysis
                        run_single_electrode_group_analysis(
                            subject_ids     = subject_ids,
                            electrodes      = electrodes,
                            h5_dir          = H5_DIR,
                            output_dir      = elec_out,
                            label_left      = lbl_left,
                            label_right     = lbl_right,
                            condition_left  = 'Regular',
                            condition_right = 'Random',
                        )
                        processing_history.append(f"單一電極群體分析（{', '.join(electrodes)}）")

            except ImportError:
                print("❌ 群體分析模組未安裝")
                print("請確認 group_ersp_analysis.py 已放在 mne_python_analysis/ 目錄下")
            except Exception as e:
                print(f"❌ 群體分析時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()

        # ============================================================
        # 選項 18: EEG-行為整合分析
        # ============================================================
        elif choice == '18':
            try:
                from collections import Counter as _Counter
                import warnings as _warnings

                print("\n" + "="*60)
                print("EEG-行為整合分析（選項 18）")
                print("="*60)
                print("\n此功能直接從 PsychoPy 原始 CSV 提取行為資料，")
                print("與已計算好的 ERSP .h5 檔案整合，執行三個方向的分析：")
                print("  方向一：條件層次跨受試者相關（每人一點）")
                print("  方向二：Block 組學習曲線（RT + ERSP 雙軸）")
                print("  方向三：混合層次相關（可選 subject 或 subject×block）")

                # ── Block 對應查找表（與 epochs.py 完全一致）──────────────────
                _BLOCK_LOOKUP_18 = {
                    ('learning', 'Block7-11'):   (0,  4),
                    ('learning', 'Block12-16'):  (5,  9),
                    ('learning', 'Block17-21'):  (10, 14),
                    ('learning', 'Block22-26'):  (15, 19),
                    ('testing',  'Block27-28'):  (0,  1),
                    ('testing',  'Block29-30'):  (2,  3),
                    ('testing',  'Block31-32'):  (4,  5),
                    ('testing',  'Block33-34'):  (6,  7),
                }

                def _bid_to_group_18(block_type, bid):
                    for (bt, grp), (lo, hi) in _BLOCK_LOOKUP_18.items():
                        if bt == block_type and lo <= bid <= hi:
                            return grp
                    return None

                # ── ROI 定義（與 pipeline 一致）──────────────────────────────
                _ROI_GROUPS_18 = {
                    'Motor':                ['Fz', 'FCz', 'Cz', 'C3', 'C4'],
                    'Motor_Frontal':        ['Fz', 'FCz'],
                    'Motor_Central':        ['Cz', 'C3', 'C4'],
                    'Motor_Parietal':       ['P3', 'Pz', 'P4'],
                    'Motor_Occipital':      ['O1', 'Oz', 'O2'],
                    'Perceptual':           ['O1', 'Oz', 'O2', 'P3', 'Pz', 'P4'],
                    'Perceptual_Parietal':  ['P3', 'Pz', 'P4'],
                    'Perceptual_Occipital': ['O1', 'Oz', 'O2'],
                    'Perceptual_Frontal':   ['Fz', 'FCz'],
                    'Perceptual_Central':   ['Cz', 'C3', 'C4'],
                }

                # ────────────────────────────────────────────────────────────
                # 內部函數：從 PsychoPy CSV 提取並處理行為資料
                # 邏輯與 epochs.py create_asrt_epochs(trial_classification='triplet')
                # 及 R 的 assign_freq() 完全一致
                # ────────────────────────────────────────────────────────────
                def _build_behavior_18(psy_df, sid_label, test_ver, rt_min=150., rt_max=800., z_thr=2.0):
                    df = psy_df.copy()
                    # 篩選有 orientation_index 的列
                    if 'orientation_index' in df.columns:
                        df = df[df['orientation_index'].notna()].copy()

                    def _btype(row):
                        def _ne(v):
                            return pd.notna(v) and str(v).strip() not in ('', 'nan')
                        if _ne(row.get('initial_random_seq_files', '')):  return 'random'
                        if _ne(row.get('practice_seq_files', '')):        return 'practice'
                        if _ne(row.get('learning_seq_files', '')):        return 'learning'
                        if _ne(row.get('test_seq_files', '')):            return 'testing'
                        return 'unknown'

                    df['block_type'] = df.apply(_btype, axis=1)
                    df = df[df['block_type'].isin(['learning', 'testing'])].copy()

                    def _bid(row):
                        if row['block_type'] == 'learning':
                            v = row.get('learning_trials.thisTrialN', np.nan)
                        else:
                            v = row.get('combined_testing_trials.thisTrialN', np.nan)
                        return int(float(v)) if pd.notna(v) else np.nan

                    df['block_id'] = df.apply(_bid, axis=1)
                    df = df[df['block_id'].notna()].copy()
                    df['block_id'] = df['block_id'].astype(int)

                    def _tib(row):
                        if row['block_type'] == 'learning':
                            v = row.get('learning_loop.thisTrialN', np.nan)
                        else:
                            for c in ['motor_percept_testing_loop.thisTrialN',
                                      'percept_motor_testing_loop.thisTrialN']:
                                if c in row.index and pd.notna(row.get(c)):
                                    return float(row[c])
                            v = np.nan
                        return float(v) if pd.notna(v) else np.nan

                    df['trial_in_block'] = df.apply(_tib, axis=1)
                    df['rt_ms']   = pd.to_numeric(df.get('key_resp.rt',   np.nan), errors='coerce') * 1000.
                    df['accuracy'] = pd.to_numeric(df.get('key_resp.corr', np.nan), errors='coerce')

                    # trial_rank & position_type（仿 R: rank(trial)-6，奇偶判斷）
                    df = df.sort_values(['block_type', 'block_id', 'trial_in_block'])
                    df['trial_rank'] = df.groupby(['block_type', 'block_id']).cumcount() - 5
                    df = df[df['trial_rank'] >= 0].copy()
                    df['position_type'] = df['trial_rank'].apply(lambda r: 'regular' if r % 2 == 0 else 'random')

                    # global block num & phase & test_type
                    def _gblk(row):
                        return row['block_id'] + 7 if row['block_type'] == 'learning' else row['block_id'] + 27
                    df['block_global'] = df.apply(_gblk, axis=1)
                    df['phase'] = df['block_global'].apply(lambda g: 'Learning' if g <= 26 else 'Test')
                    def _tt(g):
                        if g < 27 or g > 34: return None
                        if test_ver == 'motor_first':
                            return 'motor' if g in [27,28,33,34] else 'perceptual'
                        return 'perceptual' if g in [27,28,33,34] else 'motor'
                    df['test_type'] = df['block_global'].apply(_tt)

                    # Triplet 建立
                    def _triplets(seq):
                        r = [None, None]
                        for i in range(2, len(seq)):
                            n2,n1,n = seq[i-2],seq[i-1],seq[i]
                            if str(n2)==str(n1)==str(n) or str(n2)==str(n):
                                r.append(None)
                            else:
                                r.append(f"{n2}{n1}{n}")
                        return r

                    trip_map = {}
                    for (bt, bid), grp in df.groupby(['block_type', 'block_id']):
                        gs = grp.sort_values('trial_in_block')
                        if 'correct_answer_index' not in gs.columns:
                            continue
                        seq = gs['correct_answer_index'].astype(float).fillna(0).astype(int).tolist()
                        for idx, tri in zip(gs.index, _triplets(seq)):
                            trip_map[idx] = tri
                    df['triplet'] = df.index.map(lambda i: trip_map.get(i))

                    # Frequency category（仿 epochs.py _assign_types()）
                    lrm = (df['phase']=='Learning') & (df['position_type']=='random') & df['triplet'].notna()
                    rc = _Counter(df.loc[lrm,'triplet'].tolist())
                    med_c = np.median(list(rc.values())) if rc else 0
                    print(f"  [{sid_label}] triplet 種類: {len(rc)}, median count: {med_c:.1f}")

                    def _fcat(row):
                        if row['position_type'] == 'regular': return 'high'
                        if row['position_type'] == 'random':
                            if pd.isna(row['triplet']): return None
                            return 'high' if rc.get(row['triplet'], 0) >= med_c else 'low'
                        return None

                    df['frequency_category'] = df.apply(_fcat, axis=1)
                    df = df[df['frequency_category'].notna()].copy()
                    df['trial_type'] = df['frequency_category']

                    # eeg_block_group
                    df['eeg_block_group'] = df.apply(
                        lambda r: _bid_to_group_18(r['block_type'], r['block_id']), axis=1)
                    df = df[df['eeg_block_group'].notna()].copy()

                    # RT 過濾（仿 R assign_freq()：150-800ms, |z|≤2 within block×trial_type）
                    n0 = len(df)
                    df = df[df['rt_ms'].between(rt_min, rt_max)].copy()
                    gcols = ['block_type', 'block_id', 'trial_type']
                    df['rt_z'] = df.groupby(gcols)['rt_ms'].transform(
                        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.)
                    df = df[df['rt_z'].abs() <= z_thr].drop(columns=['rt_z'])
                    print(f"  [{sid_label}] RT 過濾: {n0} → {len(df)} trials")

                    df['sid'] = sid_label
                    df['triplet_median_count'] = med_c   # 記錄供驗證用
                    return df, med_c

                # ────────────────────────────────────────────────────────────
                # 內部函數：聚合行為資料（block_group 層次）
                # ────────────────────────────────────────────────────────────
                def _agg_behavior_18(bdf, use_med=True):
                    fn = 'median' if use_med else 'mean'
                    bdf = bdf.copy()
                    bdf['test_type'] = bdf['test_type'].fillna('none')
                    grp = ['sid','block_type','eeg_block_group','phase','test_type',
                           'trial_type','frequency_category']
                    return (bdf.groupby(grp)['rt_ms']
                               .agg([fn, 'count'])
                               .rename(columns={fn:'rt','count':'n_trials'})
                               .reset_index())

                # ────────────────────────────────────────────────────────────
                # 內部函數：讀取 ERSP .h5
                # ────────────────────────────────────────────────────────────
                def _load_ersp_18(h5_dir_path, sids, ltype):
                    prefix = 'Response' if ltype == 'response' else 'Stimulus'
                    freq_r = (4., 8.)   if ltype == 'response' else (8., 13.)
                    time_r = (-.300, .050) if ltype == 'response' else (.100, .300)
                    rows = []
                    for fp in sorted(Path(h5_dir_path).glob(f'*_{prefix}_*.h5')):
                        stem = fp.stem.replace('_ERSP', '')
                        parts = stem.split('_')
                        try:
                            trial_type  = parts[-1].lower()
                            block_group = parts[-2]
                            phase       = parts[-3]
                            sid18       = '_'.join(parts[:-4])
                            if parts[-4] != prefix or sid18 not in sids:
                                continue
                        except IndexError:
                            continue
                        try:
                            with _warnings.catch_warnings():
                                _warnings.simplefilter('ignore')
                                tl = mne.time_frequency.read_tfrs(str(fp))
                            tfr = tl[0] if isinstance(tl, list) else tl
                        except Exception as _e:
                            print(f"  ⚠ 讀取失敗 {fp.name}: {_e}")
                            continue
                        freqs18, times18 = tfr.freqs, tfr.times
                        fm = (freqs18 >= freq_r[0]) & (freqs18 <= freq_r[1])
                        tm = (times18 >= time_r[0]) & (times18 <= time_r[1])
                        cnu = [ch.upper() for ch in tfr.ch_names]
                        for rname, rchs in _ROI_GROUPS_18.items():
                            ridx = [cnu.index(c.upper()) for c in rchs if c.upper() in cnu]
                            if not ridx: continue
                            val = tfr.data[ridx].mean(axis=0)[np.ix_(fm,tm)].mean()
                            rows.append({'sid':sid18,'lock_type':ltype,'phase':phase,
                                         'eeg_block_group':block_group,'trial_type':trial_type,
                                         'roi':rname,'ersp_mean':float(val)})
                    df18 = pd.DataFrame(rows)
                    print(f"  ✓ ERSP: {len(df18)} rows, {df18['sid'].nunique() if len(df18) else 0} 受試者")
                    return df18

                # ────────────────────────────────────────────────────────────
                # 內部函數：Join
                # ────────────────────────────────────────────────────────────
                def _join_18(ersp_df, beh_sum):
                    def _p2bt(p): return 'learning' if p=='Learning' else 'testing'
                    def _p2tt(p):
                        if p=='MotorTest': return 'motor'
                        if p=='PerceptualTest': return 'perceptual'
                        return 'none'
                    e = ersp_df.copy()
                    e['block_type'] = e['phase'].map(_p2bt)
                    e['tasktype']   = e['phase'].map(_p2tt)
                    b = beh_sum.copy()
                    b['tasktype'] = b['test_type']
                    # phase 欄位在兩邊都存在會造成衝突，先從行為資料側移除
                    b = b.drop(columns=[c for c in ['phase', 'block_type'] if c in b.columns])
                    joined = e.merge(b, on=['sid','eeg_block_group','trial_type','tasktype'],
                                     how='inner')
                    # 確保 phase 欄位存在（來自 EEG 側）
                    if 'phase' not in joined.columns and 'phase_eeg' in joined.columns:
                        joined['phase'] = joined['phase_eeg']
                    print(f"  ✓ Join: {len(joined)} rows, {joined['sid'].nunique()} 受試者")
                    print(f"  欄位: {list(joined.columns)}")
                    return joined

                # ────────────────────────────────────────────────────────────
                # 內部函數：方向一繪圖（跨受試者相關）
                # ────────────────────────────────────────────────────────────
                def _plot_dir1_18(joined, roi, phase, method, out_path):
                    from scipy import stats as _stats
                    d = joined[(joined['roi']==roi) & (joined['phase']==phase)].copy()
                    recs = []
                    for sid18, g in d.groupby('sid'):
                        rt_h  = g[g['trial_type']=='high']['rt'].mean()
                        rt_l  = g[g['trial_type']=='low']['rt'].mean()
                        erp_h = g[g['trial_type']=='high']['ersp_mean'].mean()
                        erp_l = g[g['trial_type']=='low']['ersp_mean'].mean()
                        recs.append({'sid':sid18,'x':rt_h-rt_l,'y':erp_h-erp_l})
                    pdf = pd.DataFrame(recs).dropna(subset=['x','y'])
                    if len(pdf) < 3:
                        print(f"  ⚠ 受試者不足（n={len(pdf)}），跳過方向一: {roi} × {phase}")
                        return None, {}
                    x, y = pdf['x'].values, pdf['y'].values
                    r_val, p_val = (_stats.spearmanr(x,y) if method=='spearman'
                                    else _stats.pearsonr(x,y))
                    fig, ax = plt.subplots(figsize=(7,6))
                    ax.scatter(x, y, s=80, color='#2196F3', zorder=5)
                    for _, row18 in pdf.iterrows():
                        ax.annotate(row18['sid'], (row18['x'],row18['y']),
                                    fontsize=8, color='gray',
                                    xytext=(5,5), textcoords='offset points')
                    if len(pdf) >= 3:
                        m18,b18 = np.polyfit(x,y,1)
                        xl18 = np.linspace(x.min(),x.max(),100)
                        ax.plot(xl18, m18*xl18+b18, color='#FF5722', lw=1.5, alpha=.7)
                    ax.axhline(0, color='gray', ls='--', lw=.8)
                    ax.axvline(0, color='gray', ls='--', lw=.8)
                    ax.text(.97,.97,
                            f'{method.capitalize()} r={r_val:.3f}\np={p_val:.3f}\nn={len(pdf)}',
                            transform=ax.transAxes, ha='right', va='top', fontsize=10,
                            bbox=dict(boxstyle='round', fc='white', alpha=.8))
                    ax.set_xlabel('RT Effect: High - Low (ms)', fontsize=11)
                    ax.set_ylabel('ERSP Effect: High - Low (dB)', fontsize=11)
                    ax.set_title(f'EEG-行為相關 | {roi} | {phase}',
                                 fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    fig.savefig(out_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  ✓ 方向一圖: {os.path.basename(out_path)}")
                    return fig, {'r':r_val,'p':p_val,'n':len(pdf)}

                # ────────────────────────────────────────────────────────────
                # 內部函數：方向二繪圖（學習曲線雙軸）
                # ────────────────────────────────────────────────────────────
                def _plot_dir2_18(joined, roi, phase, group_level, out_path):
                    _BLK_ORD = ['Block7-11','Block12-16','Block17-21','Block22-26',
                                'Block27-28','Block29-30','Block31-32','Block33-34']
                    d = joined[(joined['roi']==roi) & (joined['phase']==phase)].copy()
                    curve = d.groupby(['sid','eeg_block_group','trial_type']).agg(
                        rt_med=('rt','median'), ersp_avg=('ersp_mean','mean')
                    ).reset_index()
                    if group_level and curve['sid'].nunique() > 1:
                        curve = curve.groupby(['eeg_block_group','trial_type']).agg(
                            rt_se=('rt_med', lambda x: x.std()/np.sqrt(len(x))),
                            ersp_se=('ersp_avg', lambda x: x.std()/np.sqrt(len(x))),
                            rt_med=('rt_med','mean'), ersp_avg=('ersp_avg','mean'),
                            n_sub=('rt_med','count')
                        ).reset_index()
                        title_s = f'Group (N={int(curve["n_sub"].max())})'
                    else:
                        title_s = ', '.join(d['sid'].unique())
                    pblks = [b for b in _BLK_ORD if b in curve['eeg_block_group'].values]
                    curve['bord'] = curve['eeg_block_group'].map({b:i for i,b in enumerate(pblks)})
                    curve = curve.sort_values('bord')
                    colors18 = {'high':'#E74C3C','low':'#3498DB'}
                    fig2, ax1 = plt.subplots(figsize=(10,5))
                    ax2 = ax1.twinx()
                    for tt18, g18 in curve.groupby('trial_type'):
                        c18 = colors18.get(tt18,'gray')
                        ax1.plot(g18['bord'], g18['rt_med'], color=c18,
                                 marker='o', lw=1.8, label=f'RT {tt18}')
                        ax2.plot(g18['bord'], g18['ersp_avg'], color=c18,
                                 marker='^', lw=1.8, ls='--', label=f'ERSP {tt18}')
                    ax1.set_xticks(range(len(pblks)))
                    ax1.set_xticklabels(pblks, rotation=30, ha='right')
                    ax1.set_xlabel('Block Group', fontsize=11)
                    ax1.set_ylabel('RT (ms)', fontsize=11)
                    ax2.set_ylabel('ERSP (dB)', fontsize=11)
                    ax1.set_title(f'學習曲線 | {roi} | {phase} | {title_s}',
                                  fontsize=12, fontweight='bold')
                    l1,lb1 = ax1.get_legend_handles_labels()
                    l2,lb2 = ax2.get_legend_handles_labels()
                    ax1.legend(l1+l2, lb1+lb2, loc='upper right', fontsize=9)
                    plt.tight_layout()
                    fig2.savefig(out_path, dpi=200, bbox_inches='tight')
                    plt.close(fig2)
                    print(f"  ✓ 方向二圖: {os.path.basename(out_path)}")

                # ────────────────────────────────────────────────────────────
                # 內部函數：方向三繪圖（混合層次相關）
                # ────────────────────────────────────────────────────────────
                def _plot_dir3_18(joined, roi, phase, level, method, out_path):
                    from scipy import stats as _stats
                    d = joined[(joined['roi']==roi) & (joined['phase']==phase)].copy()
                    grp_cols = ['sid'] if level=='subject' else ['sid','eeg_block_group']
                    recs = []
                    for keys18, g18 in d.groupby(grp_cols):
                        sid18 = keys18 if isinstance(keys18,str) else keys18[0]
                        blk18 = None if level=='subject' else keys18[1]
                        rt_h  = g18[g18['trial_type']=='high']['rt'].mean()
                        rt_l  = g18[g18['trial_type']=='low']['rt'].mean()
                        ep_h  = g18[g18['trial_type']=='high']['ersp_mean'].mean()
                        ep_l  = g18[g18['trial_type']=='low']['ersp_mean'].mean()
                        recs.append({'sid':sid18,'block':blk18,'x':rt_h-rt_l,'y':ep_h-ep_l})
                    pdf3 = pd.DataFrame(recs).dropna(subset=['x','y'])
                    if len(pdf3) < 3:
                        print(f"  ⚠ 資料點不足（n={len(pdf3)}），跳過方向三: {roi} × {phase}")
                        return None, {}
                    x3,y3 = pdf3['x'].values, pdf3['y'].values
                    r3,p3 = (_stats.spearmanr(x3,y3) if method=='spearman'
                             else _stats.pearsonr(x3,y3))
                    cmap18 = plt.cm.get_cmap('tab10', pdf3['sid'].nunique())
                    sc18 = {s:cmap18(i) for i,s in enumerate(pdf3['sid'].unique())}
                    fig3, ax3 = plt.subplots(figsize=(7,6))
                    for _, r18 in pdf3.iterrows():
                        ax3.scatter(r18['x'], r18['y'], color=sc18[r18['sid']], s=70, zorder=5)
                        lbl3 = r18['sid'] if level=='subject' else f"{r18['sid']}\n{r18['block']}"
                        ax3.annotate(lbl3,(r18['x'],r18['y']), fontsize=7, color='gray',
                                     xytext=(4,4), textcoords='offset points')
                    if len(pdf3) >= 3:
                        m3,b3 = np.polyfit(x3,y3,1)
                        xl3 = np.linspace(x3.min(),x3.max(),100)
                        ax3.plot(xl3, m3*xl3+b3, color='gray', lw=1.5, alpha=.6)
                    ax3.axhline(0, color='gray', ls='--', lw=.8)
                    ax3.axvline(0, color='gray', ls='--', lw=.8)
                    ax3.text(.97,.97,
                             f'{method.capitalize()} r={r3:.3f}\np={p3:.3f}\nn={len(pdf3)}',
                             transform=ax3.transAxes, ha='right', va='top', fontsize=10,
                             bbox=dict(boxstyle='round', fc='white', alpha=.8))
                    ll3 = '每人一點' if level=='subject' else '每人×每Block Group一點'
                    ax3.set_xlabel('RT Effect: High - Low (ms)', fontsize=11)
                    ax3.set_ylabel('ERSP Effect: High - Low (dB)', fontsize=11)
                    ax3.set_title(f'EEG-行為相關 [{ll3}] | {roi} | {phase}',
                                  fontsize=11, fontweight='bold')
                    plt.tight_layout()
                    fig3.savefig(out_path, dpi=200, bbox_inches='tight')
                    plt.close(fig3)
                    print(f"  ✓ 方向三圖: {os.path.basename(out_path)}")
                    return fig3, {'r':r3,'p':p3,'n':len(pdf3)}

                # ════════════════════════════════════════════════════════════
                # 主流程：使用者輸入
                # ════════════════════════════════════════════════════════════

                # ── Step 1: 受試者 CSV 路徑 ──────────────────────────────────
                print("\n" + "─"*60)
                print("Step 1：輸入受試者資訊")
                print("─"*60)
                print("（每次輸入一位，留空 Enter 結束輸入）")
                print("  test_version 說明：")
                print("    motor_first    → Testing block 27,28,33,34 = motor")
                print("    perceptual_first → Testing block 27,28,33,34 = perceptual")

                # 改成 {sid: {'path': ..., 'test_version': ...}} 支援每人獨立設定
                psychopy_csv_paths_18 = {}
                while True:
                    sid18_input = input("\n  受試者 ID（Enter 結束）: ").strip()
                    if not sid18_input:
                        if not psychopy_csv_paths_18:
                            print("  ⚠ 至少需要輸入一位受試者")
                            continue
                        break
                    csv18_path = input(f"  {sid18_input} 的 PsychoPy CSV 路徑: ").strip().strip('"')
                    if not os.path.exists(csv18_path):
                        print(f"  ⚠ 找不到檔案: {csv18_path}")
                        continue
                    print(f"  {sid18_input} 的 Testing block 版本:")
                    print(f"    1. motor_first（27,28,33,34=motor; 29-32=perceptual）")
                    print(f"    2. perceptual_first（27,28,33,34=perceptual; 29-32=motor）")
                    tv18 = input(f"  請選擇 (1/2) [1]: ").strip() or '1'
                    test_ver18 = 'motor_first' if tv18 != '2' else 'perceptual_first'
                    psychopy_csv_paths_18[sid18_input] = {
                        'path': csv18_path,
                        'test_version': test_ver18
                    }
                    print(f"  ✓ 已加入 {sid18_input}（{test_ver18}）")

                subject_ids_18 = list(psychopy_csv_paths_18.keys())
                print(f"\n  受試者設定：")
                for _s, _info in psychopy_csv_paths_18.items():
                    print(f"    {_s}: {_info['test_version']}")

                # ── Step 2: 分析設定 ─────────────────────────────────────────
                print("\n" + "─"*60)
                print("Step 2：分析設定")
                print("─"*60)

                # H5 目錄
                _default_h5 = r'C:\Experiment\Result\h5'
                h5_dir_18 = input(f"  ERSP .h5 檔案目錄 [{_default_h5}]: ").strip().strip('"') or _default_h5

                # 輸出目錄
                _default_out = r'C:\Experiment\Result\eeg_behavior'
                out_dir_18 = input(f"  輸出目錄 [{_default_out}]: ").strip().strip('"') or _default_out

                # Lock type
                print("\n  Lock type:")
                print("    1. response（預設）")
                print("    2. stimulus")
                ltype_choice = input("  請選擇 (1/2) [1]: ").strip() or '1'
                lock_type_18 = 'response' if ltype_choice != '2' else 'stimulus'

                # 相關方法
                print("\n  相關分析方法:")
                print("    1. spearman（預設）")
                print("    2. pearson")
                cm_choice = input("  請選擇 (1/2) [1]: ").strip() or '1'
                corr_method_18 = 'spearman' if cm_choice != '2' else 'pearson'

                # 方向三層次
                print("\n  方向三分析層次:")
                print("    1. subject（每人一點，預設）")
                print("    2. block（每人×每 Block Group 一點）")
                d3_choice = input("  請選擇 (1/2) [1]: ").strip() or '1'
                dir3_level_18 = 'subject' if d3_choice != '2' else 'block'

                # 方向二：群體平均？
                gl_choice = input("\n  方向二是否顯示群體平均（y=群體, n=個別受試者）[y]: ").strip().lower() or 'y'
                group_level_18 = (gl_choice == 'y')

                # ROI 篩選
                print("\n  ROI 選擇（留空 Enter = 全部 10 個 ROI）:")
                roi_input_18 = input("  （逗號分隔，例如: Motor,Perceptual）: ").strip()
                if roi_input_18:
                    rois_18 = [r.strip() for r in roi_input_18.split(',') if r.strip() in _ROI_GROUPS_18]
                    if not rois_18:
                        rois_18 = list(_ROI_GROUPS_18.keys())
                        print("  ⚠ 無效的 ROI 名稱，改用全部 ROI")
                else:
                    rois_18 = list(_ROI_GROUPS_18.keys())

                phases_18 = ['Learning', 'MotorTest', 'PerceptualTest']

                print(f"\n  ✓ 設定確認")
                print(f"    Lock type:  {lock_type_18}")
                print(f"    相關方法:   {corr_method_18}")
                print(f"    方向三層次: {dir3_level_18}")
                print(f"    ROI 數量:   {len(rois_18)}")
                print(f"    受試者 test_version：")
                for _s, _info in psychopy_csv_paths_18.items():
                    print(f"      {_s}: {_info['test_version']}")
                confirm_18 = input("\n  確定執行？(y/n) [y]: ").strip().lower() or 'y'
                if confirm_18 != 'y':
                    print("  取消")
                    continue

                # ── Step 3: 行為資料 ─────────────────────────────────────────
                print("\n" + "─"*60)
                print("Step 3：讀取並處理行為資料")
                print("─"*60)

                all_beh_18 = []
                _verif_dir_18 = os.path.join(out_dir_18, 'verification')
                os.makedirs(_verif_dir_18, exist_ok=True)

                for sid18, cpath18 in psychopy_csv_paths_18.items():
                    print(f"\n  處理 {sid18}...")
                    try:
                        # 取出這位受試者的 CSV 路徑和 test_version
                        _info18    = cpath18 if isinstance(cpath18, dict) else {'path': cpath18, 'test_version': 'motor_first'}
                        _csv18     = _info18['path']
                        _tv18      = _info18['test_version']
                        print(f"    test_version: {_tv18}")
                        psy18 = pd.read_csv(_csv18)
                        beh18, med_c_18 = _build_behavior_18(psy18, sid18, _tv18)
                        all_beh_18.append(beh18)

                        # ── 輸出驗證 CSV（Python 側）──────────────────────────
                        # Learning 行的 test_type = None，需先填為 'none'
                        # 否則 groupby 預設 dropna=True 會丟棄所有 Learning 行
                        _verif_grp = (
                            beh18
                            .assign(test_type=lambda x: x['test_type'].fillna('none'))
                            .groupby(['sid', 'block_type', 'eeg_block_group',
                                      'phase', 'test_type',
                                      'position_type', 'frequency_category'])
                            .agg(
                                n_trials   = ('rt_ms', 'count'),
                                median_rt  = ('rt_ms', 'median'),
                                mean_rt    = ('rt_ms', 'mean'),
                            )
                            .reset_index()
                        )
                        _verif_grp['triplet_median_count'] = med_c_18
                        _verif_grp['source'] = 'python'

                        _verif_path = os.path.join(
                            _verif_dir_18, f'{sid18}_verification_python.csv')
                        _verif_grp.to_csv(_verif_path, index=False)
                        print(f"  ✓ 驗證 CSV (Python): {_verif_path}")

                    except Exception as _e18:
                        print(f"  ✗ {sid18} 失敗: {_e18}")
                        import traceback; traceback.print_exc()

                if not all_beh_18:
                    print("  ✗ 所有受試者行為資料處理失敗，取消")
                    continue

                beh_all_18 = pd.concat(all_beh_18, ignore_index=True)
                beh_sum_18 = _agg_behavior_18(beh_all_18, use_med=True)

                # ── Step 4: ERSP 資料 ────────────────────────────────────────
                print("\n" + "─"*60)
                print("Step 4：讀取 ERSP 資料")
                print("─"*60)

                ersp_18 = _load_ersp_18(h5_dir_18, subject_ids_18, lock_type_18)
                if len(ersp_18) == 0:
                    print(f"  ✗ 找不到任何 {lock_type_18} ERSP .h5，請先執行選項 15 或 16")
                    continue

                # ── Step 5: Join ─────────────────────────────────────────────
                print("\n" + "─"*60)
                print("Step 5：Join EEG 與行為資料")
                print("─"*60)

                joined_18 = _join_18(ersp_18, beh_sum_18)
                if len(joined_18) == 0:
                    print("  ✗ Join 結果為空，請確認受試者 ID 和 block_group 是否一致")
                    continue

                # 儲存 joined CSV
                os.makedirs(out_dir_18, exist_ok=True)
                joined_csv_18 = os.path.join(out_dir_18, f'joined_eeg_behavior_{lock_type_18}.csv')
                joined_18.to_csv(joined_csv_18, index=False)
                print(f"  ✓ Joined data 儲存: {joined_csv_18}")

                # ── Step 6: 三個方向的分析 ──────────────────────────────────
                print("\n" + "─"*60)
                print("Step 6：執行三個方向的分析")
                print("─"*60)

                dir1_out = os.path.join(out_dir_18, 'direction1_correlation')
                dir2_out = os.path.join(out_dir_18, 'direction2_learning_curve')
                dir3_out = os.path.join(out_dir_18, 'direction3_multilevel')
                for _d18 in [dir1_out, dir2_out, dir3_out]:
                    os.makedirs(_d18, exist_ok=True)

                corr_rows_18 = []
                plt.ioff()

                for roi18 in rois_18:
                    for phase18 in phases_18:
                        sub18 = joined_18[(joined_18['roi']==roi18) & (joined_18['phase']==phase18)]
                        if len(sub18) == 0:
                            continue
                        tag18 = f'{roi18}_{phase18}_{lock_type_18}'

                        # 方向一
                        _, c1 = _plot_dir1_18(joined_18, roi18, phase18, corr_method_18,
                                              os.path.join(dir1_out, f'dir1_{tag18}.png'))
                        if c1:
                            corr_rows_18.append({'dir':1,'roi':roi18,'phase':phase18,
                                                 'lock_type':lock_type_18, **c1})

                        # 方向二
                        _plot_dir2_18(joined_18, roi18, phase18, group_level_18,
                                      os.path.join(dir2_out,
                                                   f'dir2_{"group" if group_level_18 else "individual"}_{tag18}.png'))

                        # 方向三
                        _, c3 = _plot_dir3_18(joined_18, roi18, phase18, dir3_level_18,
                                              corr_method_18,
                                              os.path.join(dir3_out, f'dir3_{dir3_level_18}_{tag18}.png'))
                        if c3:
                            corr_rows_18.append({'dir':3,'roi':roi18,'phase':phase18,
                                                 'lock_type':lock_type_18,
                                                 'level':dir3_level_18, **c3})

                plt.ion()

                # ── Step 7: 輸出摘要 CSV ─────────────────────────────────────
                if corr_rows_18:
                    corr_df_18 = pd.DataFrame(corr_rows_18)
                    corr_out_18 = os.path.join(out_dir_18,
                                               f'correlation_summary_{lock_type_18}.csv')
                    corr_df_18.to_csv(corr_out_18, index=False)
                    print(f"\n  ✓ 相關摘要 CSV: {corr_out_18}")
                    print(f"\n{'─'*60}")
                    print("相關分析摘要（p < .05 標記 *）")
                    print(f"{'─'*60}")
                    for _, rw in corr_df_18.iterrows():
                        sig = ' *' if rw['p'] < .05 else ''
                        phase_lbl = rw.get('phase','')
                        print(f"  Dir{int(rw['dir'])} | {rw['roi']:25s} | {phase_lbl:15s} | "
                              f"r={rw['r']:+.3f}  p={rw['p']:.3f}{sig}")

                print(f"\n{'='*60}")
                print("✓ EEG-行為整合分析完成")
                print(f"  輸出目錄: {out_dir_18}")
                print(f"{'='*60}")
                processing_history.append(f"EEG-行為整合分析（{len(subject_ids_18)} 位受試者，{lock_type_18}）")

            except Exception as e:
                print(f"\n✗ EEG-行為整合分析時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()

        elif choice == '16':
            # Response ERSP 分析（per-trial logratio baseline）

            print("\n" + "="*70)
            print("Response ERSP 分析（per-trial logratio baseline）")
            print("="*70)
            print("\n方法:")
            print("  1. 逐 trial Morlet wavelet（average=False）")
            print("  2. 各 trial 以自己的 Blank 靜息期做 logratio baseline")
            print("  3. 平均所有校正後 trial")
            
            # ===== 檢查前置條件 =====
            if current_epochs is None:
                print("\n⚠️  請先建立 Response-locked Epochs（選項 8）")
                continue

            # ===== 確認執行 =====
            confirm = input("\n確定執行 Response ERSP 分析？(y/n): ").strip().lower()
            if confirm != 'y':
                print("取消分析")
                continue
            
            try:
                # 導入函數
                from asrt_response_ersp_from_epochs import response_ersp_from_current_epochs
                
                # ===== 設定參數 =====
                print("\n" + "-"*70)
                print("參數設定")
                print("-"*70)
                
                # 頻率範圍
                print("\n頻率範圍:")
                print("  預設: 4-40 Hz")
                freq_input = input("使用預設？(y/n) [y]: ").strip().lower() or 'y'
                
                if freq_input == 'y':
                    freqs = np.arange(4, 41, 1)  # 修正：改成 41 才會包含 40
                else:
                    freq_min = float(input("  最小頻率 (Hz): ").strip())
                    freq_max = float(input("  最大頻率 (Hz): ").strip())
                    freq_step = float(input("  頻率間隔 (Hz): ").strip())
                    freqs = np.arange(freq_min, freq_max + freq_step, freq_step)
                
                print(f"  → 使用 {freqs[0]}-{freqs[-1]} Hz")
                
                # n_cycles
                print("\nn_cycles 設定:")
                print("  預設: freqs / 2.0")
                n_cycles_input = input("使用預設？(y/n) [y]: ").strip().lower() or 'y'
                
                if n_cycles_input == 'y':
                    n_cycles_func = None
                else:
                    # 這裡可以擴展其他 n_cycles 設定
                    n_cycles_func = None
                
                # 降採樣
                print("\n降採樣因子:")
                print("  預設: 1（不降採樣）")
                decim_input = input("使用預設？(y/n) [y]: ").strip().lower() or 'y'
                
                if decim_input == 'y':
                    decim = 1
                else:
                    decim = int(input("  降採樣因子: ").strip())
                
                print(f"  → 降採樣因子: {decim}")
                
                # Baseline 設定
                print("\nBaseline 設定:")
                print("  1. 只做 FD baseline（推薦）")
                print("  2. TD + FD baseline")
                bl_choice = input("請選擇 (1/2) [1]: ").strip() or '1'
                do_td_baseline = bl_choice == '2'

                print("\nBaseline 方法:")
                print("  1. Per-trial pre-stimulus blank 期（目前做法）")
                print("  2. 整段 epoch 平均（Lum et al. 2023）")
                bl_method_choice = input("請選擇 (1/2) [1]: ").strip() or '1'
                baseline_method = 'whole_epoch' if bl_method_choice == '2' else 'pre_stim'

                # 平行處理
                print("\n平行處理:")
                print("  建議: n_jobs=1（避免 Windows GIL 問題）")
                n_jobs = 1
                
                # 輸出設定
                print("\n輸出設定:")
                save_output = input("是否儲存結果？(y/n) [y]: ").strip().lower() or 'y'
                
                if save_output == 'y':
                    is_triplet = (
                        hasattr(current_epochs, 'metadata') and
                        current_epochs.metadata is not None and
                        'classification' in current_epochs.metadata.columns and
                        current_epochs.metadata['classification'].iloc[0] == 'triplet'
                    )
                    default_base = r'C:\Experiment\Result\triplet' if is_triplet else r'C:\Experiment\Result'
                    _base_dir = input(f"輸出資料夾路徑 [{default_base}]: ").strip() or default_base
                    output_dir = os.path.join(_base_dir, 'h5')
                    plot_dir_response = _base_dir

                    # 受試者 ID
                    subject_id = 'sub'
                    subject_id = input(f"受試者 ID [{subject_id}]: ").strip() or subject_id
                else:
                    output_dir = None
                    subject_id = None
                
                print("\n" + "-"*70)
                print("開始分析...")
                print("-"*70)

                # ===== 執行分析 =====
                power_response = response_ersp_from_current_epochs(
                    response_epochs=current_epochs,
                    freqs=freqs,
                    n_cycles_func=n_cycles_func,
                    decim=decim,
                    n_jobs=n_jobs,
                    output_dir=output_dir,
                    subject_id=subject_id,
                    do_td_baseline=do_td_baseline,
                    baseline_method=baseline_method,
                )
                
                print("\n✓ Response ERSP 分析完成！")
                processing_history.append(
                    "Response ERSP 分析 (Stimulus baseline → Response 對齊 → 整段平均)"
                )

                # ── 單一電極個人圖 ──
                if output_dir and subject_id:
                    print("\n" + "─"*60)
                    elec_choice16 = input("是否產生單一電極 ERSP 比較圖？(y/n) [n]: ").strip().lower() or 'n'
                    if elec_choice16 == 'y':
                        elec_input16 = input("請輸入電極名稱（逗號分隔，例如: Fz,Cz,Pz）: ").strip()
                        if elec_input16:
                            electrodes16 = [e.strip() for e in elec_input16.split(',') if e.strip()]

                            is_triplet_mode = (
                                hasattr(current_epochs, 'metadata') and
                                current_epochs.metadata is not None and
                                'classification' in current_epochs.metadata.columns and
                                current_epochs.metadata['classification'].iloc[0] == 'triplet'
                            )
                            lbl_left16  = 'high' if is_triplet_mode else 'Regular'
                            lbl_right16 = 'low'  if is_triplet_mode else 'Random'

                            elec_out16 = r'C:\Experiment\Result\single_electrode'
                            os.makedirs(elec_out16, exist_ok=True)

                            import glob as _glob16
                            import numpy as _np16
                            import matplotlib.pyplot as _plt16
                            from mne_python_analysis.group_ersp_analysis import _load_h5_single_electrode

                            _h5_search_dir = r'C:\Experiment\Result\h5'
                            h5_files_reg = sorted(_glob16.glob(
                                os.path.join(r'C:\Experiment\Result\h5', f'{subject_id}_Response_*_Regular_ERSP.h5')))

                            for electrode16 in electrodes16:
                                print(f"\n  電極: {electrode16}")
                                for fp_l in h5_files_reg:
                                    fp_r = fp_l.replace('_Regular_', '_Random_')
                                    if not os.path.exists(fp_r):
                                        continue
                                    try:
                                        el, freqs_e, times_e = _load_h5_single_electrode(fp_l, electrode16)
                                        er, _, _             = _load_h5_single_electrode(fp_r, electrode16)
                                    except Exception as ex:
                                        print(f"    ⚠ {os.path.basename(fp_l)}: {ex}")
                                        continue

                                    diff_e = el - er
                                    x_min_e, x_max_e = -0.5, 0.5
                                    t_mask_e = (times_e >= x_min_e) & (times_e <= x_max_e)
                                    combined_e = _np16.concatenate([el[:, t_mask_e].ravel(), er[:, t_mask_e].ravel()])
                                    vmax_e = _np16.percentile(_np16.abs(combined_e), 95)
                                    vmax_d = _np16.percentile(_np16.abs(diff_e[:, t_mask_e].ravel()), 95)
                                    lv_c = _np16.linspace(-vmax_e, vmax_e, 20)
                                    lv_d = _np16.linspace(-vmax_d, vmax_d, 20)

                                    base = os.path.basename(fp_l).replace('_ERSP.h5', '').replace(f'{subject_id}_Response_', '')
                                    block_label16 = '_'.join(base.split('_')[:-1])

                                    fig16, axes16 = _plt16.subplots(1, 3, figsize=(18, 5))
                                    for ax16, data16, title16, lv16, vm16, cbl16 in [
                                        (axes16[0], el,     f'{lbl_left16}',                        lv_c, vmax_e, 'Power (dB)'),
                                        (axes16[1], er,     f'{lbl_right16}',                       lv_c, vmax_e, 'Power (dB)'),
                                        (axes16[2], diff_e, f'Difference ({lbl_left16} - {lbl_right16})', lv_d, vmax_d, 'Power Difference (dB)'),
                                    ]:
                                        im16 = ax16.contourf(times_e, freqs_e, data16, levels=lv16,
                                                             cmap='RdBu_r', vmin=-vm16, vmax=vm16, extend='both')
                                        ax16.axvline(0, color='black', linestyle='--', linewidth=1.5)
                                        ax16.axhline(8,  color='white', linestyle=':', linewidth=1, alpha=0.6)
                                        ax16.axhline(13, color='white', linestyle=':', linewidth=1, alpha=0.6)
                                        ax16.set_xlabel('Time (s)', fontsize=11)
                                        ax16.set_ylabel('Frequency (Hz)', fontsize=11)
                                        ax16.set_title(title16, fontsize=11, fontweight='bold')
                                        ax16.set_xlim([x_min_e, x_max_e])
                                        _plt16.colorbar(im16, ax=ax16, label=cbl16)

                                    fig16.suptitle(
                                        f'{subject_id} | Response-locked | {block_label16} | Electrode: {electrode16}',
                                        fontsize=12, fontweight='bold'
                                    )
                                    _plt16.tight_layout()
                                    out_fig16 = os.path.join(elec_out16,
                                        f'{subject_id}_response_{electrode16}_{block_label16}_comparison.png')
                                    fig16.savefig(out_fig16, dpi=300, bbox_inches='tight')
                                    _plt16.close(fig16)
                                    print(f"    ✓ 已儲存: {out_fig16}")

                            processing_history.append(f"單一電極個人圖（{elec_input16}）")
                
            except ImportError as e:
                print(f"\n✗ 找不到必要的模組: {e}")
                print("  請確認 asrt_response_ersp_from_epochs.py 在正確位置")
            except Exception as e:
                print(f"\n✗ 分析時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()
        
        elif choice == '21':
            # 輸出 RT 資料到 CSV
            print("\n" + "="*60)
            print("輸出 RT 資料到 CSV")
            print("="*60)

            print("\n分類方式:")
            print("  1. Trigger 分類（Regular / Random）")
            print("  2. Triplet 分類（high / low）")
            cls_choice = input("請選擇 (1/2) [1]: ").strip() or '1'
            is_triplet = (cls_choice == '2')
            test_type = 'triplet' if is_triplet else 'trigger'

            import re as _re_rt
            _base_sid = _re_rt.sub(
                r'[_-](resp|stim|ASRT)[_-].*$', '', subject_id, flags=_re_rt.IGNORECASE
            ).rstrip('_-') or subject_id

            def _load_epoch_interactive(search_patterns, label, exclude_trip=False):
                """搜尋並讓使用者確認載入 epoch 檔案，回傳 Epochs 物件或 None"""
                found = []
                for pat in search_patterns:
                    candidates = glob_module.glob(pat)
                    if exclude_trip:
                        candidates = [f for f in candidates if 'trip' not in os.path.basename(f).lower()]
                    if candidates:
                        found = candidates
                        break
                if found:
                    if len(found) > 1:
                        print(f"\n找到多個 {label} 檔案，請選擇：")
                        for i, f in enumerate(found):
                            print(f"  {i+1}. {f}")
                        sel = input("請輸入編號 [1]: ").strip()
                        try:
                            path = found[int(sel) - 1] if sel else found[0]
                        except (ValueError, IndexError):
                            path = found[0]
                    else:
                        path = found[0]
                        print(f"\n自動找到 {label}: {path}")
                    print("載入中...")
                    epo = mne.read_epochs(path, preload=False)
                    print(f"✓ 載入完成：{len(epo)} 個 epochs")
                    return epo
                else:
                    print(f"\n⚠️  找不到 {label}（搜尋模式: {search_patterns[:2]}...）")
                    manual = input(f"請手動輸入路徑（Enter 跳過）: ").strip().strip('"')
                    if manual:
                        if not os.path.exists(manual):
                            print(f"❌ 找不到檔案: {manual}")
                            return None
                        epo = mne.read_epochs(manual, preload=False)
                        print(f"✓ 載入完成：{len(epo)} 個 epochs")
                        return epo
                    return None

            if is_triplet:
                # === Triplet 分類：直接從 metadata join，不做時間 matching ===

                default_sid = subject_id if subject_id else 'sub'
                sid = input(f"\n受試者 ID [{default_sid}]: ").strip() or default_sid

                rt_epochs = _load_epoch_interactive([
                    f"*{_base_sid}*resp*triplet*-epo.fif",
                    f"*{_base_sid}*resp*trip*-epo.fif",
                    f"*resp*triplet*-epo.fif",
                    f"*resp*trip*-epo.fif",
                ], label="Triplet resp Epochs")

                if rt_epochs is None:
                    if current_epochs is None:
                        print("\n⚠️  沒有可用的 Epochs，無法繼續")
                        continue
                    rt_epochs = current_epochs
                    print("⚠️  使用已載入的 epochs")

                if not hasattr(rt_epochs, 'metadata') or rt_epochs.metadata is None:
                    print("\n⚠️  Epochs 沒有 metadata")
                    continue

                meta = rt_epochs.metadata
                required_cols = {'block', 'trial_in_block', 'stim_sample', 'resp_sample', 'trial_type'}
                missing = required_cols - set(meta.columns)
                if missing:
                    print(f"\n⚠️  metadata 缺少欄位: {missing}")
                    continue

                sfreq = rt_epochs.info['sfreq']

                # ── EEG lookup {(block, trial_in_block): dict} ───────────────
                eeg_lookup = {}
                for _, row in meta.iterrows():
                    blk = int(row['block'])
                    tib = int(row['trial_in_block'])
                    if tib == -1:
                        continue
                    eeg_lookup[(blk, tib)] = {
                        'trial_type':       row['trial_type'],
                        'eeg_stim_time_s':  float(row['stim_sample']) / sfreq,
                        'eeg_resp_time_s':  float(row['resp_sample']) / sfreq,
                        'rt_eeg_ms':        (float(row['resp_sample']) - float(row['stim_sample'])) / sfreq * 1000,
                    }
                print(f"  EEG lookup：{len(eeg_lookup)} trials")

                # ── 載入行為資料 CSV ─────────────────────────────────────────
                behav_path = input("請輸入行為資料 CSV 路徑: ").strip().strip('"')
                if not os.path.exists(behav_path):
                    print(f"❌ 找不到檔案: {behav_path}")
                    continue
                try:
                    behav_raw = pd.read_csv(behav_path)
                except Exception as _e:
                    print(f"❌ 讀取失敗: {_e}")
                    continue

                # ── 建立行為資料表（只取 triplet epoch 有的 block 範圍）───────
                _learn_trial_col = 'learning_loop.thisTrialN'
                _learn_block_col = 'learning_trials.thisTrialN'
                _test_block_col  = 'combined_testing_trials.thisTrialN'
                _test_loop_cols  = ['percept_motor_testing_loop.thisTrialN',
                                    'motor_percept_testing_loop.thisTrialN']
                _test_trial_col  = next((c for c in _test_loop_cols if c in behav_raw.columns), None)

                behav_rows = []
                if _learn_trial_col in behav_raw.columns and _learn_block_col in behav_raw.columns:
                    lmask = behav_raw[_learn_trial_col].notna() & behav_raw[_learn_block_col].notna()
                    for _, r in behav_raw[lmask].iterrows():
                        blk     = int(float(r[_learn_block_col])) + 7
                        tib     = int(float(r[_learn_trial_col]))
                        rt_b    = pd.to_numeric(r.get('key_resp.rt'),       errors='coerce')
                        arrow_s = pd.to_numeric(r.get('arrowhead.started'), errors='coerce')
                        behav_rows.append({
                            'phase': 'Learning', 'block': blk, 'trial_in_block': tib,
                            'beh_stim_time_s':   float(arrow_s)        if pd.notna(arrow_s)                    else None,
                            'beh_resp_time_s':   float(arrow_s + rt_b) if pd.notna(arrow_s) and pd.notna(rt_b) else None,
                            'rt_behavioral_ms':  float(rt_b) * 1000    if pd.notna(rt_b)                        else None,
                        })

                if _test_block_col in behav_raw.columns and _test_trial_col is not None:
                    tmask = behav_raw[_test_block_col].notna() & behav_raw[_test_trial_col].notna()
                    for _, r in behav_raw[tmask].iterrows():
                        blk     = int(float(r[_test_block_col])) + 27
                        tib     = int(float(r[_test_trial_col]))
                        rt_b    = pd.to_numeric(r.get('key_resp.rt'),       errors='coerce')
                        arrow_s = pd.to_numeric(r.get('arrowhead.started'), errors='coerce')
                        behav_rows.append({
                            'phase': 'Test', 'block': blk, 'trial_in_block': tib,
                            'beh_stim_time_s':   float(arrow_s)        if pd.notna(arrow_s)                    else None,
                            'beh_resp_time_s':   float(arrow_s + rt_b) if pd.notna(arrow_s) and pd.notna(rt_b) else None,
                            'rt_behavioral_ms':  float(rt_b) * 1000    if pd.notna(rt_b)                        else None,
                        })

                behav_df = pd.DataFrame(behav_rows).sort_values(['block', 'trial_in_block']).reset_index(drop=True)

                # ── 合併（以 EEG triplet epoch 為主軸，triplet epoch 才有 trial_type）──
                result_rows = []
                for (blk, tib), eeg in eeg_lookup.items():
                    bkey = behav_df[(behav_df['block'] == blk) & (behav_df['trial_in_block'] == tib)]
                    brow = bkey.iloc[0] if len(bkey) > 0 else None
                    result_rows.append({
                        'sid':              sid,
                        'phase':            'Learning' if blk <= 26 else 'Test',
                        'block':            blk,
                        'trial_in_block':   tib,
                        'trial_type':       eeg['trial_type'],
                        'eeg_stim_time_s':  eeg['eeg_stim_time_s'],
                        'eeg_resp_time_s':  eeg['eeg_resp_time_s'],
                        'rt_eeg_ms':        eeg['rt_eeg_ms'],
                        'beh_stim_time_s':  float(brow['beh_stim_time_s'])  if brow is not None and brow['beh_stim_time_s']  is not None else None,
                        'beh_resp_time_s':  float(brow['beh_resp_time_s'])  if brow is not None and brow['beh_resp_time_s']  is not None else None,
                        'rt_behavioral_ms': float(brow['rt_behavioral_ms']) if brow is not None and brow['rt_behavioral_ms'] is not None else None,
                    })

                df = pd.DataFrame(result_rows)[[
                    'sid', 'phase', 'block', 'trial_in_block', 'trial_type',
                    'eeg_stim_time_s', 'eeg_resp_time_s', 'rt_eeg_ms',
                    'beh_stim_time_s', 'beh_resp_time_s', 'rt_behavioral_ms'
                ]].sort_values(['block', 'trial_in_block']).reset_index(drop=True)

                n_both   = df['rt_eeg_ms'].notna().sum()
                n_miss_b = df['rt_behavioral_ms'].isna().sum()

                print(f"\n  合計：{len(df)} trials")
                print(f"  EEG + 行為都有：{n_both}")
                print(f"  Miss trials（行為無作答）：{n_miss_b}")

                out_dir  = r'C:\Experiment\ersp_csv\triplet\rt_data'
                out_file = os.path.join(out_dir, f'{sid}_rt_triplet.csv')
                os.makedirs(out_dir, exist_ok=True)
                df.to_csv(out_file, index=False)
                print(f"\n✓ 已輸出 {len(df)} 筆")
                print(f"  → {out_file}")
                processing_history.append(f"輸出 RT 驗證 CSV（triplet，{sid}）")

            else:
                # === Trigger 分類：直接從 metadata + 行為資料 join，不做時間 matching ===

                default_sid = subject_id if subject_id else 'sub'
                sid = input(f"\n受試者 ID [{default_sid}]: ").strip() or default_sid

                behav_path = input("請輸入行為資料 CSV 路徑: ").strip().strip('"')
                if not os.path.exists(behav_path):
                    print(f"❌ 找不到檔案: {behav_path}")
                    continue
                try:
                    behav_raw = pd.read_csv(behav_path)
                except Exception as _e:
                    print(f"❌ 讀取失敗: {_e}")
                    continue

                # ── 建立行為資料表 ────────────────────────────────────────────
                _learn_trial_col = 'learning_loop.thisTrialN'    # trial in block (0-84)
                _learn_block_col = 'learning_trials.thisTrialN'  # block index (0-19) → +7
                _test_block_col  = 'combined_testing_trials.thisTrialN'  # (0-7) → +27
                _test_loop_cols  = ['percept_motor_testing_loop.thisTrialN',
                                    'motor_percept_testing_loop.thisTrialN']
                _test_trial_col  = next((c for c in _test_loop_cols if c in behav_raw.columns), None)

                behav_rows = []

                if _learn_trial_col in behav_raw.columns and _learn_block_col in behav_raw.columns:
                    lmask = behav_raw[_learn_trial_col].notna() & behav_raw[_learn_block_col].notna()
                    for _, r in behav_raw[lmask].iterrows():
                        blk     = int(float(r[_learn_block_col])) + 7
                        tib     = int(float(r[_learn_trial_col]))
                        rt_b    = pd.to_numeric(r.get('key_resp.rt'),       errors='coerce')
                        arrow_s = pd.to_numeric(r.get('arrowhead.started'), errors='coerce')
                        color   = r.get('arrow_color', '')
                        behav_rows.append({
                            'phase': 'Learning', 'block': blk, 'trial_in_block': tib,
                            'trial_type':        'Regular' if color == 'white' else ('Random' if color == 'red' else None),
                            'beh_stim_time_s':   float(arrow_s)        if pd.notna(arrow_s)                    else None,
                            'beh_resp_time_s':   float(arrow_s + rt_b) if pd.notna(arrow_s) and pd.notna(rt_b) else None,
                            'rt_behavioral_ms':  float(rt_b) * 1000    if pd.notna(rt_b)                        else None,
                        })

                if _test_block_col in behav_raw.columns and _test_trial_col is not None:
                    tmask = behav_raw[_test_block_col].notna() & behav_raw[_test_trial_col].notna()
                    for _, r in behav_raw[tmask].iterrows():
                        blk     = int(float(r[_test_block_col])) + 27
                        tib     = int(float(r[_test_trial_col]))
                        rt_b    = pd.to_numeric(r.get('key_resp.rt'),       errors='coerce')
                        arrow_s = pd.to_numeric(r.get('arrowhead.started'), errors='coerce')
                        color   = r.get('arrow_color', '')
                        behav_rows.append({
                            'phase': 'Test', 'block': blk, 'trial_in_block': tib,
                            'trial_type':        'Regular' if color == 'white' else ('Random' if color == 'red' else None),
                            'beh_stim_time_s':   float(arrow_s)        if pd.notna(arrow_s)                    else None,
                            'beh_resp_time_s':   float(arrow_s + rt_b) if pd.notna(arrow_s) and pd.notna(rt_b) else None,
                            'rt_behavioral_ms':  float(rt_b) * 1000    if pd.notna(rt_b)                        else None,
                        })

                if not behav_rows:
                    print("❌ 行為資料解析失敗，找不到 learning_loop / testing_loop 欄位")
                    continue

                behav_df = (pd.DataFrame(behav_rows)
                              .sort_values(['block', 'trial_in_block'])
                              .reset_index(drop=True))
                print(f"\n  行為資料：{len(behav_df)} trials"
                      f"（Learning: {(behav_df['phase']=='Learning').sum()}，"
                      f"Test: {(behav_df['phase']=='Test').sum()}）")
                print(f"  Miss trials（無作答）：{behav_df['rt_behavioral_ms'].isna().sum()}")

                # ── 載入 Response epoch ───────────────────────────────────────
                resp_epo = _load_epoch_interactive([
                    f"*{_base_sid}*ASRT*resp*all-epo.fif",
                    f"*{_base_sid}*ASRT*resp*learn-epo.fif",
                    f"*{_base_sid}*ASRT*resp*test-epo.fif",
                    f"*{_base_sid}*resp*all-epo.fif",
                    f"*{_base_sid}*resp*learn-epo.fif",
                    f"*{_base_sid}*resp*-epo.fif",
                    f"*ASRT*resp*all-epo.fif",
                    f"*resp*-epo.fif",
                ], label="Response Epochs (trigger)", exclude_trip=True)

                if resp_epo is None:
                    print("❌ 無法載入 Response epochs")
                    continue

                resp_meta = resp_epo.metadata.copy()
                _req  = {'block', 'trial_in_block', 'stim_sample', 'resp_sample'}
                _miss = _req - set(resp_meta.columns)
                if _miss:
                    print(f"❌ metadata 缺少欄位：{_miss}")
                    continue

                sfreq = resp_epo.info['sfreq']
                n_invalid = (resp_meta['trial_in_block'] == -1).sum()
                if n_invalid:
                    print(f"  ⚠️  {n_invalid} 筆 trial_in_block = -1（stim_sample 查無對應），略過")

                # ── 建立 EEG lookup {(block, trial_in_block): dict} ──────────
                eeg_lookup = {}
                for _, row in resp_meta.iterrows():
                    blk = int(row['block'])
                    tib = int(row['trial_in_block'])
                    if tib == -1:
                        continue
                    eeg_lookup[(blk, tib)] = {
                        'eeg_stim_time_s': float(row['stim_sample']) / sfreq,
                        'eeg_resp_time_s': float(row['resp_sample']) / sfreq,
                        'rt_eeg_ms':       (float(row['resp_sample']) - float(row['stim_sample'])) / sfreq * 1000,
                    }
                print(f"  EEG lookup：{len(eeg_lookup)} trials")

                # ── 補齊 miss trial 的 eeg_stim_time_s（從 Stimulus epoch 取）──
                stim_epo = _load_epoch_interactive([
                    f"*{_base_sid}*ASRT*stim*all-epo.fif",
                    f"*{_base_sid}*ASRT*stim*learn-epo.fif",
                    f"*{_base_sid}*stim*all-epo.fif",
                    f"*{_base_sid}*stim*-epo.fif",
                    f"*ASRT*stim*all-epo.fif",
                ], label="Stimulus Epochs (補 miss trial stim time)", exclude_trip=True)

                eeg_stim_lookup = {}  # {(block, trial_in_block): stim_time_s}
                if stim_epo is not None:
                    stim_meta = stim_epo.metadata.copy()
                    stim_events = stim_epo.events[:, 0]  # stim sample = epoch event sample
                    stim_sfreq = stim_epo.info['sfreq']

                    # trial_in_block = 在同一 block 內的出現順序（0-based）
                    stim_meta = stim_meta.copy()
                    stim_meta['_stim_sample'] = stim_events
                    stim_meta['_tib'] = (stim_meta
                                         .groupby('block', sort=False)
                                         .cumcount())

                    for _, row in stim_meta.iterrows():
                        key = (int(row['block']), int(row['_tib']))
                        eeg_stim_lookup[key] = float(row['_stim_sample']) / stim_sfreq

                    print(f"  Stimulus lookup：{len(eeg_stim_lookup)} trials")
                else:
                    print("  ⚠️  未載入 Stimulus epochs，miss trial 的 eeg_stim_time_s 將為 None")

                # ── 合併（以行為資料為主軸）──────────────────────────────────
                result_rows = []
                for _, brow in behav_df.iterrows():
                    key = (int(brow['block']), int(brow['trial_in_block']))
                    eeg = eeg_lookup.get(key)
                    # miss trial 優先從 stim_lookup 取，有 resp 的從 eeg_lookup 取
                    _eeg_stim = eeg['eeg_stim_time_s'] if eeg else eeg_stim_lookup.get(key)
                    result_rows.append({
                        'sid':              sid,
                        'phase':            brow['phase'],
                        'block':            brow['block'],
                        'trial_in_block':   brow['trial_in_block'],
                        'trial_type':       brow['trial_type'],
                        'eeg_stim_time_s':  _eeg_stim,
                        'eeg_resp_time_s':  eeg['eeg_resp_time_s'] if eeg else None,
                        'rt_eeg_ms':        eeg['rt_eeg_ms']        if eeg else None,
                        'beh_stim_time_s':  brow['beh_stim_time_s'],
                        'beh_resp_time_s':  brow['beh_resp_time_s'],
                        'rt_behavioral_ms': brow['rt_behavioral_ms'],
                    })

                df = pd.DataFrame(result_rows)[[
                    'sid', 'phase', 'block', 'trial_in_block', 'trial_type',
                    'eeg_stim_time_s', 'eeg_resp_time_s', 'rt_eeg_ms',
                    'beh_stim_time_s', 'beh_resp_time_s', 'rt_behavioral_ms'
                ]]

                n_both   = df['rt_eeg_ms'].notna().sum()
                n_miss_b = df['rt_behavioral_ms'].isna().sum()
                n_no_eeg = (df['rt_eeg_ms'].isna() & df['rt_behavioral_ms'].notna()).sum()

                print(f"\n  合計：{len(df)} trials")
                print(f"  EEG + 行為都有：{n_both}")
                print(f"  Miss trials（行為無作答）：{n_miss_b}")
                print(f"  行為有作答但 EEG 無配對（artifact rejected 或 trial_in_block 錯誤）：{n_no_eeg}")

                out_dir  = r'C:\Experiment\ersp_csv\trigger\rt_data'
                out_file = os.path.join(out_dir, f'{sid}_rt_trigger.csv')
                os.makedirs(out_dir, exist_ok=True)
                df.to_csv(out_file, index=False)
                print(f"\n✓ 已輸出 {len(df)} 筆")
                print(f"  → {out_file}")
                processing_history.append(f"輸出 RT 驗證 CSV（trigger，{sid}）")

        # === 儲存與退出 (選項 12-14, 0) ===
        # 來源：原 main.py 2426-2465 行
        elif choice == '12':
            save_raw_interactive(current_raw, subject_id)

        elif choice == '13':
            save_epochs_interactive(current_epochs, subject_id)

        elif choice == '14':
            display_processing_history(processing_history)
            
            if asrt_results:
                print("\nASRT 分析結果摘要:")
                for analysis_type in asrt_results.keys():
                    print(f"  - {analysis_type}")
        
        elif choice == '0':
            confirm = input("\n確定要返回主選單嗎？(y/n): ").strip().lower()
            if confirm == 'y':
                print("✓ 返回主選單")
                break


# ============================================================
# 主程式入口 (簡化版)
# 來源：原 main.py 2468-2514 行
# 修改：使用 ui.menu.display_welcome_message()
# ============================================================

def main():
    """主程式入口點"""
    set_matplotlib_properties()
    display_welcome_message(ASRT_MODULES_AVAILABLE)
    
    while True:
        try:
            data_path, file_format = select_data_source()
            if data_path is None:
                print("\n程式結束")
                break
            if not os.path.exists(data_path):
                print(f"⚠️  路徑不存在: {data_path}")
                continue
            if file_format == 'BIDS':
                process_bids_data(Path(data_path))
            else:
                process_single_file(data_path, file_format)
        except KeyboardInterrupt:
            print("\n\n使用者中斷程式")
            break
        except Exception as e:
            print(f"\n⚠️  發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            retry = input("\n是否繼續? (y/n): ").strip().lower()
            if retry != 'y':
                break
    print("\n感謝使用 EEG 資料分析系統!")


if __name__ == "__main__":
    main()
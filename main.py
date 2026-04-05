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
        process_eeg_data(subject_id, {'raw': data_obj, 'events': events}, file_path)



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

def process_eeg_data(subject_id, subject_data, data_path=None):
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
                epochs_result = create_epochs_interactive(current_raw, subject_id)
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

                    result = asrt_ersp_full_analysis(
                        stim_epochs,
                        subject_id=analysis_sid,
                        phase=phase,
                        lock_type='stimulus',
                        save_for_group_analysis=save_for_group,
                        group_data_dir=r'C:\Experiment\Result\h5',
                        do_td_baseline=do_td_baseline,
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

            except Exception as e:
                print(f"\nERSP 分析時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()
        
        elif choice == '18':
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

        elif choice == '19':
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

                H5_DIR     = r'C:\Experiment\Result\h5'
                OUTPUT_DIR = r'C:\Experiment\Result\group_ersp'

                print("\n" + "="*60)
                print("ASRT 群體分析 (Group-level ERSP)")
                print("="*60)
                print("\n自動執行全部組合：")
                print("  階段  : Learning + Testing")
                print("  測試  : Motor + Perceptual  (Testing 階段)")
                print("  鎖定  : Stimulus-locked + Response-locked")
                print("  ROI   : Theta + Alpha")
                print("  比較  : Regular vs Random")
                print(f"\n資料來源  : {H5_DIR}")
                print(f"輸出根目錄: {OUTPUT_DIR}")
                print("\n⚠  資料前置需求：")
                print("   Stimulus-locked → 先執行選項 15，分析中選「儲存供群體分析用：y」")
                print("   Response-locked → 先執行選項 17")

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

                results = auto_group_ersp_analysis(
                    subject_ids         = subject_ids,
                    data_dir            = H5_DIR,
                    output_dir          = OUTPUT_DIR,
                    do_permutation_test = do_permutation,
                    n_permutations      = 1000,
                )

                n_done = sum(1 for v in results.values() if v)
                processing_history.append(
                    f"ASRT 群體分析（全自動，{len(subject_ids)} 位受試者，{n_done} 個組合完成）"
                )

            except ImportError:
                print("❌ 群體分析模組未安裝")
                print("請確認 group_ersp_analysis.py 已放在 mne_python_analysis/ 目錄下")
            except Exception as e:
                print(f"❌ 群體分析時發生錯誤: {str(e)}")
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

                # 平行處理
                print("\n平行處理:")
                print("  建議: n_jobs=1（避免 Windows GIL 問題）")
                n_jobs = 1
                
                # 輸出設定
                print("\n輸出設定:")
                save_output = input("是否儲存結果？(y/n) [y]: ").strip().lower() or 'y'
                
                if save_output == 'y':
                    _base_dir = input(r"輸出資料夾路徑 [C:\Experiment\Result]: ").strip() or r'C:\Experiment\Result'
                    output_dir = os.path.join(_base_dir, 'h5')

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
                )
                
                # ===== 繪圖 =====
                print("\n" + "-"*70)
                print("繪製結果")
                print("-"*70)
                
                plot_result = input("\n是否繪製 ERSP？(y/n) [y]: ").strip().lower() or 'y'
                
                if plot_result == 'y':
                    import matplotlib.pyplot as plt
                    
                    # 選擇要繪製的 channels
                    print("\n選擇要繪製的 channels:")
                    print("  1. Fz, Cz, Pz")
                    print("  2. 自訂")
                    
                    ch_choice = input("請選擇 (1/2) [1]: ").strip() or '1'
                    
                    if ch_choice == '1':
                        picks = ['Fz', 'Cz', 'Pz']
                    else:
                        ch_input = input("輸入 channel 名稱（逗號分隔）: ").strip()
                        picks = [ch.strip() for ch in ch_input.split(',')]
                    
                    # 繪圖

                    # 統一成 {cond_name: power} 的 dict，方便後續處理
                    if isinstance(power_response, dict):
                        power_dict = power_response
                    else:
                        power_dict = {'ERSP': power_response}

                    for cond_name, power_obj in power_dict.items():
                        figs = power_obj.plot(
                            picks=picks,
                            baseline=None,
                            mode=None,
                            title=f'Response-locked ERSP – {cond_name}',
                            show=False
                        )

                        plt.show()

                        # 儲存圖片
                        if output_dir and subject_id:
                            os.makedirs(output_dir, exist_ok=True)

                            # 處理單一或多個 figures（每個 channel 一張）
                            if isinstance(figs, list):
                                for i, fig in enumerate(figs):
                                    ch_name = picks[i] if i < len(picks) else str(i)
                                    fig_file = os.path.join(
                                        output_dir,
                                        f"{subject_id}_response_ersp_{cond_name}_{ch_name}.png"
                                    )
                                    fig.savefig(fig_file, dpi=300, bbox_inches='tight')
                                    print(f"  ✓ 已儲存: {fig_file}")
                            else:
                                fig_file = os.path.join(
                                    output_dir,
                                    f"{subject_id}_response_ersp_{cond_name}.png"
                                )
                                figs.savefig(fig_file, dpi=300, bbox_inches='tight')
                                print(f"  ✓ 已儲存: {fig_file}")
                
                print("\n✓ Response ERSP 分析完成！")
                processing_history.append(
                    "Response ERSP 分析 (Stimulus baseline → Response 對齊 → 整段平均)"
                )
                
            except ImportError as e:
                print(f"\n✗ 找不到必要的模組: {e}")
                print("  請確認 asrt_response_ersp_from_epochs.py 在正確位置")
            except Exception as e:
                print(f"\n✗ 分析時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()
        
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
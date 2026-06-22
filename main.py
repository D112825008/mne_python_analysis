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

import matplotlib
# 選擇 backend：優先互動式（供 ICA 視窗使用），儲存圖片兩種 backend 都支援
import sys
_mpl_interactive = False
for _be in ['TkAgg', 'Qt5Agg', 'WXAgg']:
    try:
        matplotlib.use(_be)
        _mpl_interactive = True
        break
    except Exception:
        continue
if not _mpl_interactive:
    matplotlib.use('Agg')  # 回退：無互動式 backend 可用
import os
import re
import glob as glob_module
from pathlib import Path
import matplotlib.pyplot as plt
plt.switch_backend(matplotlib.get_backend())  # 確認 backend 已套用
matplotlib.rcParams["axes.unicode_minus"] = False  # 用 ASCII hyphen 取代 Unicode minus sign
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
        # ── 自動偵測並載入既有壞通道/BAD 片段標記 ──────────────
        try:
            from mne_python_analysis.preprocessing import load_bad_marking, _bad_marking_path
            _mark_path = _bad_marking_path(subject_id)
            if os.path.exists(_mark_path):
                print(f"\n  偵測到既有標記檔案: {_mark_path}")
                data_obj, _ok = load_bad_marking(data_obj, subject_id)
                if _ok:
                    _n_bad_seg = sum(1 for a in data_obj.annotations
                                     if a['description'].startswith('BAD'))
                    print(f"  ✓ 標記已自動套用")
                    print(f"    壞通道: {data_obj.info['bads']}")
                    print(f"    BAD 片段: {_n_bad_seg} 個")
            else:
                print("  （未找到既有標記檔案，如需標記請使用選項 4）")
        except Exception as _e_mark:
            print(f"  ⚠ 自動載入標記時發生錯誤: {_e_mark}")

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

        # ── 自動偵測並載入既有壞通道/BAD 片段標記 ──────────────
        try:
            from mne_python_analysis.preprocessing import load_bad_marking, _bad_marking_path
            _sid = Path(raw_path).stem
            _mark_path = _bad_marking_path(_sid)
            if os.path.exists(_mark_path):
                print(f"\n  偵測到既有標記檔案: {_mark_path}")
                loaded_obj, _ok = load_bad_marking(loaded_obj, _sid)
                if _ok:
                    print(f"  ✓ 標記已自動套用（壞通道: {loaded_obj.info['bads']}，"
                          f"BAD 片段: {sum(1 for a in loaded_obj.annotations if a['description'].startswith('BAD'))} 個）")
            else:
                print("  （未找到既有標記檔案）")
        except Exception as _e_mark:
            print(f"  ⚠ 自動載入標記時發生錯誤: {_e_mark}")

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
                # 先問是否載入既有標記 JSON
                from mne_python_analysis.preprocessing import load_bad_marking, _bad_marking_path
                import os as _os18
                _mark_path = _bad_marking_path(subject_id)
                if _os18.path.exists(_mark_path):
                    print(f"\n  偵測到既有標記檔案: {_mark_path}")
                    _load_prev = input("  是否先載入既有標記？(y/n) [y]: ").strip().lower() or 'y'
                    if _load_prev == 'y':
                        current_raw, _ = load_bad_marking(current_raw, subject_id)
                current_raw = mark_bad_segments_interactive(current_raw, subject_id=subject_id)
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
                                # === 步驟 0: artifact rejection 前先做 Stim/Resp 一對一同步 ===
                                # 修正根本問題：Response 有對應 Stimulus 才保留，確保數量一致
                                valid_resp_indices = sorted(resp_to_stim_map.keys())
                                valid_stim_indices = sorted(set(resp_to_stim_map[i] for i in valid_resp_indices))

                                # 確保雙向一致：每個 Stimulus 只被一個 Response 對應到
                                stim_to_resp = {}
                                for r_idx, s_idx in resp_to_stim_map.items():
                                    if s_idx not in stim_to_resp:
                                        stim_to_resp[s_idx] = r_idx
                                # 以 Stimulus 為主，保留最早的 Response 對應
                                valid_resp_indices = sorted(stim_to_resp.values())
                                valid_stim_indices = sorted(stim_to_resp.keys())

                                n_before_sync_r = len(epochs_resp)
                                n_before_sync_s = len(epochs_stim)
                                epochs_resp = epochs_resp[valid_resp_indices]
                                epochs_stim = epochs_stim[valid_stim_indices]

                                print(f"\n[Stim/Resp 同步]")
                                print(f"  Response: {n_before_sync_r} → {len(epochs_resp)} "
                                      f"（移除 {n_before_sync_r - len(epochs_resp)} 個無對應 Stim 的 Resp）")
                                print(f"  Stimulus: {n_before_sync_s} → {len(epochs_stim)} "
                                      f"（移除 {n_before_sync_s - len(epochs_stim)} 個無對應 Resp 的 Stim）")
                                assert len(epochs_resp) == len(epochs_stim),                                     f"同步後數量仍不一致: Resp={len(epochs_resp)}, Stim={len(epochs_stim)}"

                                # 更新 resp_to_stim_map（同步後 index 重新從 0 開始）
                                resp_to_stim_map = {new_r: new_s
                                                    for new_r, (new_s, _) in enumerate(
                                                        (s, r) for s, r in enumerate(valid_stim_indices)
                                                    )}
                                # 實際上同步後是 1:1，直接建立 identity map
                                resp_to_stim_map = {i: i for i in range(len(epochs_resp))}

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
                                    
                                    # 同步後 resp_to_stim_map 是 identity，直接用 kept_resp_indices
                                    kept_stim_indices = sorted(set(
                                        resp_to_stim_map[r] for r in kept_resp_indices
                                        if r in resp_to_stim_map
                                    ))
                                    
                                    print(f"  對應到 {len(kept_stim_indices)} 個 Stimulus trials")
                                    assert len(kept_stim_indices) == len(kept_resp_indices),                                         f"Stim/Resp 保留數量不一致: Stim={len(kept_stim_indices)}, Resp={len(kept_resp_indices)}"
                                    
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
                print("\n輸出資料夾:")
                print("  1. Trigger 分類結果 → C:\\Experiment\\ersp_results")
                print("  2. Triplet 分類結果 → C:\\Experiment\\ersp_results\\triplet")
                dir_choice = input("請選擇 (1/2) [1]: ").strip() or '1'
                output_dir = r'C:\Experiment\ersp_results\triplet' if dir_choice == '2' else r'C:\Experiment\ersp_results'

                for phase in ['learning', 'testing']:
                    print(f"\n" + "="*60)
                    print(f"開始 ERSP 分析（{phase.capitalize()} 階段）...")
                    print("="*60)
                    print(f"  受試者: {analysis_sid}")
                    print(f"  階段: {phase.capitalize()}")
                    print("  鎖定: Stimulus-locked")
                    print(f"  Epochs: {len(stim_epochs)}")
                    print(f"  儲存群體資料: {'是' if save_for_group else '否'}")

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

                            elec_out15 = os.path.join(_h5_dir15, '..', 'single_electrode')
                            os.makedirs(elec_out15, exist_ok=True)

                            import glob as _glob15
                            import numpy as _np15
                            import matplotlib.pyplot as _plt15
                            matplotlib.rcParams["axes.unicode_minus"] = False
                            from mne_python_analysis.group_ersp_analysis import _load_h5_single_electrode

                            # 三組比較：Regular High vs Random Low / vs Random High / Random High vs Random Low
                            _triplet_pairs15 = [
                                ('regular_high', 'random_low',  'Regular High', 'Random Low'),
                                ('regular_high', 'random_high', 'Regular High', 'Random High'),
                                ('random_high',  'random_low',  'Random High',  'Random Low'),
                            ]

                            def _single_elec_plot15(fp_l15, fp_r15, electrode15, disp_l, disp_r, suffix):
                                try:
                                    el15, freqs15, times15, _nave15 = _load_h5_single_electrode(fp_l15, electrode15)
                                    er15, _, _, _          = _load_h5_single_electrode(fp_r15, electrode15)
                                except Exception as ex15:
                                    print(f"    ⚠ {os.path.basename(fp_l15)}: {ex15}")
                                    return
                                diff15 = el15 - er15
                                x_min15, x_max15 = -0.5, 0.5
                                t_mask15 = (times15 >= x_min15) & (times15 <= x_max15)
                                combined15 = _np15.concatenate([el15[:, t_mask15].ravel(), er15[:, t_mask15].ravel()])
                                vmax15_c = _np15.percentile(_np15.abs(combined15), 95)
                                vmax15_d = _np15.percentile(_np15.abs(diff15[:, t_mask15].ravel()), 95)
                                if vmax15_c < 1e-10: vmax15_c = 1e-10
                                if vmax15_d < 1e-10: vmax15_d = 1e-10
                                lv15_c = _np15.linspace(-vmax15_c, vmax15_c, 20)
                                lv15_d = _np15.linspace(-vmax15_d, vmax15_d, 20)
                                base15 = os.path.basename(fp_l15).replace('_ERSP.h5', '').replace(f'{analysis_sid}_Stimulus_', '')
                                block_label15 = '_'.join(base15.split('_')[:-1])
                                fig15, axes15 = _plt15.subplots(1, 3, figsize=(18, 5))
                                for ax15, data15, title15, lv15, vm15, cbl15 in [
                                    (axes15[0], el15,   disp_l,                              lv15_c, vmax15_c, 'Power (dB)'),
                                    (axes15[1], er15,   disp_r,                              lv15_c, vmax15_c, 'Power (dB)'),
                                    (axes15[2], diff15, f'Diff ({disp_l} − {disp_r})', lv15_d, vmax15_d, 'Power Diff (dB)'),
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
                                    f'{analysis_sid} | Stimulus-locked | {block_label15} | Electrode: {electrode15} | {disp_l} vs {disp_r}',
                                    fontsize=11, fontweight='bold'
                                )
                                _plt15.tight_layout()
                                out_fig15 = os.path.join(elec_out15,
                                    f'{analysis_sid}_stimulus_{electrode15}_{block_label15}_{suffix}_comparison.png')
                                fig15.savefig(out_fig15, dpi=300, bbox_inches='tight')
                                _plt15.close(fig15)
                                print(f"    ✓ {disp_l} vs {disp_r}: {out_fig15}")

                            for electrode15 in electrodes15:
                                print(f"\n  電極: {electrode15}")
                                for lbl_l15, lbl_r15, disp_l15, disp_r15 in _triplet_pairs15:
                                    h5_files_l15 = sorted(_glob15.glob(
                                        os.path.join(_h5_dir15, f'{analysis_sid}_Stimulus_*_{lbl_l15}_ERSP.h5')))
                                    if not h5_files_l15:
                                        print(f"    ⚠ 找不到 {lbl_l15} h5 檔案，跳過")
                                        continue
                                    suffix15 = f"{lbl_l15}_vs_{lbl_r15}"
                                    for fp_l15 in h5_files_l15:
                                        fp_r15 = fp_l15.replace(f'_{lbl_l15}_', f'_{lbl_r15}_')
                                        if not os.path.exists(fp_r15):
                                            continue
                                        _single_elec_plot15(fp_l15, fp_r15, electrode15, disp_l15, disp_r15, suffix15)

                                # ── Epoch 4 vs Epoch 1（每個 triplet 條件）──
                                print(f"\n  [{electrode15}] Epoch 4 vs Epoch 1 比較")
                                _DISP15 = {
                                    'regular_high': 'Regular High',
                                    'random_high':  'Random High',
                                    'random_low':   'Random Low',
                                }
                                for _cond15 in ['regular_high', 'random_high', 'random_low']:
                                    _fp_e1_15 = os.path.join(
                                        _h5_dir15,
                                        f'{analysis_sid}_Stimulus_Learning_Block7-11_{_cond15}_ERSP.h5')
                                    _fp_e4_15 = os.path.join(
                                        _h5_dir15,
                                        f'{analysis_sid}_Stimulus_Learning_Block22-26_{_cond15}_ERSP.h5')
                                    if not os.path.exists(_fp_e1_15) or not os.path.exists(_fp_e4_15):
                                        print(f"    ⚠ Epoch4vsEpoch1 {_cond15}: 檔案不存在，跳過")
                                        continue
                                    _disp15 = _DISP15.get(_cond15, _cond15)
                                    _suffix15_e = f'epoch4_vs_epoch1_{_cond15}'
                                    # 左=Epoch4, 右=Epoch1, Diff=Epoch4-Epoch1
                                    _single_elec_plot15(
                                        _fp_e4_15, _fp_e1_15, electrode15,
                                        f'Epoch 4 (Block22-26)\n{_disp15}',
                                        f'Epoch 1 (Block7-11)\n{_disp15}',
                                        _suffix15_e)

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

                # ── Log 同步輸出到檔案 ──
                import sys, datetime
                _log_path = r'C:\Experiment\Result\group_ersp_analysis_log.txt'
                class _Tee:
                    def __init__(self, *files):
                        self.files = files
                    def write(self, obj):
                        for f in self.files:
                            f.write(obj)
                            f.flush()
                    def flush(self):
                        for f in self.files:
                            f.flush()
                _log_file = open(_log_path, 'w', encoding='utf-8')
                _log_file.write(f"=== Group ERSP Analysis Log ===\n")
                _log_file.write(f"Start: {datetime.datetime.now()}\n\n")
                _orig_stdout = sys.stdout
                sys.stdout = _Tee(_orig_stdout, _log_file)

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
                    H5_DIR     = r'C:\Experiment\Result\triplet\h5'
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
                print("  1. 每個圖各自計算（預設）")
                print("  2. Section-level 統一（推薦）：")
                print("     同一節內所有投影片共用同一 colorbar，")
                print("     涵蓋 所有 condition pair × Motor ROI × Perceptual ROI")
                print("     ─ Learning Ep1/Ep2/Ep3/Ep4 共用一組")
                print("     ─ Learning Ep4-Ep1 共用一組")
                print("     ─ Testing Motor / Perceptual 各共用一組")
                cb_choice = input("請選擇 (1/2) [2]: ").strip() or '2'
                unified_colorbar = (cb_choice == '2')

                # ── 詢問是否跳過群體 ROI 分析，直接進行單一電極分析 ──
                print("\n" + "─"*60)
                skip_roi = input("是否跳過群體 ROI 分析，直接進行單一電極分析？(y/n) [n]: ").strip().lower() or 'n'

                if skip_roi != 'y':
                    results = auto_group_ersp_analysis(
                        subject_ids         = subject_ids,
                        pkl_dir             = PKL_DIR,
                        h5_dir              = H5_DIR,
                        output_dir          = OUTPUT_DIR,
                        do_permutation_test = do_permutation,
                        n_permutations      = 1000,
                        unified_colorbar    = unified_colorbar,
                        display_label1      = None,
                        display_label2      = None,
                    )

                    n_done = sum(1 for v in results.values() if v)
                    processing_history.append(
                        f"ASRT 群體分析（全自動，{len(subject_ids)} 位受試者，{n_done} 個組合完成）"
                    )
                else:
                    print("  → 跳過群體 ROI 分析")

                # ── 單一電極群體分析 ──
                print("\n" + "─"*60)
                elec_choice = input("是否進行單一電極群體分析？(y/n) [n]: ").strip().lower() or 'n'
                if elec_choice == 'y':
                    elec_input = input("請輸入電極名稱（逗號分隔，例如: Fz,Cz,Pz）: ").strip()
                    if elec_input:
                        electrodes = [e.strip() for e in elec_input.split(',') if e.strip()]
                        elec_out = os.path.join(OUTPUT_DIR, 'single_electrode')

                        if analysis_type == '2':
                            lbl_left, lbl_right = 'regular_high', 'random_low'
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
                            condition_left  = lbl_left,
                            condition_right = lbl_right,
                            stim_h5_dir     = PKL_DIR,
                        )
                        processing_history.append(f"單一電極群體分析（{', '.join(electrodes)}）")

            except ImportError:
                print("❌ 群體分析模組未安裝")
                print("請確認 group_ersp_analysis.py 已放在 mne_python_analysis/ 目錄下")
            except Exception as e:
                print(f"❌ 群體分析時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                try:
                    sys.stdout = _orig_stdout
                    _log_file.write(f"\nEnd: {datetime.datetime.now()}\n")
                    _log_file.close()
                    print(f"\n✓ Log 已儲存至: {_log_path}")
                except Exception:
                    pass

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
                    # Bug 修正：原本順序是 Parietal,Occipital,Frontal,Central，
                    # 跟上面 Motor_* 的 Frontal,Central,Parietal,Occipital 順序不一致。
                    # 下面 zip(_learn_motor_rois, _learn_perc_rois) 是按「字典順序的位置」
                    # 配對，順序不一致會導致 Motor_Frontal 被錯配到 Perceptual_Parietal、
                    # Motor_Central 配到 Perceptual_Occipital 等，全部子區域配對都錯。
                    # 改成跟 Motor_* 同樣順序，才能讓 Frontal 配 Frontal、Central 配 Central。
                    'Perceptual_Frontal':   ['Fz', 'FCz'],
                    'Perceptual_Central':   ['Cz', 'C3', 'C4'],
                    'Perceptual_Parietal':  ['P3', 'Pz', 'P4'],
                    'Perceptual_Occipital': ['O1', 'Oz', 'O2'],
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

                    # Triplet 建立（Kóbor 2019：Trill/Repetition 排除）
                    def _triplets(seq):
                        r = [None, None]
                        for i in range(2, len(seq)):
                            n2,n1,n = seq[i-2],seq[i-1],seq[i]
                            if n2 == n1 == n:   # Repetition
                                r.append(None)
                            elif n2 == n:        # Trill
                                r.append(None)
                            else:
                                r.append(f"{n2}{n1}{n}")
                        return r

                    trip_map = {}
                    for (bt, bid), grp in df.groupby(['block_type', 'block_id']):
                        gs = grp.sort_values('trial_in_block')
                        if 'correct_answer_direction' not in gs.columns:
                            continue
                        seq = gs['correct_answer_direction'].astype(float).fillna(0).astype(int).tolist()
                        for idx, tri in zip(gs.index, _triplets(seq)):
                            trip_map[idx] = tri
                    df['triplet'] = df.index.map(lambda i: trip_map.get(i))

                    # ── P 序列萃取（consecutive P pair 方法，與 R 一致）──────────
                    _p_consec_pairs = set()
                    med_c = np.nan   # Bug 5 修正：外層先初始化；P 序列成功時為 nan，fallback 時覆寫
                    _key_color = 'arrow_color'
                    _key_lb    = 'learning_trials.thisTrialN'
                    _key_lt    = 'learning_loop.thisTrialN'
                    if all(c in df.columns for c in [_key_color, _key_lb, _key_lt, 'correct_answer_direction']):
                        _blk0 = df[
                            (df[_key_lb].astype(float).fillna(-1).astype(int) == 0) &
                            (df[_key_lt].astype(float) >= 5) &
                            (df[_key_color].str.lower() == 'white') &
                            df['correct_answer_direction'].notna()
                        ].sort_values(_key_lt)
                        _p_seq = _blk0['correct_answer_direction'].astype(int).tolist()[:4]
                        if len(_p_seq) == 4:
                            _p_consec_pairs = {(_p_seq[i], _p_seq[(i+1)%4]) for i in range(4)}
                            print(f"  [{sid_label}] P 序列: {_p_seq}  Consecutive pairs: {_p_consec_pairs}")
                        else:
                            print(f"  [{sid_label}] ⚠  P 序列萃取失敗，回退至 median 分類")
                    else:
                        print(f"  [{sid_label}] ⚠  缺少必要欄位，回退至 median 分類")

                    # Bug 5 修正：fallback 的 Counter 和 median 移到 _fcat 外部計算一次
                    _rc_fallback = None
                    if not _p_consec_pairs:
                        _lrm = (
                            (df['phase'] == 'Learning') &
                            (df['position_type'] == 'random') &
                            df['triplet'].notna()
                        )
                        _rc_fallback = _Counter(df.loc[_lrm, 'triplet'].tolist())
                        med_c = np.median(list(_rc_fallback.values())) if _rc_fallback else 0.
                        print(f"  [{sid_label}] Fallback median count: {med_c}")

                    # Frequency category（consecutive P pair 方法，Trill/Repetition → None 排除）
                    def _fcat(row):
                        if row['position_type'] == 'regular':
                            return 'regular_high'
                        if row['position_type'] == 'random':
                            if pd.isna(row['triplet']):
                                return None   # Trill / Repetition 排除
                            if _p_consec_pairs:
                                tri = str(row['triplet'])
                                t1_val, t3_val = int(tri[0]), int(tri[2])
                                return 'random_high' if (t1_val, t3_val) in _p_consec_pairs else 'random_low'
                            else:
                                cnt = _rc_fallback.get(row['triplet'], 0)
                                return 'random_high' if cnt >= med_c else 'random_low'
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
                    df['triplet_median_count'] = med_c   # 現在外層 scope 一定有 med_c
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
                def _load_ersp_18(h5_dir_path, sids, ltype, freq_r=None, time_r=None):
                    prefix = 'Response' if ltype == 'response' else 'Stimulus'
                    if freq_r is None:
                        freq_r = (4., 8.)   if ltype == 'response' else (8., 13.)
                    if time_r is None:
                        time_r = (-.300, .050) if ltype == 'response' else (.100, .300)
                    _VALID_TT = {'regular_high', 'random_high', 'random_low'}
                    rows = []
                    for fp in sorted(Path(h5_dir_path).glob(f'*_{prefix}_*.h5')):
                        stem = fp.stem.replace('_ERSP', '')
                        parts = stem.split('_')
                        # 檔名格式：{sid}_{prefix}_{phase}_{block_group}_{tt_p1}_{tt_p2}
                        # trial_type 含底線（regular_high 等），共佔 parts[-2] 和 parts[-1]
                        # 所以索引整體向左移一位：prefix 在 parts[-5]
                        try:
                            if len(parts) < 6:
                                continue
                            trial_type  = f"{parts[-2]}_{parts[-1]}".lower()
                            block_group = parts[-3]
                            phase       = parts[-4]
                            sid18       = '_'.join(parts[:-5])
                            if parts[-5] != prefix or sid18 not in sids:
                                continue
                            if trial_type not in _VALID_TT or not block_group.startswith('Block'):
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
                    if len(df18) > 0:
                        print(f"    trial_type: {sorted(df18['trial_type'].unique())}")
                        print(f"    phase:      {sorted(df18['phase'].unique())}")
                    return df18

                # ────────────────────────────────────────────────────────────
                # 內部函數：Join
                # ────────────────────────────────────────────────────────────
                def _join_18(ersp_df, beh_sum):
                    def _p2bt(p): return 'learning' if p == 'Learning' else 'testing'
                    def _p2tt(p):
                        if p == 'MotorTest':      return 'motor'
                        if p == 'PerceptualTest': return 'perceptual'
                        return 'none'
                    e = ersp_df.copy()
                    e['block_type'] = e['phase'].map(_p2bt)
                    e['tasktype']   = e['phase'].map(_p2tt)
                    b = beh_sum.copy()
                    # Bug 2 修正：相容兩種 beh_sum 來源
                    # PsychoPy 模式 → 有 test_type，沒有 tasktype
                    # R CSV 模式   → 有 tasktype（已映射完畢），沒有 test_type
                    if 'test_type' in b.columns and 'tasktype' not in b.columns:
                        b['tasktype'] = b['test_type'].fillna('none')
                    # R CSV 模式的 beh_sum 已有正確的 tasktype，直接沿用，不做任何操作
                    b = b.drop(columns=[c for c in ['phase', 'block_type'] if c in b.columns])
                    print("  [Debug] ERSP 側 trial_type:", sorted(e['trial_type'].unique()))
                    print("  [Debug] ERSP 側 tasktype:  ", sorted(e['tasktype'].unique()))
                    print("  [Debug] ERSP 側 block_group:", sorted(e['eeg_block_group'].unique()))
                    print("  [Debug] 行為 側 trial_type:", sorted(b['trial_type'].unique()))
                    print("  [Debug] 行為 側 tasktype:  ", sorted(b['tasktype'].unique()))
                    print("  [Debug] 行為 側 block_group:", sorted(b['eeg_block_group'].unique()))
                    joined = e.merge(b, on=['sid', 'eeg_block_group', 'trial_type', 'tasktype'],
                                        how='inner')
                    if 'phase' not in joined.columns and 'phase_eeg' in joined.columns:
                        joined['phase'] = joined['phase_eeg']
                    # Bug 5 修正：Bug 4 的正規化會把 trial_type 從
                    # 'regular_high'/'random_high'/'random_low' 攤平成只剩 'high'/'low'，
                    # 導致 regular_high 與 random_high 被混在一起，下游無法做
                    # Regular_High vs Random_Low 的核心比較（R 端 compute_eeg_corr_matrix
                    # 與 Shiny EEG-行為整合 tab 皆受影響）。
                    # 修法：在正規化「之前」先把完整三分類存進新欄位 condition，
                    # 保留給需要精細分類的分析使用；trial_type/frequency_category
                    # 仍維持原本的 high/low 正規化，不影響既有下游函式
                    # （_plot_dir1_18 等皆依賴 trial_type == 'high'/'low'）。
                    joined['condition'] = joined['trial_type']  # regular_high / random_high / random_low
                    # Bug 4 修正：正規化 trial_type 和 frequency_category 為 'high'/'low'
                    # trial_type    → Python 繪圖函式使用
                    # frequency_category → Shiny pivot_wider 使用
                    _NORM = {'regular_high': 'high', 'random_high': 'high', 'random_low': 'low'}
                    joined['trial_type'] = (
                        joined['trial_type'].map(_NORM).fillna(joined['trial_type'])
                    )
                    if 'frequency_category' in joined.columns:
                        joined['frequency_category'] = (
                            joined['frequency_category'].map(_NORM).fillna(joined['frequency_category'])
                        )
                    print(f"  ✓ Join: {len(joined)} rows, {joined['sid'].nunique()} 受試者")
                    print(f"  trial_type 正規化後: {sorted(joined['trial_type'].unique())}")
                    print(f"  condition（完整三分類，供精細分析用）: {sorted(joined['condition'].unique())}")
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
                    ax.set_title(f'EEG-Behavior Correlation | {roi} | {phase}',
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
                    ax1.set_title(f'Learning Curve | {roi} | {phase} | {title_s}',
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
                    ll3 = 'per subject' if level=='subject' else 'per subject x Block Group'
                    ax3.set_xlabel('RT Effect: High - Low (ms)', fontsize=11)
                    ax3.set_ylabel('ERSP Effect: High - Low (dB)', fontsize=11)
                    ax3.set_title(f'EEG-Behavior Correlation [{ll3}] | {roi} | {phase}',
                                  fontsize=11, fontweight='bold')
                    plt.tight_layout()
                    fig3.savefig(out_path, dpi=200, bbox_inches='tight')
                    plt.close(fig3)
                    print(f"  ✓ 方向三圖: {os.path.basename(out_path)}")
                    return fig3, {'r':r3,'p':p3,'n':len(pdf3)}

                # ════════════════════════════════════════════════════════════
                # 主流程：使用者輸入
                # ════════════════════════════════════════════════════════════

                # ── Step 1: 行為資料來源 & 受試者設定 ──────────────────────────
                print("\n" + "─"*60)
                print("Step 1：行為資料來源 & 受試者資訊")
                print("─"*60)

                print("\n  行為資料來源:")
                print("    1. 讀取 R 輸出的 all_d_summary.csv（推薦，與 R 完全對齊）")
                print("    2. 從 PsychoPy CSV 重建（獨立計算，需輸入每人 CSV 路徑）")
                _beh_src_18 = input("  請選擇 (1/2) [1]: ").strip() or '1'
                _use_r_csv_18 = (_beh_src_18 == '1')

                psychopy_csv_paths_18 = {}

                if _use_r_csv_18:
                    # ── R CSV 模式：只需受試者 ID 清單 ──────────────────────
                    print("\n  請輸入要分析的受試者 ID（留空 Enter 結束）：")
                    while True:
                        _sid_in = input("  受試者 ID（Enter 結束）: ").strip()
                        if not _sid_in:
                            if not psychopy_csv_paths_18:
                                print("  ⚠ 至少需要輸入一位受試者")
                                continue
                            break
                        psychopy_csv_paths_18[_sid_in] = {'path': None, 'test_version': None}
                        print(f"  ✓ 已加入 {_sid_in}")
                    _default_r_csv = r'C:\Experiment\asrt_percept_motor-master_debug\analysis - modify_asrt\proc_data\all_d_summary.csv'
                    _r_csv_path_18 = input(f"  all_d_summary.csv 路徑 [{_default_r_csv}]: ").strip().strip('"') or _default_r_csv
                else:
                    # ── PsychoPy CSV 模式：原有輸入方式 ──────────────────────
                    print("（每次輸入一位，留空 Enter 結束輸入）")
                    print("  test_version 說明：")
                    print("    motor_first    → Testing block 27,28,33,34 = motor")
                    print("    perceptual_first → Testing block 27,28,33,34 = perceptual")
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
                    _r_csv_path_18 = None

                subject_ids_18 = list(psychopy_csv_paths_18.keys())
                print(f"\n  受試者設定：")
                for _s, _info in psychopy_csv_paths_18.items():
                    if _use_r_csv_18:
                        print(f"    {_s}")
                    else:
                        print(f"    {_s}: {_info['test_version']}")

                # ── Step 2: 分析設定 ─────────────────────────────────────────
                print("\n" + "─"*60)
                print("Step 2：分析設定")
                print("─"*60)

                # Lock type（先選，才能設對預設 h5 目錄）
                print("\n  Lock type:")
                print("    1. response（預設）")
                print("    2. stimulus")
                ltype_choice = input("  請選擇 (1/2) [1]: ").strip() or '1'
                lock_type_18 = 'response' if ltype_choice != '2' else 'stimulus'

                # H5 目錄（根據 lock type 設對應預設路徑）
                # response-locked triplet ERSP 存在 triplet\h5（high/low 命名）
                # stimulus-locked ERSP 存在 h5（high/low 命名）
                if lock_type_18 == 'response':
                    _default_h5 = r'C:\Experiment\Result\triplet\h5'
                else:
                    _default_h5 = r'C:\Experiment\Result\h5'
                h5_dir_18 = input(f"  ERSP .h5 檔案目錄 [{_default_h5}]: ").strip().strip('"') or _default_h5

                # 輸出目錄
                _default_out = r'C:\Experiment\Result\eeg_behavior'
                out_dir_18 = input(f"  輸出目錄 [{_default_out}]: ").strip().strip('"') or _default_out

                # 相關分析方法
                print("\n  相關分析方法:")
                print("    1. spearman（預設）")
                print("    2. pearson")
                cm_choice = input("  請選擇 (1/2) [1]: ").strip() or '1'
                corr_method_18 = 'spearman' if cm_choice != '2' else 'pearson'

                # 方向三分析層次
                print("\n  方向三分析層次:")
                print("    1. subject（每人一點，預設）")
                print("    2. block（每人×每 Block Group 一點）")
                d3_choice = input("  請選擇 (1/2) [1]: ").strip() or '1'
                dir3_level_18 = 'subject' if d3_choice != '2' else 'block'

                # 方向二是否顯示群體平均
                group_level_18 = (input("  方向二是否顯示群體平均（y=群體, n=個別受試者）[y]: ").strip().lower() or 'y') == 'y'

                # ROI 選擇
                print("\n  ROI 選擇（留空 Enter = 全部 10 個 ROI）:")
                roi_input_18 = input("  （逗號分隔，例如: Motor,Perceptual）: ").strip()
                if roi_input_18:
                    rois_18 = [r.strip() for r in roi_input_18.split(',') if r.strip() in _ROI_GROUPS_18]
                    if not rois_18:
                        rois_18 = list(_ROI_GROUPS_18.keys())
                else:
                    rois_18 = list(_ROI_GROUPS_18.keys())

                # phases_18 將在 Join 之後確定（Step 5 後設定）
                phases_18 = []  # placeholder，Step 5 後更新

                print(f"\n  ✓ 設定確認")
                print(f"    Lock type:  {lock_type_18}")
                print(f"    相關方法:   {corr_method_18}")
                print(f"    方向三層次: {dir3_level_18}")
                print(f"    ROI 數量:   {len(rois_18)}")
                if not _use_r_csv_18:
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

                # ── blockID2（R 格式）→ eeg_block_group 對應表 ──────────────
                _UNITX_TO_EEG_18 = {}
                for _i in range(1, 6):   _UNITX_TO_EEG_18[f'L{_i:02d}'] = 'Block7-11'
                for _i in range(6, 11):  _UNITX_TO_EEG_18[f'L{_i:02d}'] = 'Block12-16'
                for _i in range(11, 16): _UNITX_TO_EEG_18[f'L{_i:02d}'] = 'Block17-21'
                for _i in range(16, 21): _UNITX_TO_EEG_18[f'L{_i:02d}'] = 'Block22-26'
                for _i in range(1, 3):   _UNITX_TO_EEG_18[f'T{_i:02d}'] = 'Block27-28'
                for _i in range(3, 5):   _UNITX_TO_EEG_18[f'T{_i:02d}'] = 'Block29-30'
                for _i in range(5, 7):   _UNITX_TO_EEG_18[f'T{_i:02d}'] = 'Block31-32'
                for _i in range(7, 9):   _UNITX_TO_EEG_18[f'T{_i:02d}'] = 'Block33-34'

                if _use_r_csv_18:
                    # ════ R CSV 模式 ════════════════════════════════════════
                    print(f"  讀取 R 輸出: {_r_csv_path_18}")
                    try:
                        _r_df = pd.read_csv(_r_csv_path_18)
                        print(f"  ✓ R CSV 讀取成功: {len(_r_df)} rows, 欄位: {list(_r_df.columns)}")
                    except Exception as _e:
                        print(f"  ✗ 讀取失敗: {_e}")
                        continue

                    # 只保留指定受試者
                    _r_df = _r_df[_r_df['sid'].isin(subject_ids_18)].copy()
                    if len(_r_df) == 0:
                        print(f"  ✗ R CSV 中找不到受試者 {subject_ids_18}，請確認 sid 欄位")
                        continue

                    # blockID2 → eeg_block_group
                    _r_df['eeg_block_group'] = _r_df['unitx'].map(_UNITX_TO_EEG_18)
                    _r_df = _r_df[_r_df['eeg_block_group'].notna()].copy()

                    # Bug 3 修正：組合 trial_type + frequency_category
                    # CSV 欄位：trial_type='regular'/'random'，frequency_category='high'/'low'
                    # ERSP h5：trial_type='regular_high'/'random_high'/'random_low'
                    # 必須組合才能 join 對齊
                    _r_df['trial_type'] = (
                        _r_df['trial_type'].str.strip() + '_' +
                        _r_df['frequency_category'].str.strip()
                    )
                    _VALID_TT_18 = {'regular_high', 'random_high', 'random_low'}
                    _n_before = len(_r_df)
                    _r_df = _r_df[_r_df['trial_type'].isin(_VALID_TT_18)].copy()
                    if len(_r_df) < _n_before:
                        print(f"  ⚠ 過濾掉 {_n_before - len(_r_df)} 行非預期 trial_type")

                    # R tasktype → EEG tasktype
                    # "learning"→"none", "motor"→"motor", "percept"→"perceptual"
                    _tt_map = {'learning': 'none', 'motor': 'motor', 'percept': 'perceptual'}
                    _r_df['tasktype'] = _r_df['tasktype'].map(_tt_map).fillna(_r_df['tasktype'])

                    # 聚合到 eeg_block_group 層次
                    beh_sum_18 = (
                        _r_df.groupby(['sid', 'eeg_block_group', 'trial_type',
                                       'tasktype', 'frequency_category'])
                        .agg(rt=('rt', 'median'), n_trials=('n', 'sum'))
                        .reset_index()
                    )
                    print(f"  ✓ R 行為資料聚合完成: {len(beh_sum_18)} rows")
                    print(f"    受試者:    {sorted(beh_sum_18['sid'].unique())}")
                    print(f"    trial_type:{sorted(beh_sum_18['trial_type'].unique())}")
                    print(f"    tasktype:  {sorted(beh_sum_18['tasktype'].unique())}")
                else:
                    # ════ PsychoPy CSV 模式 ═════════════════════════════════
                    all_beh_18 = []
                    _verif_dir_18 = os.path.join(out_dir_18, 'verification')
                    os.makedirs(_verif_dir_18, exist_ok=True)

                    for sid18, cpath18 in psychopy_csv_paths_18.items():
                        print(f"\n  處理 {sid18}...")
                        try:
                            _info18 = cpath18 if isinstance(cpath18, dict) else {'path': cpath18, 'test_version': 'motor_first'}
                            _csv18  = _info18['path']
                            _tv18   = _info18['test_version']
                            print(f"    test_version: {_tv18}")
                            psy18 = pd.read_csv(_csv18, low_memory=False)
                            beh18, med_c_18 = _build_behavior_18(psy18, sid18, _tv18)
                            all_beh_18.append(beh18)

                            _verif_grp = (
                                beh18
                                .assign(test_type=lambda x: x['test_type'].fillna('none'))
                                .groupby(['sid', 'block_type', 'eeg_block_group',
                                          'phase', 'test_type',
                                          'position_type', 'frequency_category'])
                                .agg(
                                    n_trials  = ('rt_ms', 'count'),
                                    median_rt = ('rt_ms', 'median'),
                                    mean_rt   = ('rt_ms', 'mean'),
                                )
                                .reset_index()
                            )
                            _verif_grp['triplet_median_count'] = med_c_18
                            _verif_grp['source'] = 'python'
                            _verif_path = os.path.join(_verif_dir_18, f'{sid18}_verification_python.csv')
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

                # 確定實際存在的 phase 清單
                phases_18 = sorted(joined_18['phase'].unique())

                # 確定 rois_18（若 join 後某些 ROI 沒有資料則篩掉）
                rois_18 = [r for r in rois_18 if r in joined_18['roi'].unique()]
                if not rois_18:
                    rois_18 = sorted(joined_18['roi'].unique())

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

                # ── Step 7: 方向四 — Band/Window 平均功率 vs 行為指標相關 ──
                print("\n" + "─"*60)
                print(f"Step 7：方向四 — 頻段平均功率 × 行為指標相關（N={len(subject_ids_18)} 位受試者）")
                print("─"*60)

                # 使用者選 band
                print("\n  頻段選擇:")
                print("    1. Theta  4–8  Hz")
                print("    2. Alpha  8–13 Hz")
                print("    3. 自訂")
                _band_ch = input("  請選擇 (1/2/3) [1]: ").strip() or '1'
                if _band_ch == '1':
                    _d4_bands = [('Theta', 4., 8.)]
                elif _band_ch == '2':
                    _d4_bands = [('Alpha', 8., 13.)]
                elif _band_ch == '3':
                    _d4_bands = [('Theta', 4., 8.), ('Alpha', 8., 13.)]
                else:
                    _flo = float(input("  低頻 (Hz): ").strip())
                    _fhi = float(input("  高頻 (Hz): ").strip())
                    _d4_bands = [(f'{_flo:.0f}-{_fhi:.0f}Hz', _flo, _fhi)]

                # 使用者選時間窗
                print("\n  時間窗（秒，預設 -0.5 ~ +0.5）:")
                _tw_in = input("  格式: tmin,tmax [預設 -0.5,0.5]: ").strip()
                if _tw_in:
                    _tw_parts = [float(x) for x in _tw_in.split(',')]
                    _d4_tmin, _d4_tmax = _tw_parts[0], _tw_parts[1]
                else:
                    _d4_tmin, _d4_tmax = -0.5, 0.5

                # 使用者選行為指標
                print("\n  行為指標:")
                print("    1. RT 中位數（high 與 low 分開）")
                print("    2. High − Low RT 差值（sequence effect）")
                print("    3. 兩者都要")
                _beh_ch = input("  請選擇 (1/2/3) [3]: ").strip() or '3'
                _d4_beh_modes = []
                if _beh_ch in ('1', '3'): _d4_beh_modes += ['high', 'low']
                if _beh_ch in ('2', '3'): _d4_beh_modes += ['diff']

                dir4_out = os.path.join(out_dir_18, 'direction4_band_corr')
                os.makedirs(dir4_out, exist_ok=True)
                _d4_rows = []

                for _bname, _flo, _fhi in _d4_bands:
                    # 重新讀取 h5，使用指定 band/time
                    _ersp_d4 = _load_ersp_18(
                        h5_dir_18, subject_ids_18, lock_type_18,
                        freq_r=(_flo, _fhi), time_r=(_d4_tmin, _d4_tmax)
                    )
                    if len(_ersp_d4) == 0:
                        print(f"  ⚠ {_bname}: 讀不到 ERSP 資料，跳過")
                        continue

                    # 從 joined_18 取行為資料
                    _beh_d4 = joined_18[['sid','phase','eeg_block_group',
                                          'trial_type','rt']].drop_duplicates()

                    # 針對每個 ROI × phase 做相關
                    for _roi4 in rois_18:
                        for _ph4 in phases_18:
                            _e4 = _ersp_d4[(_ersp_d4['roi'] == _roi4) &
                                            (_ersp_d4['phase'] == _ph4)]
                            if len(_e4) == 0:
                                continue

                            for _bmode in _d4_beh_modes:
                                if _bmode in ('high', 'low'):
                                    # 每人取指定 trial_type 的 RT 中位數
                                    _e4_tt = _e4[_e4['trial_type'] == _bmode]
                                    _b4 = (_beh_d4[(_beh_d4['phase'] == _ph4) &
                                                    (_beh_d4['trial_type'] == _bmode)]
                                           .groupby('sid')['rt'].median().reset_index())
                                    _e4_sid = _e4_tt.groupby('sid')['ersp_mean'].mean().reset_index()
                                    _merge4 = _e4_sid.merge(_b4, on='sid')
                                    _xlabel = f'RT median ({_bmode})'
                                elif _bmode == 'diff':
                                    # 每人計算 high RT − low RT
                                    _b_hi = (_beh_d4[(_beh_d4['phase'] == _ph4) &
                                                      (_beh_d4['trial_type'] == 'high')]
                                             .groupby('sid')['rt'].median())
                                    _b_lo = (_beh_d4[(_beh_d4['phase'] == _ph4) &
                                                      (_beh_d4['trial_type'] == 'low')]
                                             .groupby('sid')['rt'].median())
                                    _b4 = (_b_hi - _b_lo).reset_index()
                                    _b4.columns = ['sid', 'rt']
                                    _e4_sid = _e4.groupby('sid')['ersp_mean'].mean().reset_index()
                                    _merge4 = _e4_sid.merge(_b4, on='sid')
                                    _xlabel = 'RT diff (high - low)'

                                if len(_merge4) < 3:
                                    continue

                                from scipy.stats import spearmanr, pearsonr
                                _corr_fn = spearmanr if corr_method_18 == 'spearman' else pearsonr
                                _r4, _p4 = _corr_fn(_merge4['ersp_mean'], _merge4['rt'])
                                _sig4 = ' *' if _p4 < .05 else ''
                                _d4_rows.append({
                                    'band': _bname, 'tmin': _d4_tmin, 'tmax': _d4_tmax,
                                    'roi': _roi4, 'phase': _ph4, 'beh_mode': _bmode,
                                    'lock_type': lock_type_18, 'n': len(_merge4),
                                    'r': float(_r4), 'p': float(_p4)
                                })

                                # 散點圖
                                _fig4, _ax4 = plt.subplots(figsize=(5, 4))
                                for _, _row4 in _merge4.iterrows():
                                    _ax4.scatter(_row4['rt'], _row4['ersp_mean'],
                                                 s=80, zorder=3)
                                    _ax4.annotate(_row4['sid'],
                                                  (_row4['rt'], _row4['ersp_mean']),
                                                  textcoords='offset points', xytext=(5,3),
                                                  fontsize=8)
                                # 趨勢線
                                import numpy as _np4
                                if len(_merge4) >= 2:
                                    _z4 = _np4.polyfit(_merge4['rt'], _merge4['ersp_mean'], 1)
                                    _px4 = _np4.linspace(_merge4['rt'].min(), _merge4['rt'].max(), 50)
                                    _ax4.plot(_px4, _np4.poly1d(_z4)(_px4), 'r--', alpha=0.6)
                                _ax4.set_xlabel(_xlabel, fontsize=10)
                                _ax4.set_ylabel(f'ERSP mean (dB) [{_bname}]', fontsize=10)
                                _ax4.set_title(
                                    f'{_roi4} | {_ph4} | {lock_type_18} | N={len(_merge4)}\n'
                                    f'{corr_method_18} r={_r4:+.3f}  p={_p4:.3f}{_sig4}',
                                    fontsize=9
                                )
                                _ax4.grid(True, alpha=0.3)
                                plt.tight_layout()
                                _fn4 = (f'dir4_{_bname}_{_roi4}_{_ph4}_{_bmode}'
                                        f'_{lock_type_18}.png')
                                _fig4.savefig(os.path.join(dir4_out, _fn4),
                                              dpi=200, bbox_inches='tight')
                                plt.close(_fig4)
                                print(f"  ✓ 方向四圖: {_fn4}")

                if _d4_rows:
                    _d4_df = pd.DataFrame(_d4_rows)
                    _d4_csv = os.path.join(out_dir_18, f'direction4_band_corr_{lock_type_18}.csv')
                    _d4_df.to_csv(_d4_csv, index=False)
                    print(f"\n  ✓ 方向四摘要 CSV: {_d4_csv}")
                    print(f"\n{'─'*60}")
                    print("方向四相關摘要（p < .05 標記 *）")
                    print(f"{'─'*60}")
                    for _, _rw4 in _d4_df.iterrows():
                        _sg4 = ' *' if _rw4['p'] < .05 else ''
                        print(f"  {_rw4['band']:10s} | {_rw4['roi']:25s} | "
                              f"{_rw4['phase']:15s} | {_rw4['beh_mode']:6s} | "
                              f"r={_rw4['r']:+.3f}  p={_rw4['p']:.3f}{_sg4}")

                # ── Step 8: 輸出摘要 CSV ─────────────────────────────────────
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

                # ── Step 9: 方向五 — 時間序列相關 CSV（供 Shiny 讀取）────────
                print("\n" + "─"*60)
                print("Step 9：方向五 — 時間序列相關 CSV（供 Shiny 讀取）")
                print("─"*60)
                print("\n  頻段選擇（時間序列）:")
                print("    1. Theta 4-8 Hz")
                print("    2. Alpha 8-13 Hz")
                print("    3. 兩者都要（預設）")
                _ts_band_ch = input("  請選擇 (1/2/3) [3]: ").strip() or '3'
                _ts_bands = []
                if _ts_band_ch in ('1', '3'): _ts_bands.append(('theta', 4., 8.))
                if _ts_band_ch in ('2', '3'): _ts_bands.append(('alpha', 8., 13.))

                _ts_rows_cross  = []
                _ts_rows_single = []
                _ts_prefix = 'Response' if lock_type_18 == 'response' else 'Stimulus'
                # Bug 修正：原本只處理 regular_high/random_high，且 _ts_data 的 key
                # 沒有 trial_type，導致 regular_high 與 random_high 互相覆蓋，
                # random_low 完全沒被納入。現在改為三條件獨立保存，並把 condition
                # 隨資料一起輸出到 CSV，供 R 端 Shiny 時間序列相關分頁的三組比較
                # 分面顯示使用（對應 _join_18 的 condition 欄位修正）。
                _CONDITIONS_18 = ('regular_high', 'random_high', 'random_low')

                for _ts_bname, _ts_flo, _ts_fhi in _ts_bands:
                    print(f"\n  處理 {_ts_bname}...")
                    # key: (sid, roi, phase_group, block_group, condition) → power_time_array
                    # phase_group: 'Learning', 'MotorTest', 'PerceptualTest'
                    # 跨受試者相關：同一 phase_group、同一 condition 內，Motor ROI vs Perceptual ROI
                    _ts_data = {}   # (sid, roi, phase_group, bg, condition) → power_time_array
                    _times_ref = None

                    for _fp in sorted(Path(h5_dir_18).glob(f'*_{_ts_prefix}_*.h5')):
                        _stem  = _fp.stem.replace('_ERSP', '')
                        _parts = _stem.split('_')
                        try:
                            if len(_parts) < 6:
                                continue
                            _tt18  = f"{_parts[-2]}_{_parts[-1]}".lower()  # 'regular_high' 等
                            _bg18  = _parts[-3]   # block_group，e.g. 'Block27-28' / 'AllBlocks'
                            _ph18  = _parts[-4]   # 'Learning' / 'MotorTest' / 'PerceptualTest'
                            _sid18 = '_'.join(_parts[:-5])
                            if _parts[-5] != _ts_prefix or _sid18 not in subject_ids_18:
                                continue
                            # 處理三條件（regular_high / random_high / random_low）
                            if _tt18 not in _CONDITIONS_18:
                                continue
                            # Testing：只用 AllBlocks 避免同一受試者重複
                            if _ph18 in ('MotorTest', 'PerceptualTest') and _bg18 != 'AllBlocks':
                                continue
                        except IndexError:
                            continue
                        except IndexError:
                            continue
                        try:
                            import warnings as _w18
                            with _w18.catch_warnings():
                                _w18.simplefilter('ignore')
                                _tl18 = mne.time_frequency.read_tfrs(str(_fp))
                            _tfr18 = _tl18[0] if isinstance(_tl18, list) else _tl18
                        except Exception:
                            continue

                        _freqs18 = _tfr18.freqs
                        _times18 = _tfr18.times
                        _fm18 = (_freqs18 >= _ts_flo) & (_freqs18 <= _ts_fhi)
                        if _times_ref is None:
                            _times_ref = _times18

                        _is_motor = 'MotorTest' in _ph18
                        _is_perc  = 'PerceptualTest' in _ph18
                        _is_learn = _ph18 == 'Learning'

                        _cnu18 = [c.upper() for c in _tfr18.ch_names]
                        for _rn18, _rch18 in _ROI_GROUPS_18.items():
                            _ridx18 = [_cnu18.index(c.upper()) for c in _rch18 if c.upper() in _cnu18]
                            if not _ridx18: continue
                            _power_t = _tfr18.data[_ridx18].mean(axis=0)[_fm18].mean(axis=0)
                            _ts_data[(_sid18, _rn18, _ph18, _bg18, _tt18)] = _power_t
                            # 單一受試者模式用
                            _ts_rows_single.append({
                                'sid': _sid18, 'roi': _rn18, 'phase': _ph18,
                                'band': _ts_bname, 'block_group': _bg18,
                                'condition': _tt18,
                                'tasktype': 'motor' if _is_motor else ('perceptual' if _is_perc else 'learning'),
                                'times': list(_times18),
                                'power': list(_power_t)
                            })

                    # 跨受試者相關：每個時間點
                    # Learning:  Motor ROI vs Perceptual ROI（同一人，同一 block group）
                    # Testing:   MotorTest_AllBlocks vs PerceptualTest_AllBlocks（同一人）
                    from scipy.stats import spearmanr as _sp18

                    # === Learning phase ===
                    _learn_motor_rois = [r for r in _ROI_GROUPS_18 if r.startswith('Motor')]
                    _learn_perc_rois  = [r for r in _ROI_GROUPS_18 if r.startswith('Perceptual')]
                    _learn_bgs = sorted(set(
                        k[3] for k in _ts_data if k[2] == 'Learning'
                    ))
                    for _bg in _learn_bgs:
                        for _cond18 in _CONDITIONS_18:
                            for _mr, _pr in zip(_learn_motor_rois, _learn_perc_rois):
                                _m_vecs = [_ts_data.get((s, _mr, 'Learning', _bg, _cond18))
                                           for s in subject_ids_18
                                           if (s, _mr, 'Learning', _bg, _cond18) in _ts_data]
                                _p_vecs = [_ts_data.get((s, _pr, 'Learning', _bg, _cond18))
                                           for s in subject_ids_18
                                           if (s, _pr, 'Learning', _bg, _cond18) in _ts_data]
                                if len(_m_vecs) < 3 or len(_m_vecs) != len(_p_vecs): continue
                                _m_mat = np.array(_m_vecs)
                                _p_mat = np.array(_p_vecs)
                                if _m_mat.shape != _p_mat.shape: continue
                                _n18 = _m_mat.shape[0]
                                for _ti, _t in enumerate(_times_ref):
                                    _r18, _p18 = _sp18(_m_mat[:, _ti], _p_mat[:, _ti])
                                    _se18 = (1 - float(_r18)**2) / np.sqrt(max(_n18-2, 1))
                                    _ts_rows_cross.append({
                                        'motor_roi': _mr, 'perceptual_roi': _pr,
                                        'phase': 'Learning', 'block_group': _bg,
                                        'condition': _cond18,
                                        'band': _ts_bname, 'time': float(_t),
                                        'r_cross': float(_r18), 'p_cross': float(_p18),
                                        'se_cross': float(_se18), 'sid_n': _n18
                                    })

                    # === Testing phase 跨受試者相關
                    # 分為三種：
                    #   MotorTest    : Motor ROI vs Perceptual ROI，兩者均來自 MotorTest
                    #   PerceptualTest: Motor ROI vs Perceptual ROI，兩者均來自 PerceptualTest
                    #   Testing      : Motor ROI(MotorTest) vs Perceptual ROI(PerceptualTest)（跨模態）
                    for _phase_label, _motor_ph, _percept_ph in [
                        ('MotorTest',       'MotorTest',      'MotorTest'),
                        ('PerceptualTest',  'PerceptualTest', 'PerceptualTest'),
                        ('Testing',         'MotorTest',      'PerceptualTest'),
                    ]:
                        for _cond18 in _CONDITIONS_18:
                            for _mr, _pr in zip(_learn_motor_rois, _learn_perc_rois):
                                _m_vecs = [_ts_data.get((s, _mr, _motor_ph, 'AllBlocks', _cond18))
                                           for s in subject_ids_18
                                           if (s, _mr, _motor_ph, 'AllBlocks', _cond18) in _ts_data]
                                _p_vecs = [_ts_data.get((s, _pr, _percept_ph, 'AllBlocks', _cond18))
                                           for s in subject_ids_18
                                           if (s, _pr, _percept_ph, 'AllBlocks', _cond18) in _ts_data]
                                if len(_m_vecs) < 3 or len(_m_vecs) != len(_p_vecs):
                                    continue
                                _m_mat = np.array(_m_vecs)
                                _p_mat = np.array(_p_vecs)
                                if _m_mat.shape != _p_mat.shape:
                                    continue
                                _n18 = _m_mat.shape[0]
                                for _ti, _t in enumerate(_times_ref):
                                    _r18, _p18 = _sp18(_m_mat[:, _ti], _p_mat[:, _ti])
                                    _se18 = (1 - float(_r18)**2) / np.sqrt(max(_n18-2, 1))
                                    _ts_rows_cross.append({
                                        'motor_roi': _mr, 'perceptual_roi': _pr,
                                        'phase': _phase_label, 'block_group': 'AllBlocks',
                                        'condition': _cond18,
                                        'band': _ts_bname, 'time': float(_t),
                                        'r_cross': float(_r18), 'p_cross': float(_p18),
                                        'se_cross': float(_se18), 'sid_n': _n18
                                    })
                    print(f"    跨受試者相關列數: {sum(1 for r in _ts_rows_cross if r['band']==_ts_bname)}")

                    # 儲存本 band 的跨受試者 CSV（在迴圈內，每個 band 各存一次）
                    _band_cross = [r for r in _ts_rows_cross if r['band'] == _ts_bname]
                    if _band_cross:
                        _ts_cross_df = pd.DataFrame(_band_cross)
                        _ts_cross_path = os.path.join(out_dir_18, f'timeseries_corr_{_ts_bname}_{lock_type_18}.csv')
                        _ts_cross_df.to_csv(_ts_cross_path, index=False)
                        print(f"  ✓ 跨受試者時間序列相關 CSV: {_ts_cross_path}")

                    # 儲存本 band 的單一受試者 CSV（flat 格式）
                    _band_single = [r for r in _ts_rows_single if r['band'] == _ts_bname]
                    if _band_single:
                        _flat = []
                        for _row in _band_single:
                            for _t, _pv in zip(_row['times'], _row['power']):
                                _flat.append({'sid':_row['sid'],'roi':_row['roi'],
                                              'phase':_row['phase'],'band':_row['band'],
                                              'block_group':_row['block_group'],
                                              'condition':_row['condition'],
                                              'tasktype':_row['tasktype'],
                                              'time':float(_t),'ersp':float(_pv)})
                        # Bug 修正：groupby key 加入 condition，避免 regular_high/
                        # random_high/random_low 三條件的功率被平均混在一起
                        _ts_sw = (
                            pd.DataFrame(_flat)
                            .groupby(['sid','roi','phase','band','condition','time'], as_index=False)['ersp']
                            .mean()
                        )
                        _ts_sp = os.path.join(out_dir_18, f'timeseries_single_sub_{_ts_bname}_{lock_type_18}.csv')
                        _ts_sw.to_csv(_ts_sp, index=False)
                        print(f"  ✓ 單一受試者時間序列 CSV: {_ts_sp}")
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

                            elec_out16 = r'C:\Experiment\Result\single_electrode'
                            os.makedirs(elec_out16, exist_ok=True)

                            import glob as _glob16
                            import numpy as _np16
                            import matplotlib.pyplot as _plt16
                            from mne_python_analysis.group_ersp_analysis import _load_h5_single_electrode

                            # triplet 模式走 triplet\h5，非 triplet 走 h5
                            is_triplet_mode = (
                                hasattr(current_epochs, 'metadata') and
                                current_epochs.metadata is not None and
                                'classification' in current_epochs.metadata.columns and
                                current_epochs.metadata['classification'].iloc[0] == 'triplet'
                            )
                            if is_triplet_mode:
                                _h5_search_dir = r'C:\Experiment\Result\triplet\h5'
                            else:
                                _h5_search_dir = r'C:\Experiment\Result\h5'

                            # 三組比較：Regular High vs Random Low / vs Random High / Random High vs Random Low
                            _triplet_pairs16 = [
                                ('regular_high', 'random_low',  'Regular High', 'Random Low'),
                                ('regular_high', 'random_high', 'Regular High', 'Random High'),
                                ('random_high',  'random_low',  'Random High',  'Random Low'),
                            ]
                            # 非 triplet 模式只畫一組
                            _nontriplet_pairs16 = [
                                ('Regular', 'Random', 'Regular', 'Random'),
                            ]
                            _pairs16 = _triplet_pairs16 if is_triplet_mode else _nontriplet_pairs16

                            def _single_elec_plot16(fp_l, fp_r, electrode16, disp_l, disp_r, suffix):
                                try:
                                    el, freqs_e, times_e, _nave_e = _load_h5_single_electrode(fp_l, electrode16)
                                    er, _, _, _          = _load_h5_single_electrode(fp_r, electrode16)
                                except Exception as ex:
                                    print(f"    ⚠ {os.path.basename(fp_l)}: {ex}")
                                    return
                                diff_e = el - er
                                x_min_e, x_max_e = -0.5, 0.5
                                t_mask_e = (times_e >= x_min_e) & (times_e <= x_max_e)
                                combined_e = _np16.concatenate([el[:, t_mask_e].ravel(), er[:, t_mask_e].ravel()])
                                vmax_e = _np16.percentile(_np16.abs(combined_e), 95)
                                vmax_d = _np16.percentile(_np16.abs(diff_e[:, t_mask_e].ravel()), 95)
                                if vmax_e < 1e-10: vmax_e = 1e-10
                                if vmax_d < 1e-10: vmax_d = 1e-10
                                lv_c = _np16.linspace(-vmax_e, vmax_e, 20)
                                lv_d = _np16.linspace(-vmax_d, vmax_d, 20)
                                base = os.path.basename(fp_l).replace('_ERSP.h5', '').replace(f'{subject_id}_Response_', '')
                                block_label16 = '_'.join(base.split('_')[:-1])
                                fig16, axes16 = _plt16.subplots(1, 3, figsize=(18, 5))
                                for ax16, data16, title16, lv16, vm16, cbl16 in [
                                    (axes16[0], el,     disp_l,                              lv_c, vmax_e, 'Power (dB)'),
                                    (axes16[1], er,     disp_r,                              lv_c, vmax_e, 'Power (dB)'),
                                    (axes16[2], diff_e, f'Diff ({disp_l} − {disp_r})', lv_d, vmax_d, 'Power Diff (dB)'),
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
                                    f'{subject_id} | Response-locked | {block_label16} | Electrode: {electrode16} | {disp_l} vs {disp_r}',
                                    fontsize=11, fontweight='bold'
                                )
                                _plt16.tight_layout()
                                out_fig16 = os.path.join(elec_out16,
                                    f'{subject_id}_response_{electrode16}_{block_label16}_{suffix}_comparison.png')
                                fig16.savefig(out_fig16, dpi=300, bbox_inches='tight')
                                _plt16.close(fig16)
                                print(f"    ✓ {disp_l} vs {disp_r}: {out_fig16}")

                            for electrode16 in electrodes16:
                                print(f"\n  電極: {electrode16}")
                                for lbl_l16, lbl_r16, disp_l16, disp_r16 in _pairs16:
                                    h5_files_l16 = sorted(_glob16.glob(
                                        os.path.join(_h5_search_dir, f'{subject_id}_Response_*_{lbl_l16}_ERSP.h5')))
                                    if not h5_files_l16:
                                        print(f"    ⚠ 找不到 {lbl_l16} h5 檔案，跳過")
                                        continue
                                    suffix16 = f"{lbl_l16}_vs_{lbl_r16}"
                                    for fp_l in h5_files_l16:
                                        fp_r = fp_l.replace(f'_{lbl_l16}_', f'_{lbl_r16}_')
                                        if not os.path.exists(fp_r):
                                            continue
                                        _single_elec_plot16(fp_l, fp_r, electrode16, disp_l16, disp_r16, suffix16)

                                # ── Epoch 4 vs Epoch 1（每個 triplet 條件）──
                                print(f"\n  [{electrode16}] Epoch 4 vs Epoch 1 比較")
                                _DISP16 = {
                                    'regular_high': 'Regular High',
                                    'random_high':  'Random High',
                                    'random_low':   'Random Low',
                                }
                                _e_pairs16 = (
                                    ['regular_high', 'random_high', 'random_low']
                                    if is_triplet_mode else ['Regular', 'Random']
                                )
                                for _cond16 in _e_pairs16:
                                    _fp_e1_16 = os.path.join(
                                        _h5_search_dir,
                                        f'{subject_id}_Response_Learning_Block7-11_{_cond16}_ERSP.h5')
                                    _fp_e4_16 = os.path.join(
                                        _h5_search_dir,
                                        f'{subject_id}_Response_Learning_Block22-26_{_cond16}_ERSP.h5')
                                    if not os.path.exists(_fp_e1_16) or not os.path.exists(_fp_e4_16):
                                        print(f"    ⚠ Epoch4vsEpoch1 {_cond16}: 檔案不存在，跳過")
                                        continue
                                    _disp16 = _DISP16.get(_cond16, _cond16)
                                    _suffix16_e = f'epoch4_vs_epoch1_{_cond16}'
                                    # 左=Epoch4, 右=Epoch1, Diff=Epoch4-Epoch1
                                    _single_elec_plot16(
                                        _fp_e4_16, _fp_e1_16, electrode16,
                                        f'Epoch 4 (Block22-26)\n{_disp16}',
                                        f'Epoch 1 (Block7-11)\n{_disp16}',
                                        _suffix16_e)

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
    matplotlib.rcParams["axes.unicode_minus"] = False  # 強制用 ASCII hyphen，避免 Glyph 8722 warning
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
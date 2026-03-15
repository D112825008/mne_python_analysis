"""
數據輸入輸出模組 - Data IO Module

處理EEG資料的載入與儲存功能。
版本: 3.0 - 整合所有檔案格式支援
"""

import os
import json
import pandas as pd
import mne
import numpy as np

# 用於將 Neuroscan 的通道命名轉換為 MNE 標準命名
CHANNEL_RENAME_MAP = {
    # 前額區 (Fp 不是 FP)
    'FP1': 'Fp1',
    'FP2': 'Fp2',
    # 中線電極 (z 小寫)
    'FZ': 'Fz',
    'FCZ': 'FCz',
    'CZ': 'Cz',
    'CPZ': 'CPz',
    'PZ': 'Pz',
    'POZ': 'POz',
    'OZ': 'Oz',
}

#QUICKCAP_CHANNEL_RENAME_MAP = {
    # 大小寫轉換
#    'FP1': 'Fp1',
#    'FP2': 'Fp2',
#    'FZ': 'Fz',
#    'FCZ': 'FCz',
#    'CZ': 'Cz',
#    'CPZ': 'CPz',
#    'PZ': 'Pz',
#    'OZ': 'Oz',
    # 舊命名轉新命名
#    'T3': 'T7',
#    'T4': 'T8',
#    'T5': 'P7',
#    'T6': 'P8',
    # 參考電極（如果使用 M 命名）
#    'M1': 'A1',
#    'M2': 'A2'
#}


def rename_channels_to_standard(raw):
    """
    將通道名稱重新命名為標準命名
    """
    channels_in_data = raw.ch_names
    rename_dict = {}
    
    for old_name, new_name in CHANNEL_RENAME_MAP.items():
        if old_name in channels_in_data:
            rename_dict[old_name] = new_name
    
    if rename_dict:
        print(f"\n重新命名通道以符合標準:")
        for old, new in rename_dict.items():
            print(f"  {old} → {new}")
        raw.rename_channels(rename_dict)
        return True
    return False


def load_bids_eeg(root_dir, subject_range=(1, 2)):
    """
    讀取BIDS格式的EEG資料。
    
    參數:
        root_dir (str): BIDS資料的根目錄路徑
        subject_range (tuple): 受試者編號範圍，例如(1, 165)表示從sub-001到sub-165
        
    返回:
        dict: 包含受試者EEG數據的字典
    """
    data_dict = {}

    for sub_num in range(subject_range[0], subject_range[1] + 1):
        sub_id = f"sub-{str(sub_num).zfill(3)}"

        try:
            eeg_set = os.path.join(root_dir, f"{sub_id}_task-restingstate_eeg.set")
            eeg_fdt = os.path.join(root_dir, f"{sub_id}_task-restingstate_eeg.fdt")
            eeg_json = os.path.join(root_dir, f"{sub_id}_task-restingstate_eeg.json")
            coordsystem_json = os.path.join(root_dir, f"{sub_id}_task-restingstate_coordsystem.json")
            channels_tsv = os.path.join(root_dir, f"{sub_id}_task-restingstate_channels.tsv")
            events_tsv = os.path.join(root_dir, f"{sub_id}_task-restingstate_events.tsv")
            electrodes_tsv = os.path.join(root_dir, f"{sub_id}_task-restingstate_electrodes.tsv")

            if not all(os.path.exists(f) for f in [eeg_set, eeg_fdt, eeg_json]):
                print(f"警告: {sub_id} 有缺檔案，跳過")
                continue

            raw = mne.io.read_raw_eeglab(eeg_set, preload=True)
            
            with open(eeg_json, 'r') as f:
                eeg_info = json.load(f)

            events_df = pd.read_csv(events_tsv, sep='\t') if os.path.exists(events_tsv) else None
            electrodes_df = pd.read_csv(electrodes_tsv, sep='\t') if os.path.exists(electrodes_tsv) else None
            coordsystem_df = pd.read_json(coordsystem_json, lines=True) if os.path.exists(coordsystem_json) else None
            channels_df = pd.read_csv(channels_tsv, sep='\t') if os.path.exists(channels_tsv) else None

            data_dict[sub_id] = {
                'raw': raw,
                'eeg_info': eeg_info,
                'channels': channels_df,
                'events': events_df,
                'electrodes_info': electrodes_df,
                'coordsystem_info': coordsystem_df
            }

            print(f"成功讀取 {sub_id} 的資料")
            
        except Exception as e:
            print(f"讀取 {sub_id} 時發生錯誤: {str(e)}")
            continue

    return data_dict


def load_cnt_file(file_path):
    """
    讀取Neuroscan .cnt格式的EEG資料。
    
    參數:
        file_path (str): CNT檔案路徑
        
    返回:
        tuple: (raw物件, events陣列) 或 (None, None)
    """
    try:
        print(f"\n正在讀取CNT檔案: {file_path}")
        
        raw = mne.io.read_raw_cnt(file_path, preload=True, verbose=True)
        
        try:
            events, event_id = mne.events_from_annotations(raw)
            sfreq = float(raw.info['sfreq'])
            events = events[np.argsort(events[:, 0])]
        except:
            events = None
            event_id = {}
        
        print("CNT 檔案讀取成功！")
        print(f"通道數: {len(raw.ch_names)}")
        print(f"取樣頻率: {raw.info['sfreq']} Hz")
        if events is not None:
            print(f"找到 {len(events)} 個事件")
            print(f"事件種類: {event_id}")
        
        # 重新命名通道
        rename_channels_to_standard(raw)
        
        # === 修改：Quick-Cap 32 自定義 montage 設定（替換原有的 Grael V2 設定） ===
        print("\n設定 Quick-Cap 32 電極位置...")
        
        # 步驟 1: Quick-Cap 32 通道名稱標準化
        #rename_dict = {}
        #for old_name, new_name in QUICKCAP_CHANNEL_RENAME_MAP.items():
        #    if old_name in raw.ch_names:
        #        rename_dict[old_name] = new_name
        
        #if rename_dict:
        #    print(f"  重新命名 {len(rename_dict)} 個通道以符合標準:")
        #    for old, new in rename_dict.items():
        #        print(f"    {old} → {new}")
        #    raw.rename_channels(rename_dict)
        
        # 步驟 1: 設定通道類型
        try:
            channel_types = {}
            for ch_name in raw.ch_names:
                if 'EOG' in ch_name.upper() or 'VEOG' in ch_name.upper() or 'HEOG' in ch_name.upper():
                    channel_types[ch_name] = 'eog'
                elif ch_name.upper() in ['TRIGGER', 'TRIG', 'STI', 'STIM']:
                    channel_types[ch_name] = 'stim'
                elif ch_name.upper() in ['DIOLE', 'REF', 'GND', 'GROUND', 'A1', 'A2', 'M1', 'M2']:
                    channel_types[ch_name] = 'misc'
                else:
                    channel_types[ch_name] = 'eeg'
            
            raw.set_channel_types(channel_types)
            
            eeg_channels = [ch for ch, tp in channel_types.items() if tp == 'eeg']
            print(f"  • EEG 通道: {len(eeg_channels)} 個")
        except Exception as e:
            print(f"  ⚠️ 設定通道類型時發生錯誤: {str(e)}")
        
        # 步驟 2: Quick-Cap 32 完整電極列表（根據官方配置圖）
        try:
            montage_1005 = mne.channels.make_standard_montage('standard_1005')
            all_positions = montage_1005.get_positions()['ch_pos']
            
            # 完整的命名映射
            old_naming_map = {
                'FP1': 'Fp1', 'FP2': 'Fp2',
                'FZ': 'Fz', 'FCZ': 'FCz', 'CZ': 'Cz',
                'CPZ': 'CPz', 'PZ': 'Pz', 'OZ': 'Oz',
                'T3': 'T7', 'T4': 'T8',
                'T5': 'P7', 'T6': 'P8',
                'M1': 'A1', 'M2': 'A2'
            }
            
            # 提取所有信號通道的座標
            quickcap_positions = {}
            all_signal_channels = [ch for ch in raw.ch_names 
                                  if raw.get_channel_types([ch])[0] in ['eeg', 'eog']]
            
            for ch in all_signal_channels:
                if ch in old_naming_map:
                    standard_name = old_naming_map[ch]
                    if standard_name in all_positions:
                        quickcap_positions[ch] = all_positions[standard_name]
                elif ch in all_positions:
                    quickcap_positions[ch] = all_positions[ch]
            
            print(f"  • 匹配到 {len(quickcap_positions)} 個電極座標")
            
            if quickcap_positions:
                custom_montage = mne.channels.make_dig_montage(
                    ch_pos=quickcap_positions,
                    coord_frame='head'
                )
                
                raw.set_montage(custom_montage, on_missing='warn')
                
                unmatched = set(all_signal_channels) - set(quickcap_positions.keys())
                if unmatched:
                    print(f"  ⚠️ 無法找到座標的電極: {unmatched}")
                else:
                    print(f"  ✓ 所有電極都已匹配")
                    
                print(f"  ✓ Quick-Cap 32 電極位置設定完成")
            else:
                print("  ⚠️ 未找到任何匹配的電極座標")
                
        except Exception as e:
            print(f"  ⚠️ 設定 montage 時發生錯誤: {str(e)}")
        
        return raw, events
            
    except Exception as e:  # ← 這行縮排必須與上面的 try 對齊（4 個空格）
        print(f"讀取 CNT 檔案時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None
        


def load_cdt_file(file_path):
    """
    讀取Curry .cdt格式的EEG資料。
    
    參數:
        file_path (str): CDT檔案路徑
        
    返回:
        tuple: (raw物件, events陣列) 或 (None, None)
    """
    try:
        print(f"\n正在讀取 CDT 檔案: {file_path}")
        
        try:
            import curryreader as cr
            
            curry_data = cr.read(file_path, plotdata=0, verbosity=2)
            
            eeg_data = curry_data['data']
            info_dict = curry_data['info']
            sfreq = info_dict['samplingfreq']
            ch_names = curry_data['labels']
            
            eeg_data = eeg_data.T
            
            info = mne.create_info(
                ch_names=ch_names,
                sfreq=sfreq,
                ch_types='eeg'
            )
            
            raw = mne.io.RawArray(eeg_data, info)
            
            events = None
            if 'events' in curry_data and len(curry_data['events']) > 0:
                curry_events = curry_data['events']
                events = np.column_stack([
                    curry_events[:, 0].astype(int),
                    np.zeros(len(curry_events), dtype=int),
                    curry_events[:, 1].astype(int)
                ])
                print(f"找到 {len(events)} 個 events/triggers")
            
            print("CDT 檔案讀取成功！")
            print(f"通道數: {len(ch_names)}")
            print(f"取樣頻率: {sfreq} Hz")
            print(f"資料長度: {eeg_data.shape[1] / sfreq:.2f} 秒")
            
        except ImportError:
            print("未安裝 curryreader，嘗試使用 MNE 讀取...")
            raw = mne.io.read_raw_curry(file_path, preload=True)
            
            events = None
            try:
                if len(raw.annotations) > 0:
                    events, event_id = mne.events_from_annotations(raw)
                    print(f"  - 找到 {len(events)} 個events")
                else:
                    events = np.array([])
            except:
                events = np.array([])
            
            print(f"成功讀取CDT檔案")
            print(f"  - 通道數: {len(raw.ch_names)}")
            print(f"  - 採樣率: {raw.info['sfreq']} Hz")
        
        # 設定 Grael V2 EEG Montage
        print("\n設定電極位置 (Grael V2 EEG)...")
        
        try:
            channel_types = {}
            for ch_name in raw.ch_names:
                if 'EOG' in ch_name.upper():
                    channel_types[ch_name] = 'eog'
                elif ch_name.upper() in ['TRIGGER', 'TRIG', 'STI', 'STIM']:
                    channel_types[ch_name] = 'stim'
                elif ch_name.upper() in ['DIOLE', 'REF', 'GND', 'GROUND']:
                    channel_types[ch_name] = 'misc'
                else:
                    channel_types[ch_name] = 'eeg'
            
            raw.set_channel_types(channel_types)
            
            eeg_channels = [ch for ch, tp in channel_types.items() if tp == 'eeg']
            print(f"  • EEG 通道: {len(eeg_channels)} 個")
            
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='ignore')
            
            extra_positions = {
                'F11': [-0.063, 0.066, -0.009],
                'F12': [0.063, 0.066, -0.009],
                'FT11': [-0.078, 0.024, -0.009],
                'FT12': [0.078, 0.024, -0.009],
            }
            
            current_montage = raw.get_montage()
            if current_montage is not None:
                ch_pos = current_montage.get_positions()['ch_pos'].copy()
                
                added_count = 0
                for ch_name, pos in extra_positions.items():
                    if ch_name in eeg_channels:
                        ch_pos[ch_name] = np.array(pos)
                        added_count += 1
                
                if added_count > 0:
                    new_montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
                    raw.set_montage(new_montage, on_missing='ignore')
                    print(f"  • 已設定 {added_count} 個額外電極位置")
            
            print("✓ 電極位置設定完成")
            
        except Exception as e:
            print(f"⚠️ 設定 montage 時發生錯誤: {str(e)}")
        
        return raw, events
        
    except Exception as e:
        print(f"讀取 CDT 檔案時發生錯誤: {str(e)}")
        return None, None


def load_neuroscan_64_to_32(file_path):
    """
    讀取 Neuroscan 64通道系統的資料，但使用32通道電極帽
    
    參數:
        file_path (str): CNT檔案路徑
        
    返回:
        tuple: (raw物件, events陣列)
    """
    # 實際使用的 32 個通道
    USED_CHANNELS_32 = [
        'FP1','FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'FC3', 'FCZ', 'FC4',
        'FT7', 'FT8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'CP3','CPZ', 'CP4',
        'TP7', 'TP8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
        'O1', 'OZ', 'O2',
        'M1', 'M2'
    ]
    
    try:
        print(f"\n正在讀取 Neuroscan 檔案: {file_path}")
        print("配置：64通道系統 → 32通道電極帽")
        
        raw = mne.io.read_raw_cnt(file_path, preload=True, verbose=True)
        
        try:
            events, event_id = mne.events_from_annotations(raw)
            events = events[np.argsort(events[:, 0])]
        except:
            events = None
            event_id = {}
        
        sfreq = float(raw.info['sfreq'])
        
        print(f"\n原始配置：")
        print(f"  通道數: {len(raw.ch_names)}")
        print(f"  取樣頻率: {sfreq} Hz")
        
        # 識別要保留和移除的通道
        channels_to_keep = []
        channels_to_remove = []
        non_eeg_channels = []
        
        for ch in raw.ch_names:
            ch_upper = ch.upper()
            
            if any(keyword in ch_upper for keyword in ['EOG', 'EKG', 'ECG', 'EMG', 'TRIGGER', 'TRIG', 'STI', 'STATUS']):
                non_eeg_channels.append(ch)
            elif any(used_ch.upper() == ch_upper for used_ch in USED_CHANNELS_32):
                channels_to_keep.append(ch)
            else:
                channels_to_remove.append(ch)
        
        print(f"\n通道分類：")
        print(f"  ✅ 保留 EEG 通道: {len(channels_to_keep)} 個")
        print(f"  ❌ 移除 EEG 通道: {len(channels_to_remove)} 個")
        if channels_to_remove:
            print(f"     {channels_to_remove}")
        print(f"  🔵 非 EEG 通道: {len(non_eeg_channels)} 個")
        if non_eeg_channels:
            print(f"     {non_eeg_channels}")
        
        # 設定通道類型
        print(f"\n設定通道類型...")
        channel_types = {}
        for ch in raw.ch_names:
            ch_upper = ch.upper()
            if 'VEOG' in ch_upper or 'HEOG' in ch_upper or 'EOG' in ch_upper:
                channel_types[ch] = 'eog'
            elif 'EKG' in ch_upper or 'ECG' in ch_upper:
                channel_types[ch] = 'ecg'
            elif 'EMG' in ch_upper:
                channel_types[ch] = 'emg'
            elif 'TRIGGER' in ch_upper or 'TRIG' in ch_upper or 'STI' in ch_upper or 'STATUS' in ch_upper:
                channel_types[ch] = 'stim'
            elif ch in channels_to_keep:
                if ch_upper in ['CB1', 'CB2']:
                    channel_types[ch] = 'misc'
                else:
                    channel_types[ch] = 'eeg'
            else:
                channel_types[ch] = 'misc'
        
        raw.set_channel_types(channel_types)
        
        # 移除未使用的通道
        if channels_to_remove:
            print(f"移除 {len(channels_to_remove)} 個未使用的通道...")
            raw.drop_channels(channels_to_remove)
        
        print(f"\n最終配置：")
        print(f"  剩餘通道數: {len(raw.ch_names)}")
        print(f"  EEG 通道: {len([ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == 'eeg'])} 個")
        
        # 重新命名通道
        rename_channels_to_standard(raw)
        
        # 設定 Montage
        print(f"\n設定電極位置...")
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='ignore')
            print("✓ 電極位置設定完成")
            
            montage_check = raw.get_montage()
            if montage_check:
                n_pos = len(montage_check.get_positions()['ch_pos'])
                print(f"  • 成功設定 {n_pos} 個通道的位置")
        except Exception as e:
            print(f"⚠️ 設定 montage 時發生錯誤: {str(e)}")
        
        return raw, events
        
    except Exception as e:
        print(f"讀取 Neuroscan 檔案時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def load_bdf_file(file_path):
    """
    讀取Biosemi BDF格式的EEG資料。
    
    參數:
        file_path (str): BDF檔案路徑
        
    返回:
        tuple: (raw物件, events陣列) 或 (None, None)
    """
    try:
        print(f"\n正在讀取BDF檔案: {file_path}")
        
        # 使用MNE讀取BDF檔案
        raw = mne.io.read_raw_bdf(file_path, preload=True)
        
        print(f"成功讀取BDF檔案")
        print(f"  - 通道數: {len(raw.ch_names)}")
        print(f"  - 採樣率: {raw.info['sfreq']} Hz")
        print(f"  - 資料長度: {raw.times[-1]:.2f} 秒")
        
        # 嘗試讀取events
        events = None
        try:
            events = mne.find_events(raw, stim_channel='Status')
            print(f"  - 找到 {len(events)} 個events")
        except:
            try:
                # 備選方案：從 annotations 讀取
                if len(raw.annotations) > 0:
                    events, event_id = mne.events_from_annotations(raw)
                    print(f"  - 從 annotations 找到 {len(events)} 個events")
                else:
                    events = np.array([])
                    print("  - 未找到events")
            except:
                events = np.array([])
                print("  - 未找到events或無stim channel")
        
        # 設定 Biosemi64 montage
        print("\n設定電極位置 (Biosemi64)...")
        
        try:
            # Biosemi 系統通常使用標準的 biosemi64 或 biosemi128 montage
            # 根據通道數自動選擇
            n_channels = len([ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == 'eeg'])
            
            if n_channels <= 64:
                montage_name = 'biosemi64'
            elif n_channels <= 128:
                montage_name = 'biosemi128'
            else:
                montage_name = 'biosemi256'
            
            montage = mne.channels.make_standard_montage(montage_name)
            raw.set_montage(montage, on_missing='warn')
            
            print(f"✓ 電極位置設定完成 (使用 {montage_name})")
            
        except Exception as e:
            print(f"⚠️ 設定 montage 時發生錯誤: {str(e)}")
        
        return raw, events
        
    except Exception as e:
        print(f"讀取 BDF 檔案時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def load_eeg_file(file_path):
    """
    自動偵測並讀取EEG檔案。
    
    參數:
        file_path (str): EEG檔案路徑
        
    返回:
        tuple: (raw物件, events陣列, 檔案格式)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.cnt':
        raw, events = load_cnt_file(file_path)
        file_format = 'CNT'
    elif ext == '.cdt':
        raw, events = load_cdt_file(file_path)
        file_format = 'CDT'
    elif ext == '.bdf':
        raw, events = load_bdf_file(file_path)
        file_format = 'BDF (Biosemi)'
    elif ext == '.set':
        try:
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
            try:
                events = mne.find_events(raw)
            except:
                events = np.array([])
            file_format = 'EEGLAB SET'
            print(f"成功讀取EEGLAB SET檔案")
        except Exception as e:
            print(f"讀取SET檔案時發生錯誤: {str(e)}")
            return None, None, None
    elif ext == '.fif':
        try:
            # 判斷是 epochs 還是 raw 檔案
            if '-epo.fif' in file_path or '_epo.fif' in file_path:
                # 讀取 epochs 檔案
                print("偵測到 Epochs 檔案，使用 mne.read_epochs() 讀取...")
                epochs = mne.read_epochs(file_path, preload=True)
                
                # 從 epochs 提取 events
                events = epochs.events
                
                file_format = 'FIF_EPOCHS'
                print(f"成功讀取 Epochs 檔案")
                print(f"  - Epochs 數量: {len(epochs)}")
                print(f"  - 通道數: {len(epochs.ch_names)}")
                print(f"  - 採樣率: {epochs.info['sfreq']} Hz")
                
                # 返回 epochs，而不是 raw
                return epochs, events, file_format
            else:
                # 讀取 raw 檔案
                raw = mne.io.read_raw_fif(file_path, preload=True)
                try:
                    events = mne.find_events(raw)
                except:
                    events = np.array([])
                file_format = 'FIF'
                print(f"成功讀取 Raw FIF 檔案")
                return raw, events, file_format
            
        except Exception as e:
            print(f"讀取FIF檔案時發生錯誤: {str(e)}")
            return None, None, None
    else:
        print(f"不支援的檔案格式: {ext}")
        return None, None, None
    
    return raw, events, file_format


def save_raw_data(raw, subject_id, prefix="processed_", suffix="raw"):
    """
    儲存處理後的 Raw 資料
    
    參數:
        raw (mne.io.Raw): 要儲存的 Raw 物件
        subject_id (str): 受試者 ID
        prefix (str): 檔名前綴
        suffix (str): 檔名後綴
    
    返回:
        list or str: 儲存的檔案路徑
    """
    print("\n" + "="*60)
    print("儲存 Raw 資料")
    print("="*60)
    
    # 詢問儲存格式
    print("\n請選擇儲存格式:")
    print("1. FIF (.fif) - MNE-Python 原生格式")
    print("2. EEGLAB (.set) - EEGLAB 格式")
    print("3. MAT (.mat) - MATLAB 格式")
    print("4. FIF + MAT 都儲存")

    while True:
        format_choice = input("\n請選擇 (1/2/3/4) [預設 1]: ").strip()
        if format_choice == "" or format_choice == "1":
            formats = ['fif']
            break
        elif format_choice == "2":
            formats = ['set']
            break
        elif format_choice == "3":
            formats = ['mat']
            break
        elif format_choice == "4":
            formats = ['fif', 'mat']
            break
        else:
            print("⚠️  請輸入 1, 2, 3, 或 4")

    saved_files = []

    # 儲存為 FIF
    if 'fif' in formats:
        default_fif = f"{prefix}sub_{subject_id}-{suffix}.fif"
        fif_filename = input(f"\n請輸入 FIF 檔名 [預設: {default_fif}]: ").strip() or default_fif
        if not (fif_filename.endswith('.fif') or fif_filename.endswith('.fif.gz')):
            fif_filename += '.fif'

        # 檢查檔案是否存在
        if os.path.exists(fif_filename):
            print(f"\n⚠️  FIF 檔案已存在: {fif_filename}")
            overwrite = input("是否覆蓋? (y/n) [預設 n]: ").strip().lower()
            if overwrite != 'y':
                print("✓ 取消儲存 FIF")
            else:
                raw.save(fif_filename, overwrite=True)
                print(f"已保存數據到檔案: {fif_filename}")
                saved_files.append(fif_filename)
        else:
            raw.save(fif_filename, overwrite=False)
            print(f"已保存數據到檔案: {fif_filename}")
            saved_files.append(fif_filename)

    # 儲存為 SET
    if 'set' in formats:
        default_set = f"{prefix}sub_{subject_id}-{suffix}.set"
        set_filename = input(f"\n請輸入 SET 檔名 [預設: {default_set}]: ").strip() or default_set
        if not set_filename.endswith('.set'):
            set_filename += '.set'

        # 檢查檔案是否存在
        if os.path.exists(set_filename):
            print(f"\n⚠️  SET 檔案已存在: {set_filename}")
            overwrite = input("是否覆蓋? (y/n) [預設 n]: ").strip().lower()
            if overwrite != 'y':
                print("✓ 取消儲存 SET")
            else:
                try:
                    raw.export(set_filename, fmt='eeglab', overwrite=True)
                    print(f"已保存數據到 EEGLAB 檔案: {set_filename}")
                    saved_files.append(set_filename)
                except Exception as e:
                    print(f"⚠️  儲存 SET 格式時發生錯誤: {str(e)}")
        else:
            try:
                raw.export(set_filename, fmt='eeglab', overwrite=False)
                print(f"已保存數據到 EEGLAB 檔案: {set_filename}")
                saved_files.append(set_filename)
            except Exception as e:
                print(f"⚠️  儲存 SET 格式時發生錯誤: {str(e)}")

    # 儲存為 MAT
    if 'mat' in formats:
        import scipy.io
        default_mat = f"{prefix}sub_{subject_id}-{suffix}.mat"
        mat_filename = input(f"\n請輸入 MAT 檔名 [預設: {default_mat}]: ").strip() or default_mat
        if not mat_filename.endswith('.mat'):
            mat_filename += '.mat'

        if os.path.exists(mat_filename):
            print(f"\n⚠️  MAT 檔案已存在: {mat_filename}")
            overwrite = input("是否覆蓋? (y/n) [預設 n]: ").strip().lower()
            if overwrite != 'y':
                print("✓ 取消儲存 MAT")
                mat_filename = None

        if mat_filename:
            try:
                scipy.io.savemat(mat_filename, {
                    'data': raw.get_data(),
                    'ch_names': raw.ch_names,
                    'sfreq': raw.info['sfreq'],
                    'times': raw.times,
                })
                print(f"已保存數據到 MATLAB 檔案: {mat_filename}")
                saved_files.append(mat_filename)
            except Exception as e:
                print(f"⚠️  儲存 MAT 格式時發生錯誤: {str(e)}")

    if saved_files:
        print(f"\n✓ 成功儲存 {len(saved_files)} 個檔案")
        return saved_files[0] if len(saved_files) == 1 else saved_files
    else:
        print("\n⚠️  未儲存任何檔案")
        return None


def save_epochs(epochs, subject_id, prefix="processed_", suffix="epo"):
    """
    儲存 Epochs 資料
    
    參數:
        epochs (mne.Epochs): 要儲存的 Epochs 物件
        subject_id (str): 受試者 ID
        prefix (str): 檔名前綴
        suffix (str): 檔名後綴
    
    返回:
        list or str: 儲存的檔案路徑
    """
    print("\n" + "="*60)
    print("儲存 Epochs 資料")
    print("="*60)
    
    # 詢問儲存格式
    print("\n請選擇儲存格式:")
    print("1. FIF (.fif) - MNE-Python 原生格式（保留 metadata）")
    print("2. EEGLAB (.set) - EEGLAB 格式")
    print("3. 兩者都儲存")
    
    while True:
        format_choice = input("\n請選擇 (1/2/3) [預設 1]: ").strip()
        if format_choice == "" or format_choice == "1":
            formats = ['fif']
            break
        elif format_choice == "2":
            formats = ['set']
            break
        elif format_choice == "3":
            formats = ['fif', 'set']
            break
        else:
            print("⚠️  請輸入 1, 2, 或 3")
    
    saved_files = []

    # 儲存為 FIF
    if 'fif' in formats:
        default_fif = f"{prefix}sub_{subject_id}-{suffix}.fif"
        fif_filename = input(f"\n請輸入 FIF 檔名 [預設: {default_fif}]: ").strip() or default_fif
        if fif_filename.endswith('.fif.gz'):
            pass
        elif fif_filename.endswith('.fif'):
            if not ('-epo.fif' in fif_filename or '_epo.fif' in fif_filename):
                fif_filename = fif_filename[:-4] + '-epo.fif'
        else:
            fif_filename += '-epo.fif'

        # 檢查檔案是否存在
        if os.path.exists(fif_filename):
            print(f"\n⚠️  FIF 檔案已存在: {fif_filename}")
            overwrite = input("是否覆蓋? (y/n) [預設 n]: ").strip().lower()
            if overwrite != 'y':
                print("✓ 取消儲存 FIF")
            else:
                epochs.save(fif_filename, overwrite=True)
                print(f"已保存 Epochs 到檔案: {fif_filename}")
                saved_files.append(fif_filename)
        else:
            epochs.save(fif_filename, overwrite=False)
            print(f"已保存 Epochs 到檔案: {fif_filename}")
            saved_files.append(fif_filename)

    # 儲存為 SET
    if 'set' in formats:
        default_set = f"{prefix}sub_{subject_id}-{suffix}.set"
        set_filename = input(f"\n請輸入 SET 檔名 [預設: {default_set}]: ").strip() or default_set
        if not set_filename.endswith('.set'):
            set_filename += '.set'
        
        # 檢查檔案是否存在
        if os.path.exists(set_filename):
            print(f"\n⚠️  SET 檔案已存在: {set_filename}")
            overwrite = input("是否覆蓋? (y/n) [預設 n]: ").strip().lower()
            if overwrite != 'y':
                print("✓ 取消儲存 SET")
            else:
                try:
                    epochs.export(set_filename, fmt='eeglab', overwrite=True)
                    print(f"已保存 Epochs 到 EEGLAB 檔案: {set_filename}")
                    saved_files.append(set_filename)
                except Exception as e:
                    print(f"⚠️  儲存 SET 格式時發生錯誤: {str(e)}")
                    print("   提示：EEGLAB 格式可能不支援所有 metadata")
        else:
            try:
                epochs.export(set_filename, fmt='eeglab', overwrite=False)
                print(f"已保存 Epochs 到 EEGLAB 檔案: {set_filename}")
                saved_files.append(set_filename)
            except Exception as e:
                print(f"⚠️  儲存 SET 格式時發生錯誤: {str(e)}")
                print("   提示：EEGLAB 格式可能不支援所有 metadata")
    
    if saved_files:
        print(f"\n✓ 成功儲存 {len(saved_files)} 個檔案")
        
        # 如果有 metadata，提醒使用者
        if hasattr(epochs, 'metadata') and epochs.metadata is not None and 'set' in formats:
            print("\n⚠️  注意：EEGLAB .set 格式不完整支援 metadata")
            print("   建議同時保留 .fif 格式以保存完整資訊")
        
        return saved_files[0] if len(saved_files) == 1 else saved_files
    else:
        print("\n⚠️  未儲存任何檔案")
        return None


def load_raw(file_path):
    """載入Raw數據"""
    return mne.io.read_raw_fif(file_path, preload=True)


def load_epochs(file_path):
    """載入Epochs數據"""
    return mne.read_epochs(file_path)


def select_file_interactively():
    """互動式選擇檔案"""
    print("\n請選擇資料格式:")
    print("1. BIDS格式 (資料夾)")
    print("2. CNT格式 (Neuroscan)")
    print("3. CDT格式 (Curry)")
    print("4. BDF格式 (Biosemi)")
    print("5. SET格式 (EEGLAB)")
    print("6. FIF格式 (MNE)")
    print("7. Neuroscan 64通道→32通道帽")
    
    choice = input("\n請輸入選項 (1-7): ").strip()
    
    if choice == '1':
        data_path = input("請輸入BIDS資料夾路徑: ").strip()
        return data_path, 'BIDS'
    elif choice in ['2', '3', '4', '5', '6']:
        file_path = input("請輸入檔案完整路徑: ").strip()
        format_map = {
            '2': 'CNT',
            '3': 'CDT',
            '4': 'BDF',
            '5': 'SET',
            '6': 'FIF'
        }
        return file_path, format_map[choice]
    elif choice == '7':
        file_path = input("請輸入Neuroscan CNT檔案完整路徑: ").strip()
        return file_path, 'NEUROSCAN_64_32'
    else:
        print("無效的選項")
        return None, None
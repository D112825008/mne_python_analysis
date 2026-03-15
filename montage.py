"""
電極配置模組 - Montage Module

處理EEG電極配置與座標系統。
版本: 2.0 - 新增 Grael V2 設定
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from pathlib import Path


def setup_bids_montage(raw, channels_tsv, coordsystem_json):
    """從BIDS格式的檔案設定EEG電極配置"""
    if os.path.exists(channels_tsv):
        try:
            channels_info = pd.read_csv(channels_tsv, sep='\t')
            print("成功讀取通道資訊")
            print(channels_info.head())
        except Exception as e:
            print(f'讀取{channels_tsv} 時發生錯誤: {e}')
            channels_info = None
    else:
        print(f"檔案{channels_tsv}不存在")
        channels_info = None

    with open(coordsystem_json, 'r') as f:
        coord_info = json.load(f)

    coord_system_mapping = {
        'CTF': 'ctf_head',
        'MNI': 'mni_tal',
        'RAS': 'ras',
        'Unknown': 'unknown'
    }

    coord_system = coord_info.get('EEGCoordinateSystem', 'Unknown')
    mne_coord_frame = coord_system_mapping.get(coord_system, 'unknown')

    electrodes_tsv = Path(channels_tsv).parent / Path(channels_tsv).name.replace('channels.tsv', 'electrodes.tsv')
    if electrodes_tsv.exists():
        electrodes_info = pd.read_csv(electrodes_tsv, sep='\t')

        ch_pos = {row['name']: np.array([row['x'], row['y'], row['z']]) for _, row in electrodes_info.iterrows()}

        montage = mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            coord_frame=mne_coord_frame
        )

        raw.set_montage(montage)
        raw.plot_sensors(show_names=True)
        plt.show()

        print("成功使用BIDS格式檔案設定montage")  
        print(f"座標系統: {coord_info.get('EEGCoordinateSystem', 'unknown')}")
        print(f"總通道數: {len(ch_pos)}")  

    else:
        print("未找到 electrodes.tsv檔案，將使用預設 BrainVision 64 通道設置")
        montage = mne.channels.make_standard_montage('easycap-M1')
        raw.set_montage(montage)
        
        print(raw.info)
        print(raw.ch_names)
    return raw


def setup_standard_montage(raw, montage_name='biosemi64'):
    """使用標準電極配置設定EEG montage（若已有montage則略過）"""
    current_montage = raw.get_montage()
    if current_montage is not None:
        print("略過 montage 設定（已存在於 raw 物件中）")
        montage_info = current_montage.get_positions()
        print(f"  • 現有 montage 通道數: {len(montage_info['ch_pos'])}")
        return raw

    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing='warn')
    print(f"成功設定標準 {montage_name} montage")
    return raw



def setup_quickcap_32_montage(raw):
    """
    設定 Compumedics Quick-Cap 32 電極位置（自定義精確版）
    只包含 Quick-Cap 32 實際使用的電極，基於 10-05 系統子集
    
    參數:
        raw (mne.io.Raw): 要設定montage的Raw物件
        
    返回:
        mne.io.Raw: 已設定montage的Raw物件
    """
    print("\n設定 Compumedics Quick-Cap 32 電極位置（自定義）...")
    
    try:
        # 步驟 1: Quick-Cap 32 完整電極列表（根據官方配置圖）
        quickcap_32_channels = [
            # 中線（6個）
            'FZ', 'FCZ', 'CZ', 'CPZ', 'PZ', 'OZ',
            # 左半球（包含所有變體）
            'FP1', 'F7', 'F3', 'FT7', 'FC3', 'T3', 'C3',  
            'TP7', 'CP3', 'T5', 'P3', 'O1',
            # 右半球（包含所有變體）
            'FP2', 'F8', 'F4', 'FT8', 'FC4', 'T4', 'C4',
            'TP8', 'CP4', 'T6', 'P4', 'O2',
            # 參考電極
            'A1', 'A2',
            # EOG 電極
            'HEOG', 'VEOG'
        ]
        
        print(f"  • Quick-Cap 32 標準配置參考: {len(set(quickcap_32_channels))} 個電極位置")
        
        # 步驟 2: 通道名稱標準化
#        channel_aliases = {
            # 大小寫轉換
#            'FPZ': 'Fpz', 'FP1': 'Fp1', 'FP2': 'Fp2',
#            'FZ': 'Fz', 'FCZ': 'FCz', 'CZ': 'Cz',
#            'CPZ': 'CPz', 'PZ': 'Pz', 'OZ': 'Oz',
            # 舊命名轉新命名（10-20 → 10-10）
#            'T3': 'T7',
#            'T4': 'T8',
#            'T5': 'P7',
#            'T6': 'P8',
            # 參考電極
#            'M1': 'A1',
#            'M2': 'A2'
#        }
        
#        rename_dict = {}
#        for ch in raw.ch_names:
#            if ch in channel_aliases:
#                rename_dict[ch] = channel_aliases[ch]
        
#        if rename_dict:
#            raw.rename_channels(rename_dict)
#            print(f"  • 重新命名了 {len(rename_dict)} 個通道")
        
        # 步驟 2: 設定通道類型
        channel_types = {}
        for ch_name in raw.ch_names:
            if 'EOG' in ch_name.upper() or 'VEOG' in ch_name.upper() or 'HEOG' in ch_name.upper():
                channel_types[ch_name] = 'eog'
            elif ch_name.upper() in ['TRIGGER', 'TRIG', 'STI', 'STIM']:
                channel_types[ch_name] = 'stim'
            elif ch_name.upper() in ['REF', 'GND', 'GROUND']:
                channel_types[ch_name] = 'misc'
            else:
                if raw.get_channel_types([ch_name])[0] != 'stim':
                    channel_types[ch_name] = 'eeg'
        
        if channel_types:
            raw.set_channel_types(channel_types)
        
        eeg_channels = [ch for ch in raw.ch_names 
                       if raw.get_channel_types([ch])[0] == 'eeg']
        eog_channels = [ch for ch in raw.ch_names
                        if raw.get_channel_types([ch])[0] == 'eog']
        print(f"  • EEG 通道: {len(eeg_channels)} 個")
        print(f"  • EOG 通道: {len(eog_channels)} 個")
        
        # 步驟 3: 從 standard_1005 提取 Quick-Cap 32 的電極位置
        montage_1005 = mne.channels.make_standard_montage('standard_1005')
        all_positions = montage_1005.get_positions()['ch_pos']
        
        # 舊命名 → 標準座標位置的對應
        old_naming_map = {
            'FP1': 'Fp1', 'FP2': 'Fp2',
            'FZ': 'Fz', 'FCZ': 'FCz', 'CZ': 'Cz',
            'CPZ': 'CPz', 'PZ': 'Pz', 'OZ': 'Oz',
            'T3': 'T7', 'T4': 'T8',
            'T5': 'P7', 'T6': 'P8',
            'M1': 'A1', 'M2': 'A2'
        }
        
        # 只處理 Quick-Cap 32 列表中的電極
        quickcap_positions = {}
        all_signal_channels = [ch for ch in raw.ch_names 
                              if raw.get_channel_types([ch])[0] in ['eeg', 'eog']]
        
        for ch in all_signal_channels:
            # 檢查是否在 Quick-Cap 32 列表中
            if ch in quickcap_32_channels:
                # 嘗試映射或直接匹配
                if ch in old_naming_map:
                    standard_name = old_naming_map[ch]
                    if standard_name in all_positions:
                        quickcap_positions[ch] = all_positions[standard_name]
                elif ch in all_positions:
                    quickcap_positions[ch] = all_positions[ch]
        
        print(f"  • 從資料中匹配到 {len(quickcap_positions)} 個 Quick-Cap 32 電極（含 EOG）")
        
        if quickcap_positions:
            # 步驟 4: 創建自定義 montage（使用原始命名）
            montage = mne.channels.make_dig_montage(
                ch_pos=quickcap_positions,
                coord_frame='head'
            )
            
            raw.set_montage(montage, on_missing='warn')
            
            # === 修正：只檢查 Quick-Cap 32 列表中的電極 ===
            matched_channels = set(quickcap_positions.keys())
            expected_in_data = [ch for ch in all_signal_channels if ch in quickcap_32_channels]
            unmatched = set(expected_in_data) - matched_channels
            
            if unmatched:
                print(f"  ⚠️ 以下 Quick-Cap 32 電極無法找到座標: {unmatched}")
            else:
                print(f"  ✓ 所有 Quick-Cap 32 電極都已匹配")
            
            # 檢查不在列表中的電極
            not_in_quickcap = set(all_signal_channels) - set(quickcap_32_channels)
            if not_in_quickcap:
                print(f"  ℹ️  以下電極不在 Quick-Cap 32 標準配置中（未設定座標）: {not_in_quickcap}")
            # === 修正結束 ===
            
            print(f"✓ Quick-Cap 32 電極位置設定完成（{len(quickcap_positions)} 個）")
        else:
            print("  ⚠️ 未找到任何匹配的 Quick-Cap 32 電極")
        
    except Exception as e:
        print(f"⚠️ 設定 Quick-Cap 32 montage 時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return raw

def rename_channels_to_standard(raw, standard='biosemi64'):
    """將通道名稱重命名為標準名稱"""
    standard_names = mne.channels.make_standard_montage(standard).ch_names
    
    rename_dict = {}
    for i in range(1, 33):
        if i <= len(standard_names):
            rename_dict[f"A{i}"] = standard_names[i-1]
    for i in range(1, 33):
        if i+32 <= len(standard_names):
            rename_dict[f"B{i}"] = standard_names[i+32-1]
    
    print("原始通道名稱:", raw.ch_names)
    print("映射字典:", rename_dict)
    
    valid_rename = {k: v for k, v in rename_dict.items() if k in raw.ch_names}
    print("有效的映射字典:", valid_rename)

    if valid_rename:
        raw.rename_channels(valid_rename)
        print("新的通道名稱:", raw.ch_names)
    else:
        print("警告：找不到匹配的通道名稱可以重新命名")
    
    return raw
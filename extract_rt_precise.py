"""
精確對齊：從行為資料提取 RT 並與 EEG epochs 精確對應

這個腳本會：
1. 從 epochs.metadata 中讀取每個保留的 trial 的 block 和 trial_in_block
2. 從行為資料中精確找到對應的 RT
3. 確保完美對齊（數量一致）
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def build_rt_lookup_table(behavioral_csv):
    """
    從行為資料建立 RT 查找表
    
    Parameters
    ----------
    behavioral_csv : str
        行為資料 CSV 檔案路徑
        
    Returns
    -------
    rt_lookup : dict
        字典格式：{(block, trial_in_block): RT_in_ms}
    subject_id : str
        受試者 ID
    """
    
    print(f"\n{'='*70}")
    print(f"建立 RT 查找表")
    print(f"{'='*70}")
    print(f"檔案: {os.path.basename(behavioral_csv)}")
    
    # 讀取 CSV
    df = pd.read_csv(behavioral_csv)
    
    # 提取受試者 ID
    subject_id = df['participant'].dropna().iloc[0] if 'participant' in df.columns else 'unknown'
    print(f"受試者: {subject_id}")
    
    rt_lookup = {}
    
    # === 處理 Practice blocks (Block 2-6) ===
    df_practice = df[df['practice_seq_files'].notna() & df['key_resp.rt'].notna()].copy()
    if len(df_practice) > 0:
        print(f"\n處理 Practice blocks...")
        # practice_blocks.thisN 是 block 編號（0-4對應 Block 2-6）
        df_practice['block'] = df_practice['practice_blocks.thisN'] + 2
        
        # 計算每個 trial 在其 block 中的順序
        df_practice['trial'] = df_practice.groupby('practice_seq_files').cumcount()
        
        for _, row in df_practice.iterrows():
            if pd.notna(row['block']) and pd.notna(row['trial']):
                block = int(row['block'])
                trial = int(row['trial'])
                rt_ms = row['key_resp.rt'] * 1000
                rt_lookup[(block, trial)] = rt_ms
        
        print(f"  ✓ {len(df_practice)} trials")
    
    # === 處理 Learning blocks (Block 7-26) ===
    df_learning = df[df['learning_seq_files'].notna() & df['key_resp.rt'].notna()].copy()
    if len(df_learning) > 0:
        print(f"\n處理 Learning blocks...")
        # learning_trials.thisN 是 block 編號（0-19對應 Block 7-26）
        df_learning['block'] = df_learning['learning_trials.thisN'] + 7
        
        # 計算每個 trial 在其 block 中的順序
        # 按 learning_seq_files 分組，計算組內順序
        df_learning['trial'] = df_learning.groupby('learning_seq_files').cumcount()
        
        for _, row in df_learning.iterrows():
            if pd.notna(row['block']) and pd.notna(row['trial']):
                block = int(row['block'])
                trial = int(row['trial'])
                rt_ms = row['key_resp.rt'] * 1000
                rt_lookup[(block, trial)] = rt_ms
        
        print(f"  ✓ {len(df_learning)} trials")
    
    # === 處理 Testing blocks (Block 27-34) ===
    df_test = df[df['test_seq_files'].notna() & df['key_resp.rt'].notna()].copy()
    if len(df_test) > 0:
        print(f"\n處理 Testing blocks...")
        # combined_testing_trials.thisN 是 block 編號（0-7對應 Block 27-34）
        df_test['block'] = df_test['combined_testing_trials.thisN'] + 27
        
        # 計算每個 trial 在其 block 中的順序
        df_test['trial'] = df_test.groupby('test_seq_files').cumcount()
        
        for _, row in df_test.iterrows():
            if pd.notna(row['block']) and pd.notna(row['trial']):
                block = int(row['block'])
                trial = int(row['trial'])
                rt_ms = row['key_resp.rt'] * 1000
                rt_lookup[(block, trial)] = rt_ms
        
        print(f"  ✓ {len(df_test)} trials")
    
    print(f"\n總計: {len(rt_lookup)} 個 (block, trial) → RT 對應")
    
    return rt_lookup, subject_id


def extract_rt_from_epochs_metadata(epochs, rt_lookup, behavioral_csv=None):
    """
    從 epochs.metadata 提取對應的 RT
    
    Parameters
    ----------
    epochs : mne.Epochs
        包含 metadata 的 epochs
    rt_lookup : dict or str
        RT 查找表，或行為資料 CSV 檔案路徑
    behavioral_csv : str, optional
        如果 rt_lookup 是檔案路徑，需要提供此參數
        
    Returns
    -------
    rt_array : ndarray
        對應的 RT 陣列（秒）
    matched_count : int
        成功匹配的數量
    """
    
    print(f"\n{'='*70}")
    print(f"從 epochs.metadata 提取 RT")
    print(f"{'='*70}")
    
    # 檢查 metadata
    if not hasattr(epochs, 'metadata') or epochs.metadata is None:
        raise ValueError("Epochs 沒有 metadata！")
    
    metadata = epochs.metadata
    print(f"Epochs 數量: {len(metadata)}")
    print(f"Metadata 欄位: {list(metadata.columns)}")
    
    # 檢查必要欄位
    if 'block' not in metadata.columns:
        raise ValueError("Metadata 中沒有 'block' 欄位！")
    
    if 'trial_in_block' not in metadata.columns:
        raise ValueError("Metadata 中沒有 'trial_in_block' 欄位！")
    
    # 如果 rt_lookup 是檔案路徑，先建立查找表
    if isinstance(rt_lookup, str):
        rt_lookup, _ = build_rt_lookup_table(rt_lookup)
    
    # 提取 RT
    rt_array = []
    matched_count = 0
    missing_trials = []
    
    for idx, row in metadata.iterrows():
        block = int(row['block'])
        trial = int(row['trial_in_block'])
        
        key = (block, trial)
        
        if key in rt_lookup:
            rt_ms = rt_lookup[key]
            rt_array.append(rt_ms)
            matched_count += 1
        else:
            # 找不到對應的 RT
            rt_array.append(np.nan)
            missing_trials.append(key)
    
    rt_array = np.array(rt_array)
    
    # 報告結果
    print(f"\n匹配結果:")
    print(f"  成功匹配: {matched_count}/{len(metadata)} ({matched_count/len(metadata)*100:.1f}%)")
    
    if len(missing_trials) > 0:
        print(f"  ⚠️  缺失 RT: {len(missing_trials)} trials")
        if len(missing_trials) <= 10:
            print(f"  缺失的 (block, trial): {missing_trials}")
        else:
            print(f"  缺失的 (block, trial): {missing_trials[:10]} ...")
    
    # 統計
    rt_valid = rt_array[~np.isnan(rt_array)]
    if len(rt_valid) > 0:
        print(f"\nRT 統計:")
        print(f"  範圍: {rt_valid.min():.1f} - {rt_valid.max():.1f} ms")
        print(f"  平均: {rt_valid.mean():.1f} ± {rt_valid.std():.1f} ms")
    
    return rt_array / 1000.0, matched_count  # 轉換為秒


def add_rt_to_epochs_from_behavioral(epochs, behavioral_csv, overwrite=False):
    """
    直接從行為資料加入 RT 到 epochs.metadata
    
    Parameters
    ----------
    epochs : mne.Epochs
        要加入 RT 的 epochs
    behavioral_csv : str
        行為資料 CSV 檔案路徑
    overwrite : bool
        如果已經有 'rt' 欄位，是否覆蓋
        
    Returns
    -------
    epochs : mne.Epochs
        已加入 RT 的 epochs
    success : bool
        是否成功
    """
    
    print(f"\n{'='*70}")
    print(f"把 RT 加入 epochs.metadata（精確對齊）")
    print(f"{'='*70}")
    
    # 檢查是否已有 RT
    if hasattr(epochs, 'metadata') and epochs.metadata is not None:
        if 'rt' in epochs.metadata.columns and not overwrite:
            print("⚠️  metadata 中已經有 'rt' 欄位")
            return epochs, False
    
    # 建立 RT 查找表
    rt_lookup, subject_id = build_rt_lookup_table(behavioral_csv)
    
    # 提取對應的 RT
    rt_array, matched_count = extract_rt_from_epochs_metadata(epochs, rt_lookup)
    
    # 檢查匹配率
    match_rate = matched_count / len(epochs)
    
    if match_rate < 0.95:
        print(f"\n⚠️  警告：匹配率只有 {match_rate*100:.1f}%")
        print(f"  可能原因：")
        print(f"    1. 行為資料和 EEG 資料不是來自同一個實驗")
        print(f"    2. metadata 中的 block/trial 編號有誤")
        
        confirm = input("\n仍要繼續？(y/n): ").strip().lower()
        if confirm != 'y':
            return epochs, False
    
    # 加入 metadata
    epochs.metadata['rt'] = rt_array
    
    print(f"\n✓ RT 已成功加入 epochs.metadata")
    print(f"  匹配率: {match_rate*100:.1f}%")
    print(f"  有效 RT: {np.sum(~np.isnan(rt_array))}/{len(epochs)}")
    
    return epochs, True


def save_rt_for_epochs(epochs, behavioral_csv, output_file):
    """
    提取對應的 RT 並存成檔案
    
    Parameters
    ----------
    epochs : mne.Epochs
        包含 metadata 的 epochs
    behavioral_csv : str
        行為資料 CSV 檔案路徑
    output_file : str
        輸出檔案路徑
    """
    
    # 建立 RT 查找表
    rt_lookup, subject_id = build_rt_lookup_table(behavioral_csv)
    
    # 提取對應的 RT
    rt_array, matched_count = extract_rt_from_epochs_metadata(epochs, rt_lookup)
    
    # 轉換為毫秒
    rt_ms = rt_array * 1000
    
    # 儲存
    np.savetxt(output_file, rt_ms, fmt='%.3f')
    
    print(f"\n✓ 已儲存: {output_file}")
    print(f"  數量: {len(rt_ms)}")
    print(f"  有效: {np.sum(~np.isnan(rt_ms))}")
    
    return output_file


if __name__ == '__main__':
    import sys
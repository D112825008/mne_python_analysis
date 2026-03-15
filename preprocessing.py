"""
預處理模組 - Preprocessing Module

包含EEG數據預處理相關功能。
版本: 3.0 - 修正前處理流程順序，符合標準 EEG 處理規範
"""
import numpy as np
import mne
from mne_python_analysis.signal_processing import (
    apply_average_reference,
    apply_linked_mastoid_reference,
    apply_single_electrode_reference
)


def interactive_marking_bad_segments(raw):
    """
    互動式標記壞段落（類似EEGLAB的手動artifact標記功能）。
    
    參數:
        raw (mne.io.Raw): 要標記的Raw物件
    
    返回:
        mne.io.Raw: 標記後的Raw物件
    """
    print("\n" + "="*60)
    print("互動式標記壞段落 (Interactive Bad Segment Marking)")
    print("="*60)
    print("\n使用說明：")
    print("1. 滑鼠左鍵拖曳選擇時間區間")
    print("2. 按 'a' 鍵新增標記")
    print("3. 輸入描述（建議用 'BAD_' 開頭，如 'BAD_artifact'）")
    print("4. 點擊通道名稱可標記壞通道")
    print("5. 按 '?' 查看所有快捷鍵")
    print("6. 關閉視窗完成標記")
    print("\n快捷鍵提示：")
    print("  ← → : 向左/右移動")
    print("  ↑ ↓ : 增加/減少顯示通道數")
    print("  + - : 放大/縮小振幅")
    print("  a   : 新增標記")
    print("="*60 + "\n")
    
    input("按 Enter 開啟互動視窗...")
    
    # 開啟互動式標記視窗
    try:
        fig = raw.plot(
            scalings='auto',
            n_channels=30,
            block=True,
            title='互動式標記壞段落 - 拖曳選擇後按 a 鍵標記'
        )
    except Exception as e:
        print(f"開啟互動視窗時發生錯誤: {str(e)}")
        return raw
    
    # 檢查是否有新增標記
    if len(raw.annotations) > 0:
        print(f"\n已標記 {len(raw.annotations)} 個區段：")
        for i, ann in enumerate(raw.annotations):
            print(f"  {i+1}. {ann['description']}: {ann['onset']:.2f}s - "
                  f"{ann['onset'] + ann['duration']:.2f}s "
                  f"({ann['duration']:.2f}s)")
        
        # 統計 BAD 標記
        bad_annotations = [ann for ann in raw.annotations 
                          if ann['description'].startswith('BAD')]
        if bad_annotations:
            total_bad_duration = sum([ann['duration'] for ann in bad_annotations])
            print(f"\n總共標記了 {len(bad_annotations)} 個壞段落")
            print(f"壞段落總時長: {total_bad_duration:.2f} 秒")
            
            # 顯示壞段落比例
            total_time = raw.times[-1]
            print(f"壞段落比例: {total_bad_duration/total_time*100:.2f}%")
    else:
        print("\n未標記任何區段")
    
    return raw


def preprocess_raw_data(raw, subject_id, electrode_coords_csv = None):
    """
    執行 EEG 標準前處理流程（互動式，可導航）
    
    標準處理順序：
    1. 確認 montage 設定
    2. 偵測/標記壞通道（自動偵測 or 手動輸入）← 整合選項5
    3. 重採樣 ← 選項6
    4. 濾波 ← 選項7
    5. 重參考 ← 選項8
    6. 插值壞通道
    
    導航指令：
    - 輸入 'b' 或 'back': 返回上一步驟
    - 輸入 'q' 或 'quit': 退出到主選單
    
    參數:
        raw (mne.io.Raw): 原始 EEG 資料
        subject_id (str): 受試者ID
        electrode_coords_csv (str): 電極座標檔案路徑（選用）
    
    返回:
        tuple: (處理後的 raw 物件, 處理資訊字典)
              如果用戶選擇退出，返回 (None, None)
    """
    print("\n" + "="*60)
    print("EEG 標準前處理流程（互動式）")
    print("="*60)
    print("\n導航提示：")
    print("  • 輸入 'b' 或 'back': 返回上一步驟")
    print("  • 輸入 'q' 或 'quit': 退出到主選單")
    print("="*60)

    try:
        # === 步驟 0: 檢查資料類型 ===
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError("輸入必須是 MNE Raw物件，目前是: " + str(type(raw)))
            
        # 複製 raw 物件以避免修改原始數據
        raw = raw.copy()
        n_channels = len(raw.ch_names)
        print(f"\n原始通道數: {n_channels}")
        
        # 初始化變數
        l_freq = None
        h_freq = None
        interpolated_bads = []
        
        # 步驟控制
        current_step = 1
        max_step = 6
        
        while current_step <= max_step:
            
            # ============================================================
            # 步驟 1: 確認 Montage 設定
            # ============================================================
            if current_step == 1:
                print("\n" + "="*60)
                print("步驟 1/6: 確認電極位置設定 (Montage)")
                print("="*60)
                
                current_montage = raw.get_montage()
                if current_montage is None:
                    print("⚠️  尚未設定 montage，使用標準 biosemi64")
                    montage = mne.channels.make_standard_montage('biosemi64')
                    raw.set_montage(montage, on_missing='warn')
                    print("✓ 已設定標準電極位置")
                else:
                    print("✓ Montage 已存在，略過此步驟")
                
                # 導航
                nav = input("\n按 Enter 繼續，或輸入 'q' 退出: ").strip().lower()
                if nav in ['q', 'quit']:
                    print("✓ 退出前處理流程")
                    return None, None
                
                current_step = 2
            
            # ============================================================
            # 步驟 2: 偵測/標記壞通道（整合選項5）
            # ============================================================
            elif current_step == 2:
                print("\n" + "="*60)
                print("步驟 2/6: 偵測/標記壞通道 (Bad Channels)")
                print("="*60)
                
                # 顯示目前的壞通道
                if raw.info['bads']:
                    print(f"目前標記的壞通道: {raw.info['bads']}")
                else:
                    print("目前沒有標記壞通道")
                
                # 詢問標記方式
                print("\n請選擇標記壞通道的方式:")
                print("1. 自動偵測（基於統計方法）")
                print("2. 手動輸入")
                print("3. 跳過")
                
                while True:
                    bad_choice = input("\n請輸入選項 (1-3) [預設 3] / b 返回 / q 退出: ").strip().lower()
                    
                    # 導航
                    if bad_choice in ['q', 'quit']:
                        print("✓ 退出前處理流程")
                        return None, None
                    elif bad_choice in ['b', 'back']:
                        current_step = 1
                        break
                    
                    # 選項處理
                    if bad_choice == "" or bad_choice == "3":
                        print("✓ 略過標記壞通道")
                        current_step = 3
                        break
                    
                    elif bad_choice == "1":
                        # 自動偵測壞通道
                        print("\n執行自動偵測壞通道...")
                        print("使用方法：基於振幅統計（標準差 > 5倍中位數）")
                        
                        try:
                            # 計算每個通道的標準差
                            data = raw.get_data()
                            channel_stds = data.std(axis=1)
                            median_std = np.median(channel_stds)
                            
                            # 找出異常通道（標準差 > 5倍中位數）
                            threshold = 5 * median_std
                            bad_indices = np.where(channel_stds > threshold)[0]
                            
                            if len(bad_indices) > 0:
                                auto_bad_channels = [raw.ch_names[i] for i in bad_indices]
                                print(f"\n偵測到 {len(auto_bad_channels)} 個可能的壞通道:")
                                print(f"  {auto_bad_channels}")
                                
                                confirm = input("\n是否標記這些通道為壞通道? (y/n) [預設 y]: ").strip().lower()
                                if confirm != 'n':
                                    raw.info['bads'].extend(auto_bad_channels)
                                    print(f"✓ 已標記: {auto_bad_channels}")
                                else:
                                    print("✓ 略過自動標記")
                            else:
                                print("✓ 未偵測到明顯的壞通道")
                        
                        except Exception as e:
                            print(f"⚠️  自動偵測失敗: {str(e)}")
                            print("請改用手動輸入方式")
                        
                        current_step = 3
                        break
                    
                    elif bad_choice == "2":
                        # 手動輸入壞通道
                        print("\n【手動標記壞通道】")
                        print(f"可用電極: {', '.join(raw.ch_names[:10])}... (共 {len(raw.ch_names)} 個)")
                        
                        bad_input = input("\n請輸入壞通道名稱（空格分隔，例如: Fp1 Fp2）: ").strip()
                        
                        if bad_input:
                            manual_bad_channels = bad_input.split()
                            valid_bads = [ch for ch in manual_bad_channels if ch in raw.ch_names]
                            invalid_bads = [ch for ch in manual_bad_channels if ch not in raw.ch_names]
                            
                            if valid_bads:
                                raw.info['bads'].extend(valid_bads)
                                print(f"✓ 已標記: {valid_bads}")
                            
                            if invalid_bads:
                                print(f"⚠️  找不到以下電極: {invalid_bads}")
                        else:
                            print("✓ 未輸入任何通道")
                        
                        current_step = 3
                        break
                    
                    else:
                        print("⚠️  無效的選項，請輸入 1-3")
                
                if raw.info['bads']:
                    print(f"\n✓ 壞通道偵測完成，共標記 {len(raw.info['bads'])} 個通道: {raw.info['bads']}")
                else:
                    print("\n✓ 壞通道偵測完成，沒有標記任何通道")
            
            # ============================================================
            # 步驟 3: 重採樣
            # ============================================================
            elif current_step == 3:
                print("\n" + "="*60)
                print("步驟 3/6: 重採樣 (Resampling)")
                print("="*60)

                original_sfreq = raw.info['sfreq']
                print(f"原始採樣率: {original_sfreq} Hz")
                
                while True:
                    resample_choice = input(f"\n是否要重採樣? (y/n) [預設 n] / b 返回 / q 退出: ").strip().lower()
                    
                    # 導航
                    if resample_choice in ['q', 'quit']:
                        print("✓ 退出前處理流程")
                        return None, None
                    elif resample_choice in ['b', 'back']:
                        current_step = 2
                        break
                    
                    # 選項處理
                    if resample_choice == 'y':
                        while True:
                            target_input = input(f"請輸入目標採樣率 (Hz) [建議 250-1000] / b 返回: ").strip().lower()
                            
                            if target_input in ['b', 'back']:
                                break  # 回到重採樣選擇
                            
                            try:
                                target_sfreq = float(target_input)
                                if 100 <= target_sfreq <= 10000:
                                    if target_sfreq == original_sfreq:
                                        print(f"✓ 採樣率已經是 {target_sfreq} Hz，無需重採樣")
                                    else:
                                        print(f"執行重採樣到 {target_sfreq} Hz...")
                                        raw.resample(sfreq=target_sfreq)
                                        print(f"✓ 重採樣完成: {original_sfreq} Hz → {target_sfreq} Hz")
                                    current_step = 4
                                    break
                                else:
                                    print("⚠️  採樣率應在 100-10000 Hz 之間，請重新輸入")
                            except ValueError:
                                print("⚠️  請輸入有效的數字")
                        
                        if current_step == 4:
                            break
                    
                    elif resample_choice == "" or resample_choice == "n":
                        print("✓ 略過重採樣")
                        current_step = 4
                        break
                    else:
                        print("⚠️  無效的輸入，請輸入 y/n")
            
            # ============================================================
            # 步驟 4: 濾波
            # ============================================================
            elif current_step == 4:
                print("\n" + "="*60)
                print("步驟 4/6: 濾波 (Filtering)")
                print("="*60)
                
                # High-pass Filter
                print("\n【High-pass Filter 設定】")
                print("用途：移除低頻漂移和DC offset")
                print("建議範圍：0.1 - 1.0 Hz")
                print("常用值：0.1 Hz (保留較多低頻), 0.5 Hz (一般用途), 1.0 Hz (ICA專用)")
                
                while True:
                    hp_input = input("\n請輸入 High-pass 頻率 (Hz) [預設 0.1] / b 返回 / q 退出: ").strip().lower()
                    
                    # 導航
                    if hp_input in ['q', 'quit']:
                        print("✓ 退出前處理流程")
                        return None, None
                    elif hp_input in ['b', 'back']:
                        current_step = 3
                        break
                    
                    # 選項處理
                    if hp_input == "":
                        l_freq = 0.1
                        print(f"使用預設值: {l_freq} Hz")
                        break
                    
                    try:
                        l_freq = float(hp_input)
                        if 0 < l_freq <= 5:
                            print(f"設定 High-pass filter: {l_freq} Hz")
                            break
                        else:
                            print("⚠️  頻率應在 0-5 Hz 之間，請重新輸入")
                    except ValueError:
                        print("⚠️  請輸入有效的數字")
                
                if current_step == 3:
                    continue
                
                # Low-pass Filter
                print("\n【Low-pass Filter 設定】")
                print("用途：移除高頻雜訊和肌電干擾")
                print("建議範圍：30 - 100 Hz")
                print("常用值：30 Hz (一般分析), 40 Hz (包含gamma), 100 Hz (寬頻分析)")
                
                while True:
                    lp_input = input("\n請輸入 Low-pass 頻率 (Hz) [預設 40] / b 返回: ").strip().lower()
                    
                    if lp_input in ['b', 'back']:
                        # 返回重新設定 high-pass
                        break
                    
                    if lp_input == "":
                        h_freq = 40
                        print(f"使用預設值: {h_freq} Hz")
                        break
                    
                    try:
                        h_freq = float(lp_input)
                        if l_freq < h_freq <= 200:
                            print(f"設定 Low-pass filter: {h_freq} Hz")
                            break
                        else:
                            print(f"⚠️  頻率應在 {l_freq}-200 Hz 之間，請重新輸入")
                    except ValueError:
                        print("⚠️  請輸入有效的數字")
                
                if lp_input in ['b', 'back']:
                    continue  # 重新設定 high-pass
                
                # 執行 Band-pass 濾波
                print(f"\n執行 Band-pass 濾波: {l_freq} - {h_freq} Hz...")
                raw.filter(l_freq=l_freq, h_freq=h_freq)
                print("✓ Band-pass 濾波完成")
                
                # Notch Filter
                while True:
                    notch_choice = input("\n是否要應用 Notch 濾波器 (60 Hz 和 100 Hz)? (y/n) [預設 y] / b 返回: ").strip().lower()
                    
                    if notch_choice in ['b', 'back']:
                        # 返回重新設定濾波
                        break
                    
                    if notch_choice != 'n':
                        print("執行 Notch 濾波...")
                        raw.notch_filter(freqs=[60])
                        print("✓ Notch 濾波完成 (60 Hz)")
                        current_step = 5
                        break
                    else:
                        print("✓ 略過 Notch 濾波")
                        current_step = 5
                        break
                
                if notch_choice in ['b', 'back']:
                    continue
            
            # ============================================================
            # 步驟 5: 重參考 (Re-referencing)
            # ============================================================
            elif current_step == 5:
                print("\n" + "="*60)
                print("步驟 5/6: 重參考 (Re-referencing)")
                print("="*60)
                
                print("\n請選擇 Reference 類型:")
                print("1. Linked Mastoids (M1/M2 或 A1/A2)")
                print("2. 單一電極 Reference")
                print("3. Average Reference")
                print("4. 跳過")
                
                while True:
                    ref_choice = input("\n請輸入選項 (1-4) [預設 3] / b 返回 / q 退出: ").strip().lower()
                    
                    # 導航
                    if ref_choice in ['q', 'quit']:
                        print("✓ 退出前處理流程")
                        return None, None
                    elif ref_choice in ['b', 'back']:
                        current_step = 4
                        break
                    
                    # 選項處理
                    if ref_choice == "" or ref_choice == "3":
                        # Average Reference
                        print("\n【Average Reference 設定】")
                        
                        # 詢問是否排除特定電極
                        exclude_choice = input("是否要排除特定電極？(y/n) [預設 y]: ").strip().lower()
                        
                        if exclude_choice != 'n':
                            print("\n預設排除: HEOG, VEOG, A1, A2")
                            custom_exclude = input("使用預設排除電極？(y/n) [預設 y]: ").strip().lower()
                            
                            if custom_exclude == 'n':
                                print("請輸入要排除的電極（逗號分隔）")
                                print("範例: HEOG,VEOG,A1,A2")
                                exclude_input = input("排除電極: ").strip()
                                
                                if exclude_input:
                                    exclude_channels = [ch.strip() for ch in exclude_input.split(',')]
                                else:
                                    exclude_channels = ['HEOG', 'VEOG', 'A1', 'A2']
                            else:
                                exclude_channels = ['HEOG', 'VEOG', 'A1', 'A2']
                            
                            raw = apply_average_reference(raw, exclude_channels=exclude_channels)
                        else:
                            # 不排除任何電極
                            raw = apply_average_reference(raw, exclude_channels=[])
                        
                        print("✓ 已設定 Average Reference")
                        current_step = 6
                        break
                    
                    elif ref_choice == "1":
                        # Linked Mastoids Reference
                        print("\n【Linked Mastoids Reference】")
                        print("自動偵測使用 M1/M2 或 A1/A2")
                        
                        raw = apply_linked_mastoid_reference(raw)
                        current_step = 6
                        break
                    
                    elif ref_choice == "2":
                        # 單一電極 Reference
                        print("\n【單一電極 Reference】")
                        print(f"可用的電極: {raw.ch_names[:10]}... (共 {len(raw.ch_names)} 個)")
                        
                        while True:
                            ref_ch_input = input("請輸入要作為 Reference 的電極名稱 (例如: Cz) / b 返回: ").strip()
                            
                            if ref_ch_input.lower() in ['b', 'back']:
                                break
                            
                            if ref_ch_input in raw.ch_names:
                                raw = apply_single_electrode_reference(raw, ref_ch_input)
                                current_step = 6
                                break
                            else:
                                print(f"⚠️  找不到電極: {ref_ch_input}")
                                show_all = input("是否顯示所有電極名稱? (y/n): ").strip().lower()
                                if show_all == 'y':
                                    print(f"所有電極: {', '.join(raw.ch_names)}")
                        
                        if current_step == 6:
                            break
                    
                    elif ref_choice == "4":
                        # 跳過
                        print("✓ 略過 Re-referencing")
                        current_step = 6
                        break
                    
                    else:
                        print("⚠️  無效的選項，請輸入 1-4")
            
            # ============================================================
            # 步驟 6: 插值壞通道
            # ============================================================
            elif current_step == 6:
                print("\n" + "="*60)
                print("步驟 6/6: 插值壞通道 (Bad Channel Interpolation)")
                print("="*60)
                
                interpolated_bads = []
                if raw.info['bads']:
                    print(f"對 {len(raw.info['bads'])} 個壞通道進行球面插值...")
                    print(f"壞通道清單: {raw.info['bads']}")
                    
                    while True:
                        interpolate_choice = input("\n是否要執行插值? (y/n) [預設 y] / b 返回 / q 退出: ").strip().lower()
                        
                        # 導航
                        if interpolate_choice in ['q', 'quit']:
                            print("✓ 退出前處理流程")
                            return None, None
                        elif interpolate_choice in ['b', 'back']:
                            current_step = 5
                            break
                        
                        # 選項處理
                        if interpolate_choice != 'n':
                            interpolated_bads = raw.info['bads'].copy()
                            raw = raw.interpolate_bads(reset_bads=True, method='spline')
                            print("✓ 壞通道插值完成")
                            current_step = 7  # 前處理完成
                            break
                        else:
                            print("✓ 略過插值（壞通道仍標記在 raw.info['bads']）")
                            current_step = 7  # 前處理完成
                            break
                    
                    if current_step == 5:
                        continue
                else:
                    print("✓ 沒有壞通道需要插值")
                    current_step = 7  # 前處理完成
        
        # === 前處理完成 ===
        print("\n" + "="*60)
        print("前處理完成")
        print("="*60)
        
        # 儲存選項
        save_choice = input("\n是否要儲存前處理後的資料? (y/n) [預設 n]: ").strip().lower()
        if save_choice == 'y':
            print("\n請選擇儲存格式:")
            print("1. FIF (.fif) - MNE-Python 原生格式")
            print("2. MAT (.mat) - MATLAB 格式")
            fmt_choice = input("\n請選擇 (1/2) [預設 1]: ").strip()

            if fmt_choice == '2':
                default_file = f'sub_{subject_id}_preprocessed-raw.mat'
                output_file = input(f"\n請輸入檔名 [預設: {default_file}]: ").strip() or default_file
                if not output_file.endswith('.mat'):
                    output_file += '.mat'
                import scipy.io
                scipy.io.savemat(output_file, {
                    'data': raw.get_data(),
                    'ch_names': raw.ch_names,
                    'sfreq': raw.info['sfreq'],
                    'times': raw.times,
                })
                print(f"✓ 已儲存: {output_file}")
            else:
                default_file = f'sub_{subject_id}_preprocessed-raw.fif'
                output_file = input(f"\n請輸入檔名 [預設: {default_file}]: ").strip() or default_file
                if not (output_file.endswith('.fif') or output_file.endswith('.fif.gz')):
                    output_file += '.fif'
                raw.save(output_file, overwrite=True)
                print(f"✓ 已儲存: {output_file}")
        
        # 儲存處理資訊
        processing_info = {
            'original_channels': n_channels,
            'bad_channels': interpolated_bads if interpolated_bads else raw.info['bads'],
            'final_channels': len(raw.ch_names),
            'high_pass': l_freq,
            'low_pass': h_freq,
            'sampling_rate': raw.info['sfreq']
        }
        
        print("\n前處理摘要:")
        print(f"  - 原始通道數: {processing_info['original_channels']}")
        print(f"  - 最終通道數: {processing_info['final_channels']}")
        print(f"  - 壞通道數: {len(processing_info['bad_channels'])}")
        if processing_info['bad_channels']:
            print(f"  - 壞通道清單: {processing_info['bad_channels']}")
        print(f"  - 濾波範圍: {l_freq} - {h_freq} Hz")
        print(f"  - 採樣率: {processing_info['sampling_rate']} Hz")
        
        return raw, processing_info
    
    except TypeError as e:
        print(f"資料類型錯誤: {str(e)}")
        raise e
    except Exception as e:
        print(f"Preprocessing 發生問題: {str(e)}")
        raise e


def preprocess_raw_data_interactive(raw, subject_id):
    """
    互動式 EEG 預處理（提示使用者輸入參數）
    
    參數:
        raw (mne.io.Raw): 原始 EEG 資料
        subject_id (str): 受試者ID
    
    返回:
        tuple: (處理後的 raw 物件, 處理資訊字典)
    """
    return preprocess_raw_data(raw, subject_id, electrode_coords_csv=None)
"""
時間段分析模組 - Epochs Module

包含EEG時間段創建與分析相關功能。
版本: 2.0 - 新增可調參數功能
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def select_epoch_mode():
    """
    選擇 epoch 建立模式
    
    返回:
        str: 'fixed', 'event', 或 'asrt'
    """
    print("\n" + "="*60)
    print("Epoch 建立模式選擇")
    print("="*60)
    print("\n請選擇 epoch 建立模式:")
    print("1. 固定時間切割 (Fixed-length epochs)")
    print("   - 將連續資料切成固定長度片段")
    print("   - 適用：靜息態、無特定事件標記")
    print("\n2. 事件鎖定切割 (Event-locked epochs)")
    print("   - 以特定事件為中心切割")
    print("   - 適用：事件相關電位分析")
    print("\n3. ASRT 任務專用 (ASRT task-specific)")
    print("   - 自動識別 block 結構")
    print("   - 支援 Stimulus-locked 和 Response-locked")
    print("   - 自動標記 Random/Regular 條件")
    
    while True:
        choice = input("\n請輸入選項 (1/2/3): ").strip()
        if choice == '1':
            return 'fixed'
        elif choice == '2':
            return 'event'
        elif choice == '3':
            return 'asrt'
        else:
            print("⚠️  無效選項，請輸入 1, 2, 或 3")


def epoch_data_interactive(raw, subject_id):
    """
    創建EEG epochs（互動式，可調整參數）。
    
    參數:
        raw (mne.io.Raw): 要分段的Raw物件
        subject_id (str): 受試者ID
    
    返回:
        mne.Epochs: 創建的Epochs物件
    """
    print("\n" + "="*60)
    print("Epochs 建立 - 參數設定")
    print("="*60)
    
    # 參數檢查
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw 必須是 MNE Raw物件")
    if not isinstance(subject_id, str):
        subject_id = str(subject_id)
    
    # === Epoch 長度設定 ===
    print("\n【Epoch 長度設定】")
    print("用途：將連續資料切成固定長度的片段")
    print("建議範圍：1 - 10 秒")
    print("常用值：")
    print("  - 1 秒：快速事件相關分析")
    print("  - 2 秒：一般靜息態分析")
    print("  - 4 秒：頻譜分析（需要足夠的時間解析度）")
    
    while True:
        epoch_input = input("\n請輸入 Epoch 長度 (秒) [預設 2.0]: ").strip()
        
        if epoch_input == "":
            epoch_length = 2.0
            print(f"使用預設值: {epoch_length} 秒")
            break
        
        try:
            epoch_length = float(epoch_input)
            if 0.5 <= epoch_length <= 30:
                print(f"設定 Epoch 長度: {epoch_length} 秒")
                break
            else:
                print("⚠️  長度應在 0.5-30 秒之間，請重新輸入")
        except ValueError:
            print("⚠️  請輸入有效的數字")
    
    # === Rejection threshold 設定 ===
    print("\n【Rejection Threshold 設定】")
    print("用途：自動拒絕振幅過大的 epochs")
    print("建議範圍：100 - 200 µV")
    
    while True:
        reject_input = input("\n請輸入 rejection threshold (µV) [預設 150]: ").strip()
        
        if reject_input == "":
            reject_threshold = 150e-6
            print(f"使用預設值: 150 µV")
            break
        
        try:
            reject_value = float(reject_input)
            if 50 <= reject_value <= 500:
                reject_threshold = reject_value * 1e-6  # 轉換為 V
                print(f"設定 rejection threshold: {reject_value} µV")
                break
            else:
                print("⚠️  threshold 應在 50-500 µV 之間，請重新輸入")
        except ValueError:
            print("⚠️  請輸入有效的數字")
    
    # 創建 events
    print(f"\n創建固定長度的 events (每 {epoch_length} 秒)...")
    events = mne.make_fixed_length_events(raw, id=1, duration=epoch_length)
    print(f"✓ 總共創建 {len(events)} 個 events")
    
    # 定義拒絕標準
    reject_criteria = {
        'eeg': reject_threshold
    }
    
    # 建立 Epochs
    print("\n建立 Epochs...")
    epochs = mne.Epochs(
        raw, events, event_id=1,
        tmin=0, tmax=epoch_length,
        baseline=None,
        reject=reject_criteria,
        reject_by_annotation=True,  # 自動排除標記的壞段落
        preload=True
    )
    
    print(f"✓ Epochs 建立完成")
    print(f"  - 總 epochs: {len(events)}")
    print(f"  - 保留: {len(epochs)}")
    print(f"  - 拒絕: {len(events) - len(epochs)}")
    if len(events) > 0:
        print(f"  - 保留率: {len(epochs)/len(events)*100:.1f}%")
    
    return epochs


def epoch_data_asrt(raw, subject_id):
    """
    ASRT 實驗專用的 epoch 建立函數

    功能：
    - 自動識別 block 結構（trigger 10=開始, 30=結束）
    - 排除 block 1-6（練習階段）
    - 識別並標記 triplet test（trigger 20）但保留所有 trials
    - 標記 Random (41-44) 和 Regular (46-49) 條件
    - 支援 Stimulus-locked 和 Response-locked epochs
    - 創建包含完整 metadata 的 epochs

    參數:
        raw (mne.io.Raw): 要分段的Raw物件
        subject_id (str): 受試者ID

    返回:
        mne.Epochs: 創建的Epochs物件（包含 metadata）
    """
    print("\n" + "=" * 60)
    print("ASRT 任務 Epoch 建立（Random vs Regular）")
    print("=" * 60)

    # === 步驟 1: 取得 events ===
    # 優先從 annotations 轉 events，因為你的 cnt 檔有 '10','20','41' 這類描述
    try:
        events, event_id_map = mne.events_from_annotations(raw)
        print("使用 annotations 建立 events")
    except Exception:
        print("找不到 annotations，改用 Trigger 通道偵測 events")
        events = mne.find_events(raw, stim_channel='Trigger', shortest_event=1)
        event_id_map = None

    print(f"找到事件總數: {len(events)}")
    if len(events) > 0:
        print("事件碼種類（unique event codes）：", np.unique(events[:, 2]))

    # 小工具：從描述字串（例如 '10'）取得實際 event code 數字
    def _code(label: str):
        if event_id_map is None:
            try:
                return int(label)
            except ValueError:
                return None
        else:
            return event_id_map.get(label, None)

    # === 步驟 2: 定義 ASRT 事件碼 ===
    BLOCK_START = _code("10")
    BLOCK_END = _code("30")

    RANDOM_STIM = [_code(x) for x in ["41", "42", "43", "44"]]
    REGULAR_STIM = [_code(x) for x in ["46", "47", "48", "49"]]
    RANDOM_RESP = [_code(x) for x in ["21", "22", "23", "24"]]
    REGULAR_RESP = [_code(x) for x in ["26", "27", "28", "29"]]

    # 把 None 去掉（以防有某些碼不存在）
    RANDOM_STIM = [c for c in RANDOM_STIM if c is not None]
    REGULAR_STIM = [c for c in REGULAR_STIM if c is not None]
    RANDOM_RESP = [c for c in RANDOM_RESP if c is not None]
    REGULAR_RESP = [c for c in REGULAR_RESP if c is not None]

    # block 邊界
    if BLOCK_START is None:
        block_starts = np.empty((0, 3), dtype=int)
    else:
        block_starts = events[events[:, 2] == BLOCK_START]

    if BLOCK_END is None:
        block_ends = np.empty((0, 3), dtype=int)
    else:
        block_ends = events[events[:, 2] == BLOCK_END]

    n_blocks = len(block_starts)
    print("\n分析實驗結構...")
    print(f"✓ 偵測到 {n_blocks} 個 blocks")

    if n_blocks == 0:
        print("⚠️ 無法偵測到任何 block start (事件描述 '10')，無法進行 ASRT 切段。")
        return None

    # 假設總共有 34 個 block（6 練習 + 20 學習 + 8 測驗）
    expected_total_blocks = 34
    missing_practice_blocks = max(0, expected_total_blocks - n_blocks)
    if missing_practice_blocks > 0:
        print(
            f"推估缺少前 {missing_practice_blocks} 個練習 block (1-{missing_practice_blocks})，"
            f"第一個偵測到的 block 視為第 {missing_practice_blocks + 1} 區塊。"
        )

    # === 步驟 3: 選擇 epoch 類型 ===
    print("\n" + "=" * 60)
    print("選擇 Epoch 類型")
    print("=" * 60)
    print("1. Stimulus-locked (箭頭呈現時)")
    print("   - tmin=-0.8s, tmax=1.0s")
    print("   - baseline=(-0.5, -0.1)")
    print("\n2. Response-locked (按鍵反應時)")
    print("   - tmin=-1.1s, tmax=0.5s")
    print("   - baseline=(-1.1, -0.6)")
    print("\n3. 兩者都建立 (推薦)")  # ← 新增
    print("   - 同時建立 Stimulus-locked 和 Response-locked")  # ← 新增
    print("   - 可用於完整的 ASRT 分析流程")  # ← 新增

    while True:
        epoch_choice = input("\n請選擇 (1/2/3) [預設 3]: ").strip()  # ← 改提示
        if epoch_choice == "":  # ← 改預設
            epoch_choice = "3"  # ← 預設為「兩者都建立」
            print("✓ 使用預設：兩者都建立")
            break
        elif epoch_choice == "1":
            break
        elif epoch_choice == "2":
            break
        elif epoch_choice == "3":  # ← 新增
            break
        else:
            print("⚠️ 請輸入 1, 2, 或 3")  # ← 更新錯誤訊息

    # === 步驟 4: 選擇分析階段（block 範圍） ===
    print("\n" + "=" * 60)
    print("選擇分析階段")
    print("=" * 60)
    print("1. 僅學習階段 (Block 7-26) - 推薦")
    print("2. 僅測驗階段 (Block 27-34)")
    print("3. 學習+測驗 (Block 7-34)")

    while True:
        phase_choice = input("\n請選擇 (1/2/3) [預設 1]: ").strip()
        if phase_choice == "" or phase_choice == "1":
            min_block, max_block = 7, 26
            phase_name = "Learning"
            break
        elif phase_choice == "2":
            min_block, max_block = 27, 34
            phase_name = "Test"
            break
        elif phase_choice == "3":
            min_block, max_block = 7, 34
            phase_name = "Learning+Test"
            break
        else:
            print("⚠️ 請輸入 1, 2, 或 3")
    
    # === 步驟 4.5: 如果包含 Testing 階段，詢問版本 ===
    test_version = None
    if max_block >= 27:  # 包含 Testing blocks
        print("\n" + "=" * 60)
        print("Testing Block 版本選擇")
        print("=" * 60)
        print("你的實驗包含 Testing 階段（Block 27-34）")
        print("\n請選擇你的 Testing block 版本:")
        print("1. Motor-first 版本")
        print("   - Motor: 27, 28, 33, 34")
        print("   - Perceptual: 29, 30, 31, 32")
        print("\n2. Perceptual-first 版本")
        print("   - Perceptual: 27, 28, 33, 34")
        print("   - Motor: 29, 30, 31, 32")
        
        while True:
            version_choice = input("\n請選擇 (1/2): ").strip()
            if version_choice == "1":
                test_version = "motor_first"
                print("✓ 選擇: Motor-first 版本")
                break
            elif version_choice == "2":
                test_version = "perceptual_first"
                print("✓ 選擇: Perceptual-first 版本")
                break
            else:
                print("⚠️ 請輸入 1 或 2")

    # === 步驟 5: 根據選擇建立 Epochs ===
    
    # === 定義 motor/perceptual 判斷函數（所有選項共用）===
    def get_test_type(block_num, test_version):
        """
        根據 block 編號和版本，判斷是 motor 還是 perceptual
        
        Parameters
        ----------
        block_num : int
            Block 編號
        test_version : str
            'motor_first' 或 'perceptual_first'
            
        Returns
        -------
        str or None
            'motor', 'perceptual', 或 None (非 testing block)
        """
        if block_num < 27 or block_num > 34:
            return None  # 不是 testing block
        
        if test_version == "motor_first":
            # Motor: 27, 28, 33, 34
            # Perceptual: 29, 30, 31, 32
            if block_num in [27, 28, 33, 34]:
                return "motor"
            else:
                return "perceptual"
        elif test_version == "perceptual_first":
            # Perceptual: 27, 28, 33, 34
            # Motor: 29, 30, 31, 32
            if block_num in [27, 28, 33, 34]:
                return "perceptual"
            else:
                return "motor"
        else:
            return None
    
   
    if epoch_choice == "3":
        # ============================================================
        # 選項 3: 同時建立兩種 Epochs
        # ============================================================
        print("\n" + "=" * 60)
        print("同時建立 Stimulus-locked 和 Response-locked Epochs")
        print("=" * 60)
        
        # === 5.1: 建立 Stimulus-locked epochs ===
        print("\n【1/2】建立 Stimulus-locked Epochs...")
        
        stim_event_codes = RANDOM_STIM + REGULAR_STIM
        stim_tmin, stim_tmax = -0.8, 1.0
        stim_baseline = (-0.5, -0.1)
        
        # 篩選 Stimulus events 並建立 metadata
        stim_filtered_events = []
        stim_metadata_list = []
        
        for sample, prev_id, code in events:
            if code not in stim_event_codes:
                continue
            
            block_idx = np.searchsorted(block_starts[:, 0], sample) - 1
            if block_idx < 0 or block_idx >= len(block_starts):
                continue
            
            block_num = block_idx + 1 + missing_practice_blocks
            if block_num < min_block or block_num > max_block:
                continue
            
            trial_type = "Random" if code in RANDOM_STIM else "Regular"
            phase = "Practice" if block_num <= 6 else ("Learning" if block_num <= 26 else "Test")
            test_type = get_test_type(block_num, test_version)
            
            cond_code = 1 if trial_type == "Random" else 2
            stim_filtered_events.append([sample, 0, cond_code])
            
            metadata_dict = {
                "block": block_num,
                "trial_type": trial_type,
                "phase": phase,
                "orig_event_code": int(code),
            }
            if test_type is not None:
                metadata_dict["test_type"] = test_type
            
            stim_metadata_list.append(metadata_dict)
        
        if len(stim_filtered_events) == 0:
            print("⚠️ Stimulus events 篩選後沒有任何事件")
            return None
        
        stim_filtered_events = np.array(stim_filtered_events, dtype=int)
        stim_metadata_df = pd.DataFrame(stim_metadata_list)
        
        print(f"  篩選後 Stimulus events: {len(stim_filtered_events)}")
        
        # 建立 Stimulus epochs
        if 'A1' in raw.ch_names:
            raw.set_channel_types({'A1': 'misc'})
        if 'A2' in raw.ch_names:
            raw.set_channel_types({'A2': 'misc'})
        
        epochs_stim = mne.Epochs(
            raw,
            stim_filtered_events,
            event_id={"Random": 1, "Regular": 2},
            tmin=stim_tmin,
            tmax=stim_tmax,
            baseline=stim_baseline,
            reject=None,
            reject_by_annotation=True,
            metadata=stim_metadata_df,
            preload=True,
        )
        
        print(f"✓ Stimulus-locked Epochs 建立完成")
        print(f"  總 events: {len(stim_filtered_events)}")
        print(f"  保留 epochs: {len(epochs_stim)}")
        print(f"  保留率: {len(epochs_stim) / len(stim_filtered_events) * 100:.1f}%")
        
        # === 5.2: 建立 Response-locked epochs ===
        print("\n【2/2】建立 Response-locked Epochs...")
        
        resp_event_codes = RANDOM_RESP + REGULAR_RESP
        resp_tmin, resp_tmax = -1.1, 0.5
        resp_baseline = (-1.1, -0.6)
        
        # 篩選 Response events 並建立 metadata
        resp_filtered_events = []
        resp_metadata_list = []
        
        for sample, prev_id, code in events:
            if code not in resp_event_codes:
                continue
            
            block_idx = np.searchsorted(block_starts[:, 0], sample) - 1
            if block_idx < 0 or block_idx >= len(block_starts):
                continue
            
            block_num = block_idx + 1 + missing_practice_blocks
            if block_num < min_block or block_num > max_block:
                continue
            
            trial_type = "Random" if code in RANDOM_RESP else "Regular"
            phase = "Practice" if block_num <= 6 else ("Learning" if block_num <= 26 else "Test")
            test_type = get_test_type(block_num, test_version)
            
            cond_code = 1 if trial_type == "Random" else 2
            resp_filtered_events.append([sample, 0, cond_code])
            
            metadata_dict = {
                "block": block_num,
                "trial_type": trial_type,
                "phase": phase,
                "orig_event_code": int(code),
            }
            if test_type is not None:
                metadata_dict["test_type"] = test_type
            
            resp_metadata_list.append(metadata_dict)
        
        if len(resp_filtered_events) == 0:
            print("⚠️ Response events 篩選後沒有任何事件")
            # 如果 Response 失敗但 Stimulus 成功，返回 Stimulus
            return epochs_stim
        
        resp_filtered_events = np.array(resp_filtered_events, dtype=int)
        resp_metadata_df = pd.DataFrame(resp_metadata_list)
        
        print(f"  篩選後 Response events: {len(resp_filtered_events)}")
        
        # 建立 Response epochs
        epochs_resp = mne.Epochs(
            raw,
            resp_filtered_events,
            event_id={"Random": 1, "Regular": 2},
            tmin=resp_tmin,
            tmax=resp_tmax,
            baseline=resp_baseline,
            reject=None,
            reject_by_annotation=True,
            metadata=resp_metadata_df,
            preload=True,
        )
        
        print(f"✓ Response-locked Epochs 建立完成")
        print(f"  總 events: {len(resp_filtered_events)}")
        print(f"  保留 epochs: {len(epochs_resp)}")
        print(f"  保留率: {len(epochs_resp) / len(resp_filtered_events) * 100:.1f}%")
        
        # === 5.3: 顯示摘要 ===
        print("\n" + "=" * 60)
        print("雙 Epochs 建立完成！")
        print("=" * 60)
        print(f"分析階段: {phase_name} (Block {min_block}-{max_block})")
        print(f"\nStimulus-locked:")
        print(f"  - 時間窗口: {stim_tmin} ~ {stim_tmax} s")
        print(f"  - Baseline: {stim_baseline}")
        print(f"  - Epochs: {len(epochs_stim)}")
        print(f"  - Random: {len(stim_metadata_df[stim_metadata_df['trial_type']=='Random'])}")
        print(f"  - Regular: {len(stim_metadata_df[stim_metadata_df['trial_type']=='Regular'])}")
        
        print(f"\nResponse-locked:")
        print(f"  - 時間窗口: {resp_tmin} ~ {resp_tmax} s")
        print(f"  - Baseline: {resp_baseline}")
        print(f"  - Epochs: {len(epochs_resp)}")
        print(f"  - Random: {len(resp_metadata_df[resp_metadata_df['trial_type']=='Random'])}")
        print(f"  - Regular: {len(resp_metadata_df[resp_metadata_df['trial_type']=='Regular'])}")
        
        # === 5.4: 建立對應關係 ===
        print("\n記錄 epochs 對應關係...")
        
        # Stimulus epochs 的 event samples
        stim_samples = epochs_stim.events[:, 0]
        
        # Response epochs 的 event samples
        resp_samples = epochs_resp.events[:, 0]
        
        # 建立對應表（Response → Stimulus）
        resp_to_stim_map = {}  # {resp_index: stim_index}
        
        for resp_idx, resp_sample in enumerate(resp_samples):
            # 找出在此 Response 之前最近的 Stimulus
            before_resp = stim_samples < resp_sample
            if np.any(before_resp):
                stim_idx = np.where(before_resp)[0][-1]
                resp_to_stim_map[resp_idx] = stim_idx
        
        print(f"  ✓ 成功對應 {len(resp_to_stim_map)} 個 trials")
        
        # === 5.5: 返回 dict（包含對應關係，不儲存）===
        import os
        phase_tag = {"Learning": "learn", "Test": "test", "Learning+Test": "all"}[phase_name]
        
        return {
            'stimulus': epochs_stim,
            'response': epochs_resp,
            'resp_to_stim_map': resp_to_stim_map,
            'n_stim': len(epochs_stim),
            'n_resp': len(epochs_resp),
            'phase_name': phase_name,
            'phase_tag': phase_tag,
            'min_block': min_block,
            'max_block': max_block,
            'subject_id': subject_id
        }

    elif epoch_choice == "1" or epoch_choice == "2":
        # ============================================================
        # 選項 1 或 2: 建立單一類型 Epochs（保持原邏輯）
        # ============================================================
        if epoch_choice == "1":
            # Stimulus-locked
            used_event_codes = RANDOM_STIM + REGULAR_STIM
            tmin, tmax = -0.8, 1.0
            baseline = (-0.5, -0.1)
            epoch_type = "Stimulus"
        else:
            # Response-locked
            used_event_codes = RANDOM_RESP + REGULAR_RESP
            tmin, tmax = -1.1, 0.5
            baseline = (-1.1, -0.6)
            epoch_type = "Response"
        
        # === 步驟 6: 單一迴圈，同時產生 filtered_events + metadata（確保數量一致） ===
        print("\n建立 Random/Regular metadata 並篩選 events...")

        filtered_events = []
        metadata_list = []

        for sample, prev_id, code in events:
            # 只保留我們關心的事件碼
            if code not in used_event_codes:
                continue

            # 找到這個事件屬於哪一個 block
            block_idx = np.searchsorted(block_starts[:, 0], sample) - 1
            if block_idx < 0 or block_idx >= len(block_starts):
                continue

            # 實際實驗 block 編號（加上缺少的練習 block）
            block_num = block_idx + 1 + missing_practice_blocks

            # 限制在選定的分析階段 block 範圍內
            if block_num < min_block or block_num > max_block:
                continue

            # 判斷 trial_type
            if epoch_choice == "1":  # Stimulus-locked
                if code in RANDOM_STIM:
                    trial_type = "Random"
                else:
                    trial_type = "Regular"
            else:  # Response-locked
                if code in RANDOM_RESP:
                    trial_type = "Random"
                else:
                    trial_type = "Regular"

            # 判斷 phase（只是 metadata 的標籤）
            if block_num <= 6:
                phase = "Practice"
            elif block_num <= 26:
                phase = "Learning"
            else:
                phase = "Test"
            
            # 判斷 test_type (motor/perceptual)
            test_type = get_test_type(block_num, test_version)

            # 為 Epochs 建立「條件用事件碼」：
            # 1 = Random, 2 = Regular
            cond_code = 1 if trial_type == "Random" else 2
            filtered_events.append([sample, 0, cond_code])

            # metadata：保留原始事件碼，之後還可以查回來
            metadata_dict = {
                "block": block_num,
                "trial_type": trial_type,
                "phase": phase,
                "orig_event_code": int(code),
            }
            
            # 如果是 testing block，添加 test_type
            if test_type is not None:
                metadata_dict["test_type"] = test_type
            
            metadata_list.append(metadata_dict)

        if len(filtered_events) == 0:
            print("⚠️ 篩選後沒有任何事件，可檢查事件碼或 block 選擇。")
            return None

        filtered_events = np.array(filtered_events, dtype=int)
        metadata_df = pd.DataFrame(metadata_list)

        print(f"  - 篩選後事件數量: {len(filtered_events)}")
        print(f"  - metadata 列數: {len(metadata_df)}")

        # 這裡保證兩者長度一定一致：
        assert len(filtered_events) == len(metadata_df), \
            "filtered_events 與 metadata 長度不一致，請檢查邏輯。"

        # === 步驟 7: 建立 Epochs ===
        print("\n建立 Epochs...")

        event_id = {"Random": 1, "Regular": 2}
        
        # 在建立 epochs 之前設定
        if 'A1' in raw.ch_names:
            raw.set_channel_types({'A1': 'misc'})
        if 'A2' in raw.ch_names:
            raw.set_channel_types({'A2': 'misc'})
        
        reject_criteria = None

        epochs = mne.Epochs(
            raw,
            filtered_events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=reject_criteria,
            reject_by_annotation=True,
            metadata=metadata_df,
            preload=True,
        )

        print("✓ Epochs 建立完成")
        print(f"  - 總 epochs (事件): {len(filtered_events)}")
        print(f"  - 保留 epoch: {len(epochs)}")
        print(f"  - 拒絕: {len(filtered_events) - len(epochs)}")
        if len(filtered_events) > 0:
            print(f"  - 保留率: {len(epochs) / len(filtered_events) * 100:.1f}%")

        print("\n" + "=" * 60)
        print("ASRT Epochs 摘要（Random vs Regular）")
        print("=" * 60)
        print(f"Epoch 類型: {epoch_type}-locked")
        print(f"時間窗口: {tmin} ~ {tmax} s")
        print(f"Baseline: {baseline}")
        print(f"分析階段: {phase_name} (Block {min_block}-{max_block})")
        print("Triplet: 不檢測、不區分（全部合併）")
        print(f"Random trials: {len(metadata_df[metadata_df['trial_type'] == 'Random'])}")
        print(f"Regular trials: {len(metadata_df[metadata_df['trial_type'] == 'Regular'])}")
        
        # 如果包含 testing blocks，顯示 motor/perceptual 統計
        if 'test_type' in metadata_df.columns:
            print(f"\nTesting Block 統計 ({test_version}):")
            for test_t in ['motor', 'perceptual']:
                count = len(metadata_df[metadata_df.get('test_type', pd.Series()) == test_t])
                if count > 0:
                    print(f"  {test_t.capitalize()} blocks: {count} trials")
                    # 進一步細分 Regular/Random
                    reg_count = len(metadata_df[
                        (metadata_df.get('test_type', pd.Series()) == test_t) & 
                        (metadata_df['trial_type'] == 'Regular')
                    ])
                    ran_count = len(metadata_df[
                        (metadata_df.get('test_type', pd.Series()) == test_t) & 
                        (metadata_df['trial_type'] == 'Random')
                    ])
                    print(f"    - Regular: {reg_count}, Random: {ran_count}")
        
        # === 自動儲存 Epoch 檔案 ===
        import os
        # Stimulus / Response 標籤
        lock_tag = "stim" if epoch_type == "Stimulus" else "resp"
        # Learning / Test / Learning+Test 標籤
        phase_tag = {
            "Learning": "learn",
            "Test": "test",
            "Learning+Test": "all",
        }[phase_name]

        default_fname = f"{subject_id}_ASRT_{lock_tag}_{phase_tag}-epo.fif"
        fname = input(f"\n請輸入 epochs 檔名 [預設: {default_fname}]: ").strip() or default_fname
        if fname.endswith('.fif.gz'):
            pass
        elif fname.endswith('.fif'):
            if not ('-epo.fif' in fname or '_epo.fif' in fname):
                fname = fname[:-4] + '-epo.fif'
        else:
            fname += '-epo.fif'
        save_path = os.path.join(os.getcwd(), fname)

        epochs.save(save_path, overwrite=True)
        print(f"✓ 已儲存 epochs 檔案: {fname}")

        return epochs


def epoch_data(raw, subject_id, epoch_length=2.0, tmin=0, reject_threshold=150e-6, 
               reject_by_annotation=True):
    """
    創建EEG epochs（預設參數版本）。
    
    參數:
        raw (mne.io.Raw): 要分段的Raw物件
        subject_id (str): 受試者ID
        epoch_length (float): epoch長度，預設為2.0秒
        tmin (float): 起始時間點，預設為0
        reject_threshold (float): 拒絕閾值，預設為150µV
        reject_by_annotation (bool): 是否自動排除標記的區段，預設True
    
    返回:
        mne.Epochs: 創建的Epochs物件
    """
    print("\n開始建立epochs...")
    
    try:
        # 參數檢查
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError("raw 必須是 MNE Raw物件")
        if not isinstance(subject_id, str):
            subject_id = str(subject_id)
             
        # 創建固定時間長度的事件
        events = mne.make_fixed_length_events(raw, id=1, duration=epoch_length)
        print(f"總事件數: {len(events)}")
        print(f"前五個事件時間點:")
        print(events[:5])  # 印出前幾個事件
        
        # 定義拒絕標準
        reject_criteria = {
            'eeg': reject_threshold
        }
        
        # 建立 epochs
        epochs = mne.Epochs(
            raw, events, event_id=1,
            tmin=tmin, tmax=tmin+epoch_length,
            baseline=None,
            reject=reject_criteria,
            reject_by_annotation=reject_by_annotation,
            preload=True
        )
        
        # 印出 epochs 資訊
        print("\nEpochs 資訊:")
        print(epochs)
        print(f"每個 epoch 長度: {epoch_length} 秒")
        print(f"被拒絕的 epochs: {epochs.drop_log_stats()}")
        
        # 顯示 epochs
        epochs.plot(n_channels=20, n_epochs=10, block=True)
        plt.close('all')
        
        return epochs
    except ValueError as e:
        print(f"建立 epochs 時發生錯誤: {str(e)}")
        return None
    except RuntimeError as e:
        print(f"執行時發生錯誤: {str(e)}")
        return None
    except Exception as e:
        print(f"發生未預期的錯誤: {str(e)}")
        return None
    finally:
        plt.close('all')


def compute_psd(epochs, fmin=0, fmax=50, n_fft=2048, method='welch'):
    """
    計算epochs的功率頻譜密度(PSD)。
    
    參數:
        epochs (mne.Epochs): 要分析的Epochs物件
        fmin (float): 最小頻率，預設為0 Hz
        fmax (float): 最大頻率，預設為50 Hz
        n_fft (int): FFT點數，預設為2048
        method (str): PSD計算方法，預設為'welch'
    
    返回:
        mne.time_frequency.EpochsSpectrum: 計算的PSD
    """
    print("\n計算功率頻譜密度(PSD)...")
    
    # 計算PSD
    psd = epochs.compute_psd(method=method, fmin=fmin, fmax=fmax, n_fft=n_fft)
    
    # 顯示PSD
    psd.plot()
    plt.show()
    
    return psd


def compute_tfr(epochs, freqs=None, n_cycles=None, method='morlet'):
    """
    計算epochs的時頻表示(TFR)。
    
    參數:
        epochs (mne.Epochs): 要分析的Epochs物件
        freqs (array): 頻率範圍，預設為1-50 Hz
        n_cycles (array): 循環數，預設為freqs/2
        method (str): 時頻方法，預設為'morlet'
    
    返回:
        mne.time_frequency.AverageTFR: 計算的TFR
    """
    print("\n計算時頻(TFR)...")
    
    # 設定默認參數
    if freqs is None:
        freqs = np.arange(1, 50, 1)  # 1-50Hz, 1Hz steps
    
    if n_cycles is None:
        n_cycles = freqs / 2.
    
    # 計算TFR
    if method == 'morlet':
        power = mne.time_frequency.tfr_morlet(
            epochs, freqs=freqs, 
            n_cycles=n_cycles,
            use_fft=True, 
            return_itc=False, 
            n_jobs=1
        )
    elif method == 'multitaper':
        power = mne.time_frequency.tfr_multitaper(
            epochs, freqs=freqs,
            n_cycles=n_cycles,
            time_bandwidth=4.0,
            return_itc=False,
            n_jobs=1
        )
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    # 顯示TFR
    power.plot_joint(
        baseline=None, mode='mean',
        timefreqs=[(0.5, 10), (1.0, 20)],
        title='Time-Frequency Analysis'
    )
    plt.show()
    
    return power


def create_stimulus_locked_epochs(raw, events, event_id, tmin=-0.8, tmax=1.0, 
                                  baseline=(-0.5, -0.1), preload=True):
    """
    創建 Stimulus-locked epochs
    
    Parameters
    ----------
    raw : mne.io.Raw
        原始 EEG 資料
    events : ndarray
        事件陣列 (n_events, 3)
    event_id : dict
        事件 ID 字典，例如 {'regular': 1, 'random': 2}
    tmin : float
        Epoch 開始時間（秒，相對刺激）
    tmax : float
        Epoch 結束時間（秒，相對刺激）
    baseline : tuple or None
        Baseline 時間窗口 (start, end)
    preload : bool
        是否預載入資料
        
    Returns
    -------
    epochs : mne.Epochs
        Stimulus-locked epochs
    """
    epochs = mne.Epochs(
        raw, events, event_id,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        preload=preload,
        reject=None,  # 假設已做前處理
        picks='eeg'
    )
    
    print(f"✓ 創建 Stimulus-locked epochs: {len(epochs)} trials")
    print(f"  時間範圍: {tmin} ~ {tmax} s")
    print(f"  Baseline: {baseline}")
    
    return epochs


def create_response_locked_epochs(raw, events, response_times, event_id, 
                                  tmin=-1.1, tmax=0.5, baseline=(-1.1, -0.6),
                                  preload=True):
    """
    創建 Response-locked epochs
    
    Parameters
    ----------
    raw : mne.io.Raw
        原始 EEG 資料
    events : ndarray
        刺激事件陣列 (n_events, 3)
    response_times : ndarray
        反應時間（秒），相對刺激的時間
    event_id : dict
        事件 ID 字典
    tmin : float
        Epoch 開始時間（秒，相對反應）
    tmax : float
        Epoch 結束時間（秒，相對反應）
    baseline : tuple or None
        Baseline 時間窗口 (start, end)
    preload : bool
        是否預載入資料
        
    Returns
    -------
    epochs : mne.Epochs
        Response-locked epochs
    """
    # 創建新的事件陣列（反應時間點）
    response_events = events.copy()
    
    # 調整事件時間到反應時間點
    sfreq = raw.info['sfreq']
    for i, rt in enumerate(response_times):
        response_events[i, 0] = events[i, 0] + int(rt * sfreq)
    
    # 創建 epochs
    epochs = mne.Epochs(
        raw, response_events, event_id,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        preload=preload,
        reject=None,
        picks='eeg'
    )
    
    print(f"✓ 創建 Response-locked epochs: {len(epochs)} trials")
    print(f"  時間範圍: {tmin} ~ {tmax} s (相對反應)")
    print(f"  Baseline: {baseline}")
    print(f"  RT 範圍: {np.min(response_times)*1000:.1f} - {np.max(response_times)*1000:.1f} ms")
    
    # ========== 把 RT 加入 metadata ==========
    import pandas as pd
    
    metadata = pd.DataFrame({
        'rt': response_times[:len(epochs)]  # 只取實際保留的 epochs 數量
    })
    
    # 如果 epochs 已經有 metadata，合併
    if hasattr(epochs, 'metadata') and epochs.metadata is not None:
        for col in epochs.metadata.columns:
            if col not in metadata.columns:
                metadata[col] = epochs.metadata[col].values
    
    # 設定 metadata
    epochs.metadata = metadata
    print(f"  ✓ RT 已加入 metadata")
    # ============================================
    
    return epochs


def separate_trial_types(epochs, trial_type_labels):
    """
    根據標籤分離不同類型的 trials
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 資料
    trial_type_labels : list or ndarray
        每個 trial 的類型標籤，例如 ['regular', 'random', 'regular', ...]
        
    Returns
    -------
    epochs_dict : dict
        {trial_type: epochs_subset}
    """
    unique_types = np.unique(trial_type_labels)
    epochs_dict = {}
    
    for trial_type in unique_types:
        indices = [i for i, label in enumerate(trial_type_labels) if label == trial_type]
        epochs_dict[trial_type] = epochs[indices]
        print(f"  {trial_type}: {len(indices)} trials")
    
    return epochs_dict


def extract_block_epochs(epochs, block_info):
    """
    根據 block 資訊提取不同 block 的 epochs
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs 資料
    block_info : ndarray
        每個 trial 的 block 編號
        
    Returns
    -------
    block_epochs : dict
        {block_number: epochs_subset}
    """
    unique_blocks = np.unique(block_info)
    block_epochs = {}
    
    for block_num in unique_blocks:
        indices = np.where(block_info == block_num)[0]
        block_epochs[block_num] = epochs[indices]
        print(f"  Block {block_num}: {len(indices)} trials")
    
    return block_epochs
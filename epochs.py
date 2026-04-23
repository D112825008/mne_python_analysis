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


def epoch_data_asrt(raw, subject_id, trial_classification='trigger'):
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
    print("   - tmin=-1.0s, tmax=0.5s")
    print("   - baseline=(-1.0, -0.6)")
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
        stim_baseline = None
        
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
            metadata_dict['classification'] = trial_classification

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
        resp_tmin, resp_tmax = -1.5, 0.5
        resp_baseline = None

        # 預先建立每個 block 內 Stimulus sample → trial_in_block 的對照表
        stim_event_codes_for_resp = RANDOM_STIM + REGULAR_STIM
        stim_index_by_block = {}  # {block_num: {stim_sample: trial_index}}
        for s_sample, _, s_code in events:
            if s_code not in stim_event_codes_for_resp:
                continue
            b_idx = np.searchsorted(block_starts[:, 0], s_sample) - 1
            if b_idx < 0 or b_idx >= len(block_starts):
                continue
            b_num = b_idx + 1 + missing_practice_blocks
            if b_num < min_block or b_num > max_block:
                continue
            stim_index_by_block.setdefault(b_num, []).append(s_sample)
        # 轉成 {block_num: {stim_sample: index}} （已按時間排序）
        stim_index_by_block = {
            b: {s: i for i, s in enumerate(sorted(samples))}
            for b, samples in stim_index_by_block.items()
        }

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

            stim_samples_before = [s for s, _, c in events if c in stim_event_codes_for_resp and s < sample]
            stim_sample = stim_samples_before[-1] if stim_samples_before else -1

            # 用 stim_sample 查出該 Stimulus 在此 block 內的序號
            trial_in_block = stim_index_by_block.get(block_num, {}).get(stim_sample, -1)

            cond_code = 1 if trial_type == "Random" else 2
            resp_filtered_events.append([sample, 0, cond_code])

            metadata_dict = {
                "block": block_num,
                "trial_type": trial_type,
                "phase": phase,
                "orig_event_code": int(code),
                'stim_sample': stim_sample,
                'resp_sample': int(sample),
                'trial_in_block': trial_in_block,
            }
            if test_type is not None:
                metadata_dict["test_type"] = test_type
            metadata_dict['classification'] = trial_classification

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
        print(f"  - Baseline: None (applied in ERSP step)")
        print(f"  - Epochs: {len(epochs_stim)}")
        print(f"  - Random: {len(stim_metadata_df[stim_metadata_df['trial_type']=='Random'])}")
        print(f"  - Regular: {len(stim_metadata_df[stim_metadata_df['trial_type']=='Regular'])}")
        
        print(f"\nResponse-locked:")
        print(f"  - 時間窗口: {resp_tmin} ~ {resp_tmax} s")
        print(f"  - Baseline: None (per-trial baseline applied in ERSP step)")
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
        
        # === 5.5：詢問存檔（選項 3 新增）===
        import os
        phase_tag = {"Learning": "learn", "Test": "test", "Learning+Test": "all"}[phase_name]

        print("\n" + "=" * 60)
        print("儲存 Epochs 檔案")
        print("=" * 60)

        default_stim = f"{subject_id}_ASRT_stim_{phase_tag}_all-epo.fif"
        default_resp = f"{subject_id}_ASRT_resp_{phase_tag}_all-epo.fif"

        fname_s = input(f"\nStimulus epoch 檔名 [預設: {default_stim}]: ").strip() or default_stim
        epochs_stim.save(os.path.join(os.getcwd(), fname_s), overwrite=True)
        print(f"✓ 已儲存: {fname_s}")

        fname_r = input(f"Response epoch 檔名 [預設: {default_resp}]: ").strip() or default_resp
        epochs_resp.save(os.path.join(os.getcwd(), fname_r), overwrite=True)
        print(f"✓ 已儲存: {fname_r}")

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
            baseline = (-1.0, -0.6)
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

            # 如果是 Response-locked，記錄對應的 Stimulus sample
            if epoch_choice == "2":
                stim_event_codes_for_resp = RANDOM_STIM + REGULAR_STIM
                stim_samples_before = [s for s, _, c in events if c in stim_event_codes_for_resp and s < sample]
                stim_sample = stim_samples_before[-1] if stim_samples_before else -1
                metadata_dict['stim_sample'] = stim_sample
                metadata_dict['resp_sample'] = int(sample)

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


def _tag_classification(result, label):
    """Add 'classification' column to epoch metadata (handles Epochs or dict)."""
    if result is None:
        return result
    if isinstance(result, dict):
        for key in ('stimulus', 'response'):
            ep = result.get(key)
            if ep is not None and hasattr(ep, 'metadata') and ep.metadata is not None:
                ep.metadata['classification'] = label
    elif hasattr(result, 'metadata') and result.metadata is not None:
        result.metadata['classification'] = label
    return result


def create_asrt_epochs(raw, subject_id, behavior_df=None, trial_classification='trigger'):
    """
    ASRT Epoch 建立，支援兩種 trial 分類方式。

    Parameters
    ----------
    raw : mne.io.Raw
    subject_id : str
    behavior_df : pd.DataFrame or None
        從 CSV 載入的行為資料（含 learning_loop.thisRepN、thisTrialN、
        correct_answer_index 等欄位）
    trial_classification : str
        'trigger'  — 根據 trigger code 分類 Regular / Random（預設）
        'triplet'  — 根據 triplet 出現頻率分類 high / low

    Returns
    -------
    同 epoch_data_asrt() 的回傳格式
    """
    if trial_classification == 'trigger':
        return _tag_classification(epoch_data_asrt(raw, subject_id, trial_classification='trigger'), 'trigger')

    # ── triplet 分類模式 ──────────────────────────────────────────
    if behavior_df is None:
        print("⚠  triplet 分類需要行為資料（behavior_df），已自動改為 trigger 分類")
        return _tag_classification(epoch_data_asrt(raw, subject_id, trial_classification='triplet'), 'triplet')

    from collections import Counter
    import os

    print("\n" + "=" * 60)
    print("ASRT Epoch 建立（Triplet 頻率分類：high / low）")
    print("=" * 60)

    # === 1. 取得 events ===
    try:
        events, event_id_map = mne.events_from_annotations(raw)
        print("使用 annotations 建立 events")
    except Exception:
        print("找不到 annotations，改用 Trigger 通道偵測 events")
        events = mne.find_events(raw, stim_channel='Trigger', shortest_event=1)
        event_id_map = None

    print(f"找到事件總數: {len(events)}")
    if len(events) > 0:
        print("事件碼種類:", np.unique(events[:, 2]))

    def _code(label):
        if event_id_map is None:
            try:
                return int(label)
            except ValueError:
                return None
        return event_id_map.get(label, None)

    # === 2. ASRT 事件碼 ===
    BLOCK_START  = _code("10")
    RANDOM_STIM  = [c for c in [_code(x) for x in ["41","42","43","44"]] if c is not None]
    REGULAR_STIM = [c for c in [_code(x) for x in ["46","47","48","49"]] if c is not None]
    RANDOM_RESP  = [c for c in [_code(x) for x in ["21","22","23","24"]] if c is not None]
    REGULAR_RESP = [c for c in [_code(x) for x in ["26","27","28","29"]] if c is not None]

    # === 3. Block 結構 ===
    block_starts = (events[events[:, 2] == BLOCK_START]
                    if BLOCK_START is not None
                    else np.empty((0, 3), dtype=int))
    n_blocks = len(block_starts)
    print(f"\n偵測到 {n_blocks} 個 blocks")
    if n_blocks == 0:
        print("⚠  無法偵測到 block start，無法進行 ASRT 切段")
        return None

    expected_total = 34
    missing_practice_blocks = max(0, expected_total - n_blocks)
    if missing_practice_blocks > 0:
        print(f"推估缺少前 {missing_practice_blocks} 個練習 block，"
              f"第一個偵測到的 block 視為第 {missing_practice_blocks + 1} 區塊")

    # === 4. Epoch 類型 ===
    print("\n" + "=" * 60)
    print("選擇 Epoch 類型")
    print("=" * 60)
    print("1. Stimulus-locked  (tmin=-0.8s, tmax=1.0s)")
    print("2. Response-locked  (tmin=-1.5s, tmax=0.5s)")
    print("3. 兩者都建立（推薦）")
    while True:
        ec = input("\n請選擇 (1/2/3) [預設 3]: ").strip() or "3"
        if ec in ("1", "2", "3"):
            break
        print("⚠  請輸入 1、2 或 3")

    # === 5. 分析階段 ===
    print("\n" + "=" * 60)
    print("選擇分析階段")
    print("=" * 60)
    print("1. 僅學習階段 (Block 7-26) - 推薦")
    print("2. 僅測驗階段 (Block 27-34)")
    print("3. 學習+測驗 (Block 7-34)")
    while True:
        pc = input("\n請選擇 (1/2/3) [預設 1]: ").strip() or "1"
        if pc == "1":
            min_block, max_block, phase_name = 7, 26, "Learning"
            break
        elif pc == "2":
            min_block, max_block, phase_name = 27, 34, "Test"
            break
        elif pc == "3":
            min_block, max_block, phase_name = 7, 34, "Learning+Test"
            break
        else:
            print("⚠  請輸入 1、2 或 3")

    # === 5.5. Testing 版本（若包含 testing blocks）===
    test_version = None
    if max_block >= 27:
        print("\n" + "=" * 60)
        print("Testing Block 版本選擇")
        print("=" * 60)
        print("1. Motor-first（27,28,33,34=motor；29-32=perceptual）")
        print("2. Perceptual-first（27,28,33,34=perceptual；29-32=motor）")
        while True:
            vc = input("\n請選擇 (1/2): ").strip()
            if vc == "1":
                test_version = "motor_first"
                break
            elif vc == "2":
                test_version = "perceptual_first"
                break
            else:
                print("⚠  請輸入 1 或 2")

    def _get_test_type(block_num):
        if block_num < 27 or block_num > 34:
            return None
        if test_version == "motor_first":
            return "motor" if block_num in [27, 28, 33, 34] else "perceptual"
        if test_version == "perceptual_first":
            return "perceptual" if block_num in [27, 28, 33, 34] else "motor"
        return None

    # === 6. 從 behavior_df 預先建立 triplet 序列（仿照 R import_d()）===
    # block_triplets[block_num][trial_in_block] = {'triplet': str|None, 'valid': bool}
    block_triplets = {}
    key_ans = 'correct_answer_index'

    def _build_tmap(seq):
        """從單一 block 的 correct_answer_index 序列算 triplet（per-block shift）"""
        tmap = {}
        for i, val in enumerate(seq):
            if i < 2:
                tmap[i] = {'triplet': None, 'valid': False}
                continue
            n2, n1, n = seq[i-2], seq[i-1], val
            if str(n2) == str(n1) == str(n) or str(n2) == str(n):
                tmap[i] = {'triplet': None, 'valid': False}
            else:
                tmap[i] = {'triplet': f"{n2}{n1}{n}", 'valid': True}
        return tmap

    if key_ans not in behavior_df.columns:
        print(f"⚠  behavior_df 缺少欄位 '{key_ans}'，已改為 trigger 分類")
        return _tag_classification(epoch_data_asrt(raw, subject_id, trial_classification='triplet'), 'triplet')

    print("\n從行為資料建立 triplet 序列（仿照 R import_d()，含 Learning + Testing）...")

    # --- Learning blocks（block 7-26）---
    # learning_loop.thisTrialN : 0-19 → block_num = val + 7
    # learning_trials.thisTrialN : trial 順序
    key_lb = 'learning_trials.thisTrialN'  # block 編號（0-19，20個唯一值）
    key_lt = 'learning_loop.thisTrialN'    # trial in block（0-84，85個唯一值）
    if key_lb in behavior_df.columns and key_lt in behavior_df.columns:
        learn_df = behavior_df[behavior_df[key_lb].notna() & behavior_df[key_ans].notna()].copy()
        learn_df[key_lb] = learn_df[key_lb].astype(float).astype(int)
        learn_df[key_lt] = learn_df[key_lt].astype(float)
        for blk_val, grp in learn_df.groupby(key_lb):
            block_num = int(blk_val) + 7
            if block_num < min_block or block_num > max_block:
                continue
            grp_sorted = grp.sort_values(by=key_lt)
            seq = grp_sorted[key_ans].astype(int).tolist()
            block_triplets[block_num] = _build_tmap(seq)
        print(f"  Learning: {sum(1 for k in block_triplets if k <= 26)} blocks 建立完成")
    else:
        print(f"  ⚠  Learning block 欄位缺失（{key_lb} / {key_lt}），跳過 Learning triplet")

    # --- Testing blocks（block 27-34）---
    # combined_testing_trials.thisTrialN    : 0-7 → block_num = val + 27
    # motor_percept_testing_loop / percept_motor_testing_loop : trial 順序（依版本而異）
    print(f"  behavior_df 欄位數: {len(behavior_df.columns)}")
    test_related = [c for c in behavior_df.columns if 'test' in c.lower()]
    print(f"  含 'test' 的欄位: {test_related}")
    key_tb = 'combined_testing_trials.thisTrialN'
    possible_tt = ['motor_percept_testing_loop.thisTrialN',
                   'percept_motor_testing_loop.thisTrialN']
    key_tt = next((c for c in possible_tt if c in behavior_df.columns), None)
    print(f"  possible_tt 搜尋結果: { {c: (c in behavior_df.columns) for c in possible_tt} }")
    print(f"  key_tt 解析為: {key_tt}")
    if key_tt is None:
        print("  ⚠  Testing trial 欄位缺失，Testing triplet 不可用")
    if key_tb in behavior_df.columns and key_tt is not None:
        test_df = behavior_df[behavior_df[key_tb].notna() & behavior_df[key_ans].notna()].copy()
        test_df[key_tb] = test_df[key_tb].astype(float).astype(int)
        test_df[key_tt] = test_df[key_tt].astype(float)
        for blk_val, grp in test_df.groupby(key_tb):
            block_num = int(blk_val) + 27
            if block_num < min_block or block_num > max_block:
                continue
            grp_sorted = grp.sort_values(by=key_tt)
            seq = grp_sorted[key_ans].astype(int).tolist()
            block_triplets[block_num] = _build_tmap(seq)
        print(f"  Testing:  {sum(1 for k in block_triplets if k >= 27)} blocks 建立完成")
    else:
        print(f"  ⚠  Testing block 欄位缺失（{key_tb} / {key_tt}），Testing triplet 不可用")

    print(f"✓ 共建立 {len(block_triplets)} 個 block 的 triplet 序列")

    # === 7. 輔助函式 ===
    def _collect_trials(event_codes, epoch_type):
        """掃描事件，回傳含 triplet 資訊的 trial list"""
        block_counter = {}
        trial_list = []
        for sample, _, code in events:
            if code not in event_codes:
                continue
            block_idx = np.searchsorted(block_starts[:, 0], sample) - 1
            if block_idx < 0 or block_idx >= len(block_starts):
                continue
            block_num = block_idx + 1 + missing_practice_blocks
            if block_num < min_block or block_num > max_block:
                continue

            block_counter.setdefault(block_num, 0)
            tib = block_counter[block_num]
            block_counter[block_num] += 1

            random_codes = RANDOM_STIM if epoch_type == 'stim' else RANDOM_RESP
            position_type = 'random' if code in random_codes else 'regular'
            phase = ('Practice' if block_num <= 6
                     else ('Learning' if block_num <= 26 else 'Test'))

            entry = block_triplets.get(block_num, {}).get(tib, {})

            if epoch_type == 'stim':
                stim_sample = int(sample)
                resp_sample = -1
            else:
                stim_codes = [41, 42, 43, 44, 46, 47, 48, 49]
                stim_before = [s for s, _, c in events if c in stim_codes and s < sample]
                stim_sample = int(stim_before[-1]) if stim_before else -1
                resp_sample = int(sample)

            trial_list.append({
                'sample': sample,
                'code': code,
                'block_num': block_num,
                'trial_in_block': tib,
                'position_type': position_type,
                'phase': phase,
                'test_type': _get_test_type(block_num),
                'triplet': entry.get('triplet'),
                'triplet_valid': entry.get('valid', False),
                'stim_sample': stim_sample,
                'resp_sample': resp_sample,
            })
        return trial_list

    def _debug_print(trial_list, label):
        reg = sum(1 for t in trial_list if t['position_type'] == 'regular')
        ran_h = sum(1 for t in trial_list if t['position_type'] == 'random' and t.get('trial_type') == 'high')
        ran_l = sum(1 for t in trial_list if t['position_type'] == 'random' and t.get('trial_type') == 'low')
        ran_none = sum(1 for t in trial_list if t['position_type'] == 'random' and t.get('trial_type') is None)
        print(f"  [{label}] regular: {reg}, random_high: {ran_h}, random_low: {ran_l}, random_invalid: {ran_none}")

    def _assign_types(trial_list, median_count=None):
        """計算 random triplet 頻率後賦予 trial_type（high / low / None）。

        median 只用 Learning blocks 的 random trial 計算一次，
        再套用到所有 trial（包含 Testing），仿照 R 的 assign_freq() 做法。

        若傳入 median_count，跳過計算直接使用（供 resp 共用 stim 的 median）。
        回傳 (trial_list, random_counts, median_count) 供外部重用。
        """
        learning_random = [
            t for t in trial_list
            if t['phase'] == 'Learning'
            and t['position_type'] == 'random'
            and t['triplet_valid']
            and t['triplet'] is not None
        ]
        random_counts = Counter(t['triplet'] for t in learning_random)
        if median_count is None:
            median_count = np.median(list(random_counts.values())) if random_counts else 0
            print(f"  Learning random triplet 數: {len(random_counts)}, median count: {median_count}")
            print(f"  count 分布: {sorted(random_counts.values())}")
        else:
            print(f"  [shared median] median count: {median_count}  (random_counts from this list: {len(random_counts)})")

        for t in trial_list:
            if t['position_type'] == 'regular':
                t['trial_type'] = 'high'
            elif t['position_type'] == 'random':
                if not t['triplet_valid'] or t['triplet'] is None:
                    t['trial_type'] = None
                else:
                    cnt = random_counts.get(t['triplet'], 0)
                    t['trial_type'] = 'high' if cnt >= median_count else 'low'
            else:
                t['trial_type'] = None
        return trial_list, random_counts, median_count

    def _to_mne(trial_list, tmin, tmax):
        """將 trial list 轉為 MNE Epochs"""
        cond_map = {'high': 1, 'low': 2}
        ev_rows, meta_rows = [], []
        for t in trial_list:
            if t['trial_type'] is None:
                continue
            ev_rows.append([t['sample'], 0, cond_map[t['trial_type']]])
            row = {
                'block': t['block_num'],
                'trial_in_block': t['trial_in_block'],
                'trial_type': t['trial_type'],
                'position_type': t['position_type'],
                'phase': t['phase'],
                'orig_event_code': int(t['code']),
                'triplet': t['triplet'],
                'stim_sample': t.get('stim_sample', -1),
                'resp_sample': t.get('resp_sample', -1),
            }
            if t['test_type'] is not None:
                row['test_type'] = t['test_type']
            meta_rows.append(row)

        if not ev_rows:
            return None
        ev_arr = np.array(ev_rows, dtype=int)
        meta_df = pd.DataFrame(meta_rows)

        for ch in ('A1', 'A2'):
            if ch in raw.ch_names:
                raw.set_channel_types({ch: 'misc'})

        present_types = set(t['trial_type'] for t in trial_list if t['trial_type'] is not None)
        event_id = {k: v for k, v in cond_map.items() if k in present_types}

        return mne.Epochs(
            raw, ev_arr,
            event_id=event_id,
            tmin=tmin, tmax=tmax,
            baseline=None,
            reject=None,
            reject_by_annotation=True,
            metadata=meta_df,
            preload=True,
        )

    # === 8. 建立 Epochs ===
    phase_tag = {"Learning": "learn", "Test": "test", "Learning+Test": "all"}[phase_name]

    if ec == "3":
        # --- Stimulus-locked ---
        all_stim = _collect_trials(RANDOM_STIM + REGULAR_STIM, 'stim')
        stim_trials, _stim_counts, _shared_median = _assign_types(all_stim)
        _debug_print(stim_trials, 'stim')

        epochs_stim = _to_mne(stim_trials, tmin=-0.8, tmax=1.0)
        if epochs_stim is None:
            print("⚠  Stimulus events 篩選後沒有任何事件")
            return None
        print(f"\n✓ Stimulus-locked Epochs: {len(epochs_stim)}")
        for lbl in ('high', 'low'):
            n = (epochs_stim.metadata['trial_type'] == lbl).sum()
            print(f"   {lbl}: {n}")

        # --- Response-locked（共用 stim 的 median）---
        all_resp = _collect_trials(RANDOM_RESP + REGULAR_RESP, 'resp')
        resp_trials, _, _ = _assign_types(all_resp, median_count=_shared_median)
        _debug_print(resp_trials, 'resp')
        epochs_resp = _to_mne(resp_trials, tmin=-1.5, tmax=0.5)
        if epochs_resp is None:
            print("⚠  Response events 篩選後沒有任何事件")
            epochs_resp = None
        else:
            print(f"\n✓ Response-locked Epochs: {len(epochs_resp)}")
            for lbl in ('high', 'low'):
                n = (epochs_resp.metadata['trial_type'] == lbl).sum()
                print(f"   {lbl}: {n}")

        # resp_to_stim_map（按 sample 最近配對）
        resp_to_stim_map = {}
        if epochs_resp is not None:
            stim_samples = np.array([t['sample'] for t in stim_trials
                                     if t['trial_type'] is not None])
            resp_samples_all = [t['sample'] for t in resp_trials
                                if t['trial_type'] is not None]
            for resp_i, rs in enumerate(resp_samples_all):
                before = stim_samples[stim_samples < rs]
                if len(before):
                    stim_i = int(np.where(stim_samples == before[-1])[0][0])
                    resp_to_stim_map[resp_i] = stim_i

        # 詢問儲存檔名
        default_stim = f"{subject_id}_ASRT_stim_{phase_tag}_triplet-epo.fif"
        default_resp = f"{subject_id}_ASRT_resp_{phase_tag}_triplet-epo.fif"
        fname_s = input(f"\nStimulus epoch 檔名 [預設: {default_stim}]: ").strip() or default_stim
        epochs_stim.save(os.path.join(os.getcwd(), fname_s), overwrite=True)
        print(f"✓ 已儲存: {fname_s}")

        if epochs_resp is not None:
            fname_r = input(f"Response epoch 檔名 [預設: {default_resp}]: ").strip() or default_resp
            epochs_resp.save(os.path.join(os.getcwd(), fname_r), overwrite=True)
            print(f"✓ 已儲存: {fname_r}")

        result = {
            'stimulus': epochs_stim,
            'response': epochs_resp,
            'resp_to_stim_map': resp_to_stim_map,
            'n_stim': len(epochs_stim),
            'n_resp': len(epochs_resp) if epochs_resp is not None else 0,
            'phase_name': phase_name,
            'phase_tag': phase_tag,
            'min_block': min_block,
            'max_block': max_block,
            'subject_id': subject_id,
        }
        return _tag_classification(result, 'triplet')

    # --- Stimulus 或 Response 單一模式 ---
    if ec == "1":
        trials, _, _ = _assign_types(_collect_trials(RANDOM_STIM + REGULAR_STIM, 'stim'))
        _debug_print(trials, 'stim')
        ep = _to_mne(trials, tmin=-0.8, tmax=1.0)
        lock_tag = "stim"
    else:
        trials, _, _ = _assign_types(_collect_trials(RANDOM_RESP + REGULAR_RESP, 'resp'))
        _debug_print(trials, 'resp')
        ep = _to_mne(trials, tmin=-1.5, tmax=0.5)
        lock_tag = "resp"

    if ep is None:
        print("⚠  篩選後沒有任何事件")
        return None

    print(f"\n✓ Epochs 建立完成: {len(ep)}")
    for lbl in ('high', 'low'):
        n = (ep.metadata['trial_type'] == lbl).sum()
        print(f"   {lbl}: {n}")

    default_fname = f"{subject_id}_ASRT_{lock_tag}_{phase_tag}_triplet-epo.fif"
    fname = input(f"\n請輸入 epochs 檔名 [預設: {default_fname}]: ").strip() or default_fname
    ep.save(os.path.join(os.getcwd(), fname), overwrite=True)
    print(f"✓ 已儲存: {fname}")

    ep.metadata['classification'] = 'triplet'
    return ep


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
                                  tmin=-1.1, tmax=0.5, baseline=(-1.0, -0.6),
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
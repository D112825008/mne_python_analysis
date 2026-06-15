"""
群體 ERSP 分析模組 v2.0 - Group-level ERSP Analysis Module

Block 分組結構與個人分析完全一致：
  Learning : Block7-11, 12-16, 17-21, 22-26
  Testing  : Block27-28, 29-30, 31-32, 33-34

每個 Block 組產生一張群體比較圖（Regular | Random | Difference + 顯著 cluster）。

作者: Dillian (HE-JUN, CHEN)
版本: 2.0
"""

import os
import numpy as np
import matplotlib

if __name__ == '__main__':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import mne
from pathlib import Path
import pickle
from scipy import stats
from mne.stats import permutation_cluster_1samp_test
import warnings

# ============================================================
# 全域常數
# ============================================================

ROI_GROUPS = {
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

LEARNING_GROUPS = [(7, 11), (12, 16), (17, 21), (22, 26)]
TESTING_GROUPS  = [(27, 28), (29, 30), (31, 32), (33, 34)]

# ── 全域 permutation 結果收集器 ──────────────────────────────
# 所有分析的 sig cluster 結果都 append 到這裡，最後統一印出
_PERM_SUMMARY = []   # 每筆: dict(label, roi, lock, phase, pair, cond_pair, n_sub, n_sig, n_total)

def _log_perm(label, roi, lock, phase, pair, cond_pair, n_sub, n_sig, n_total):
    """將一筆 permutation test 結果寫入全域摘要。"""
    _PERM_SUMMARY.append(dict(
        label=label, roi=roi, lock=lock, phase=phase, pair=pair,
        cond_pair=cond_pair, n_sub=n_sub, n_sig=n_sig, n_total=n_total,
    ))

def print_perm_summary():
    """在分析結尾印出所有 permutation test 結果的彙整表。"""
    if not _PERM_SUMMARY:
        print("\n[Perm Summary] 無結果可顯示")
        return
    sig_rows = [r for r in _PERM_SUMMARY if r['n_sig'] > 0]
    print("\n" + "╔" + "═"*110 + "╗")
    print("║  PERMUTATION TEST SUMMARY  ─  所有分析彙整" + " "*67 + "║")
    print("╠" + "═"*110 + "╣")
    hdr = f"  {'Label':<28} {'ROI':<22} {'Lock':<10} {'Phase':<14} {'Pair':<14} {'Cond Pair':<30} {'N':>4} {'Sig/Tot':>9}"
    print("║" + hdr + "║")
    print("╠" + "─"*110 + "╣")
    for r in _PERM_SUMMARY:
        sig_marker = " ★" if r['n_sig'] > 0 else "  "
        row = (f"  {r['label']:<28} {r['roi']:<22} {r['lock']:<10} {r['phase']:<14} "
               f"{r['pair']:<14} {r['cond_pair']:<30} {r['n_sub']:>4} "
               f"{r['n_sig']:>3}/{r['n_total']:<4}{sig_marker}")
        print("║" + row + "║")
    print("╠" + "═"*110 + "╣")
    print(f"║  合計：{len(_PERM_SUMMARY)} 項檢定，★ 顯著 {len(sig_rows)} 項（p < 0.05, cluster-corrected）" +
          " "*max(0, 75 - len(str(len(_PERM_SUMMARY))) - len(str(len(sig_rows)))) + "║")
    print("╚" + "═"*110 + "╝")

# ── G*Power 樣本數估算 ────────────────────────────────────────
def compute_power_analysis(h5_dir, subject_ids,
                            pkl_dir=None,
                            roi_configs=None,
                            alpha=0.05, power_target=0.80,
                            condition_left='regular_high',
                            condition_right='random_low'):
    """
    從 Learning block 組計算 Cohen's d 並估算所需樣本數。

    Response-locked : 從 h5_dir 讀取 .h5（MNE AverageTFR）
    Stimulus-locked : 從 pkl_dir 讀取 .pkl（與群體圖片路徑一致）
                      pkl 檔名格式：
                      {sid}_learning_stimulus_{roi_lower}_{condition}_{block}_ersp.pkl

    Windows（與 export_ersp_to_csv 一致）：
      Response-locked │ Theta 4–8 Hz │ −300 to +50 ms  │ Motor ROI
      Stimulus-locked │ Alpha 8–13Hz │ +100 to +300 ms │ Perceptual ROI

    Parameters
    ----------
    h5_dir  : str  response-locked .h5 所在目錄
    pkl_dir : str  stimulus-locked .pkl 所在目錄（None 時沿用 h5_dir，但通常兩者不同）

    Returns
    -------
    dict: 每個 (lock, roi) → {'d': float, 'n_required': int, 'power_at_n': float}
    """
    try:
        from statsmodels.stats.power import TTestPower
    except ImportError:
        print("  ⚠ statsmodels 未安裝，跳過 G*Power 計算")
        return {}

    # pkl_dir 未指定時，退而沿用 h5_dir（向後相容，但 stimulus 仍可能找不到）
    _pkl_dir = Path(pkl_dir) if pkl_dir is not None else Path(h5_dir)

    if roi_configs is None:
        roi_configs = [
            dict(lock='response', roi='Motor',      freq=(4, 8),  time=(-0.300, 0.050)),
            dict(lock='stimulus', roi='Perceptual', freq=(8, 13), time=( 0.100, 0.300)),
        ]

    results = {}
    pwr_calc = TTestPower()

    for cfg in roi_configs:
        lock    = cfg['lock']
        roi     = cfg['roi']
        freq_lo, freq_hi = cfg['freq']
        t_lo,   t_hi     = cfg['time']
        lock_cap  = lock.capitalize()
        roi_lower = roi.lower()

        diffs = []
        for sid in subject_ids:
            block_groups = ['Block7-11', 'Block12-16', 'Block17-21', 'Block22-26']
            sub_means_l, sub_means_r = [], []
            for blk in block_groups:
                # ── 依 lock 類型選擇正確的檔案格式與目錄 ──────────────────
                if lock == 'stimulus':
                    # Stimulus-locked Learning 資料以 .pkl 儲存於 pkl_dir
                    # 格式與 _find_and_load / 群體圖片路徑完全一致
                    fp_l = _pkl_dir / f'{sid}_learning_stimulus_{roi_lower}_{condition_left}_{blk}_ersp.pkl'
                    fp_r = _pkl_dir / f'{sid}_learning_stimulus_{roi_lower}_{condition_right}_{blk}_ersp.pkl'
                    if not fp_l.exists() or not fp_r.exists():
                        continue
                    try:
                        ersp_l, freqs, times, _ = _load_pkl(fp_l)
                        ersp_r, _,     _,     _ = _load_pkl(fp_r)
                    except Exception:
                        continue
                else:
                    # Response-locked Learning 資料以 .h5 儲存於 h5_dir
                    fp_l = Path(h5_dir) / f'{sid}_{lock_cap}_Learning_{blk}_{condition_left}_ERSP.h5'
                    fp_r = Path(h5_dir) / f'{sid}_{lock_cap}_Learning_{blk}_{condition_right}_ERSP.h5'
                    if not fp_l.exists() or not fp_r.exists():
                        continue
                    try:
                        ersp_l, freqs, times, _ = _load_h5_response(fp_l, roi)
                        ersp_r, _,     _,     _ = _load_h5_response(fp_r, roi)
                    except Exception:
                        continue
                # ──────────────────────────────────────────────────────────
                f_mask = (freqs >= freq_lo) & (freqs <= freq_hi)
                t_mask = (times >= t_lo)    & (times <= t_hi)
                sub_means_l.append(ersp_l[np.ix_(f_mask, t_mask)].mean())
                sub_means_r.append(ersp_r[np.ix_(f_mask, t_mask)].mean())
            if not sub_means_l:
                continue
            diffs.append(np.mean(sub_means_l) - np.mean(sub_means_r))

        if len(diffs) < 3:
            print(f"  ⚠ G*Power [{lock}|{roi}]: 有效受試者數不足（{len(diffs)}），跳過")
            continue

        diffs    = np.array(diffs)
        mean_d   = diffs.mean()
        std_d    = diffs.std(ddof=1)
        cohens_d = mean_d / std_d if std_d > 1e-12 else 0.0
        # 單尾（alternative='larger' → 預期 reg > ran）
        n_req = pwr_calc.solve_power(
            effect_size=abs(cohens_d), alpha=alpha, power=power_target,
            alternative='larger')
        n_req = int(np.ceil(n_req))
        pwr_now = pwr_calc.solve_power(
            effect_size=abs(cohens_d), alpha=alpha, nobs=len(diffs),
            alternative='larger')
        results[(lock, roi)] = dict(
            d=cohens_d, n_required=n_req,
            power_at_n=pwr_now, n_current=len(diffs),
            mean_diff=mean_d, std_diff=std_d,
        )

    # ── 印出表格 ──
    if results:
        print("\n" + "╔" + "═"*90 + "╗")
        print("║  G*Power 樣本數估算  (one-tailed, α=0.05, target power=0.80)" + " "*27 + "║")
        print("╠" + "═"*90 + "╣")
        hdr = f"  {'Lock':<10} {'ROI':<22} {'Cohen d':>10} {'Mean diff':>10} {'Std diff':>10} {'N req.':>8} {'Power@N_now':>12}"
        print("║" + hdr + "║")
        print("╠" + "─"*90 + "╣")
        for (lk, roi), v in results.items():
            row = (f"  {lk:<10} {roi:<22} {v['d']:>10.3f} {v['mean_diff']:>10.4f} "
                   f"{v['std_diff']:>10.4f} {v['n_required']:>8} {v['power_at_n']:>12.3f}")
            print("║" + row + "║")
        print("╚" + "═"*90 + "╝")

    return results


# ============================================================
# 1. 儲存個別受試者 ERSP（由 ersp.py 的 save_for_group 呼叫）
# ============================================================

def save_subject_ersp(ersp_data, subject_id, condition, phase, lock_type,
                      freqs, times, roi_name, nave=-1,
                      output_dir=r'C:\Experiment\Result\h5'):
    """
    儲存單一受試者的 ROI-averaged ERSP 資料（.pkl）。

    Parameters
    ----------
    ersp_data : np.ndarray  shape (n_freqs, n_times)
    subject_id, condition, phase, lock_type, roi_name : str
    freqs, times : np.ndarray
    output_dir : str

    Returns
    -------
    str : 儲存的檔案路徑
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{subject_id}_{phase}_{lock_type}_{roi_name}_{condition}_ersp.pkl"
    )
    filepath = output_path / filename

    data_dict = {
        'ersp'      : ersp_data,
        'freqs'     : freqs,
        'times'     : times,
        'nave'      : nave,
        'subject_id': subject_id,
        'condition' : condition,
        'phase'     : phase,
        'lock_type' : lock_type,
        'roi_name'  : roi_name,
    }

    with open(filepath, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"  ✓ Saved (pkl): {filepath}")
    return str(filepath)


# ============================================================
# 2. 資料載入（自動判斷格式）
# ============================================================

def _load_pkl(filepath):
    """讀取 stimulus lock 的 .pkl，回傳 (ersp_2d, freqs, times, nave)。"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['ersp'], data['freqs'], data['times'], int(data.get('nave', -1))


def _load_h5_response(filepath, roi_name):
    """
    讀取 response lock 的 MNE AverageTFR .h5（由 asrt_response_ersp_from_epochs.py 存出）。
    h5 內部為 MNE 原生格式（mnepython group），包含全通道 ERSP。
    此函數讀取後提取指定 ROI 的頻道並平均，回傳 (ersp_2d, freqs, times)。

    roi_name : 'theta' | 'alpha'（大小寫不拘）
    """
    # 大小寫不敏感查詢
    roi_channels = None
    for key in ROI_GROUPS:
        if key.lower() == roi_name.lower():
            roi_channels = ROI_GROUPS[key]
            break

    if roi_channels is None:
        raise ValueError(
            f"未知 ROI: '{roi_name}'，可用: {list(ROI_GROUPS.keys())}"
        )

    # 用 MNE 讀取 AverageTFR
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tfr_list = mne.time_frequency.read_tfrs(str(filepath))

    tfr = tfr_list[0] if isinstance(tfr_list, list) else tfr_list

    # 找 ROI 頻道索引
    ch_names = [ch.upper() for ch in tfr.ch_names]
    roi_idx  = []
    missing  = []
    for ch in roi_channels:
        try:
            roi_idx.append(ch_names.index(ch.upper()))
        except ValueError:
            missing.append(ch)

    if missing:
        print(f"    ⚠ ROI channels {missing} not found in file, skipped")

    if not roi_idx:
        raise ValueError(
            f"ROI '{roi_name}' 的所有頻道均不在檔案中。\n"
            f"  檔案頻道: {tfr.ch_names}\n"
            f"  ROI 頻道: {roi_channels}"
        )

    # data shape: (n_channels, n_freqs, n_times)
    power = tfr.data[roi_idx].mean(axis=0)   # (n_freqs, n_times)
    freqs = tfr.freqs
    times = tfr.times

    return power, freqs, times, int(tfr.nave)


def _find_and_load(pkl_dir, subject_id, lock_type, phase,
                   test_type, group_label, trial_type, roi_name,
                   h5_dir=None):
    """
    載入單一受試者、單一 Block 組的 ERSP 資料。
    （Learning 用；Testing 請用 _load_subject_testing_pooled）

    Stimulus lock → .pkl（從 pkl_dir 讀取）
      Learning : {sub}_learning_stimulus_{roi}_{trial}_{block}_ersp.pkl

    Response lock → h5py 格式（從 h5_dir 讀取，若未提供則沿用 pkl_dir）
      Learning : {sub}_Response_Learning_{block}_{trial}_ERSP.h5

    Returns
    -------
    (ersp_2d, freqs, times)
    """
    if lock_type == 'stimulus':
        data_path = Path(pkl_dir)
    else:
        data_path = Path(h5_dir) if h5_dir is not None else Path(pkl_dir)
    roi_lower = roi_name.lower()

    if lock_type == 'stimulus':
        cond     = f"{trial_type}_{group_label}"
        filename = f"{subject_id}_{phase.lower()}_{lock_type}_{roi_lower}_{cond}_ersp.pkl"
        filepath = data_path / filename
        if not filepath.exists():
            # Triplet mode fallback: Regular→high, Random→low
            alt_map  = {'regular_high': 'regular_high', 'random_high': 'random_high', 'random_low': 'random_low', 'Regular': 'regular_high', 'Random': 'random_low', 'regular': 'regular_high', 'random': 'random_low', 'high': 'regular_high', 'low': 'random_low'}
            alt_type = alt_map.get(trial_type)
            if alt_type:
                alt_cond = f"{alt_type}_{group_label}"
                alt_file = f"{subject_id}_{phase.lower()}_{lock_type}_{roi_lower}_{alt_cond}_ersp.pkl"
                alt_path = data_path / alt_file
                if alt_path.exists():
                    return _load_pkl(alt_path)   # 4-tuple (ersp, freqs, times, nave)
            raise FileNotFoundError(f"File not found: {filepath}")
        return _load_pkl(filepath)   # 4-tuple

    else:  # response lock
        filename = (
            f"{subject_id}_Response_Learning_{group_label}_{trial_type}_ERSP.h5"
        )
        filepath = data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return _load_h5_response(filepath, roi_name)   # 4-tuple


def _extract_block_num(filepath):
    """從檔名中提取 block 起始編號，用於排序。例如 Block27-28 → 27。"""
    import re
    m = re.search(r'Block(\d+)-\d+', filepath.name)
    return int(m.group(1)) if m else 9999


def _load_subject_testing_pair(pkl_dir, subject_id, lock_type,
                                test_type, trial_type, roi_name,
                                pair='first', h5_dir=None):
    """
    Testing 階段（反平衡設計）：

    對某位受試者，找出其在 test_type 下所有可用 block，
    按 block 號排序後分為前兩個（first pair）與後兩個（second pair），
    各自平均後回傳。

    pair : 'first' | 'second'
        'first'  → 排序後前半 block（第一次接觸）
        'second' → 排序後後半 block（第二次接觸）

    Returns
    -------
    (ersp_2d, freqs, times, block_names)
    """
    if lock_type == 'stimulus':
        data_path = Path(pkl_dir)
    else:
        data_path = Path(h5_dir) if h5_dir is not None else Path(pkl_dir)
    roi_lower = roi_name.lower()
    cond_name = 'MotorTest' if test_type == 'motor' else 'PerceptualTest'

    # ── AllBlocks 分支：直接讀 AllBlocks h5 ──────────────────
    if pair == 'allblocks':
        lock_cap = lock_type.capitalize()
        fp_ab = data_path / f'{subject_id}_{lock_cap}_{cond_name}_AllBlocks_{trial_type}_ERSP.h5'
        if not fp_ab.exists():
            # stimulus-locked AllBlocks 可能在 pkl_dir
            fp_ab_alt = Path(pkl_dir) / f'{subject_id}_{lock_cap}_{cond_name}_AllBlocks_{trial_type}_ERSP.h5'
            if fp_ab_alt.exists():
                fp_ab = fp_ab_alt
            else:
                raise FileNotFoundError(f"AllBlocks 檔案不存在：{fp_ab}")
        ersp, f, t, nave = _load_h5_response(fp_ab, roi_name)
        return ersp, f, t, [fp_ab.name], [nave]
    # ─────────────────────────────────────────────────────────

    if lock_type == 'stimulus':
        pattern = (
            f"{subject_id}_testing_stimulus_{roi_lower}"
            f"_{test_type}_{trial_type}_Block*_ersp.pkl"
        )
    else:
        pattern = (
            f"{subject_id}_Response_{cond_name}_Block*_{trial_type}_ERSP.h5"
        )

    all_files = sorted(data_path.glob(pattern), key=_extract_block_num)

    # Triplet fallback: Regular→high, Random→low
    if not all_files and lock_type == 'stimulus':
        alt_map  = {'regular_high': 'regular_high', 'random_high': 'random_high', 'random_low': 'random_low', 'Regular': 'regular_high', 'Random': 'random_low', 'regular': 'regular_high', 'random': 'random_low', 'high': 'regular_high', 'low': 'random_low'}
        alt_type = alt_map.get(trial_type)
        if alt_type:
            alt_pattern = (
                f"{subject_id}_testing_stimulus_{roi_lower}"
                f"_{test_type}_{alt_type}_Block*_ersp.pkl"
            )
            all_files = sorted(data_path.glob(alt_pattern), key=_extract_block_num)

    if not all_files:
        raise FileNotFoundError(
            f"找不到任何 {test_type}/{trial_type} block 檔案\n"
            f"  搜尋: {data_path / pattern}"
        )

    n    = len(all_files)
    half = max(n // 2, 1)
    files_to_use = all_files[:half] if pair == 'first' else all_files[half:]

    if not files_to_use:
        files_to_use = all_files  # fallback：block 數為奇數時後半可能為空

    ersp_list   = []
    nave_list   = []
    block_names = []
    freqs = times = None

    for fp in files_to_use:
        try:
            if lock_type == 'stimulus':
                ersp, f, t, nave = _load_pkl(fp)
            else:
                ersp, f, t, nave = _load_h5_response(fp, roi_name)
            ersp_list.append(ersp)
            nave_list.append(nave)
            block_names.append(fp.name)
            if freqs is None:
                freqs, times = f, t
        except Exception as e:
            print(f"      ⚠ Skipping {fp.name}: {e}")

    if not ersp_list:
        raise FileNotFoundError(f"pair={pair}: all blocks failed to load for {subject_id}")

    return np.mean(ersp_list, axis=0), freqs, times, block_names, nave_list


# ============================================================
# 3. 批次載入多位受試者
# ============================================================

def _load_group_data(subject_ids, pkl_dir, lock_type, phase,
                     test_type, group_label, trial_type, roi_name,
                     h5_dir=None, silent=False):
    """
    載入所有受試者在某個條件（某個 Block 組）的 ERSP。

    Parameters
    ----------
    silent : bool
        若為 True，不印出讀檔進度（用於 section vmax 預掃描）。

    Returns
    -------
    ersp_array : np.ndarray  (n_loaded, n_freqs, n_times) or None
    freqs, times : np.ndarray or None
    loaded_ids, missing_ids : list
    """
    ersp_list  = []
    nave_list  = []
    freqs = times = None
    loaded_ids = []
    missing_ids = []

    for sid in subject_ids:
        try:
            ersp, f, t, nave = _find_and_load(
                pkl_dir, sid, lock_type, phase,
                test_type, group_label, trial_type, roi_name,
                h5_dir=h5_dir
            )
            ersp_list.append(ersp)
            nave_list.append(nave)
            loaded_ids.append(sid)
            if freqs is None:
                freqs, times = f, t
            if not silent:
                print(f"    ✓ {sid}  shape={ersp.shape}  n_trials={nave}")
        except FileNotFoundError as e:
            if not silent:
                print(f"    ✗ {sid}: {e}")
            missing_ids.append(sid)
        except Exception as e:
            if not silent:
                print(f"    ✗ {sid}: load failed ({e})")
            missing_ids.append(sid)

    if not ersp_list:
        return None, None, None, [], missing_ids, []

    return np.array(ersp_list), freqs, times, loaded_ids, missing_ids, nave_list


# ============================================================
# 4. 繪圖
# ============================================================


# ============================================================
# 3b. Section-level colorbar 計算函式
# ============================================================

def _compute_block_section_vmax(
    subject_ids, pkl_dir, h5_dir,
    lock_type, phase, block_groups,
    condition_pairs, roi_names,
    test_type=None,
    x_min=-0.5, x_max=0.5,
    percentile=95,
    label='Section'
):
    """
    跨 condition_pair × ROI × block_group 計算全域 vmax。
    用於讓同一 section 內的所有投影片共用同一個 colorbar。

    Parameters
    ----------
    block_groups : list of str
        e.g. ['Block7-11'] 或 ['Block7-11', 'Block22-26']
    condition_pairs : list of (str, str)
        e.g. [('regular_high','random_low'), ...]
    roi_names : list of str
        e.g. ['motor', 'perceptual']

    Returns
    -------
    vmax_cond, vmax_diff : float or None
    """
    cond_vals, diff_vals = [], []
    n_loaded = 0

    for roi_name in roi_names:
        roi_lower = roi_name.lower()
        for (c1, c2) in condition_pairs:
            for bg in block_groups:
                a1, _, times, ids1, _, _ = _load_group_data(
                    subject_ids, pkl_dir, lock_type, phase,
                    test_type, bg, c1, roi_lower, h5_dir=h5_dir,
                    silent=True
                )
                a2, _, _, ids2, _, _ = _load_group_data(
                    subject_ids, pkl_dir, lock_type, phase,
                    test_type, bg, c2, roi_lower, h5_dir=h5_dir,
                    silent=True
                )
                if a1 is None or a2 is None:
                    continue
                common = [s for s in ids1 if s in ids2]
                if not common:
                    continue
                a1c = a1[[ids1.index(s) for s in common]]
                a2c = a2[[ids2.index(s) for s in common]]
                rm = a1c.mean(axis=0)
                rn = a2c.mean(axis=0)
                diff = rm - rn
                t_mask = (times >= x_min) & (times <= x_max)
                cond_vals.append(np.abs(rm[:, t_mask]).ravel())
                cond_vals.append(np.abs(rn[:, t_mask]).ravel())
                diff_vals.append(np.abs(diff[:, t_mask]).ravel())
                n_loaded += 1

    if not cond_vals:
        print(f"  ⚠ [SECTION COLORBAR] [{label}] 無有效資料，各圖獨立計算")
        return None, None

    vmax_cond = np.percentile(np.concatenate(cond_vals), percentile)
    vmax_diff = np.percentile(np.concatenate(diff_vals), percentile)

    print(f"\n{'═'*62}")
    print(f"  SECTION COLORBAR LOG  >>>  {label}")
    print(f"  掃描: {len(roi_names)} ROI × {len(condition_pairs)} 條件對 × {len(block_groups)} block組  (n_matrix={n_loaded})")
    print(f"  vmax_cond = ±{vmax_cond:.4f} dB   ← 套用至所有 Condition 面板")
    print(f"  vmax_diff = ±{vmax_diff:.4f} dB   ← 套用至所有 Difference 面板")
    print(f"{'═'*62}\n")

    return vmax_cond, vmax_diff


def _compute_epoch_diff_section_vmax(
    subject_ids, pkl_dir, h5_dir,
    lock_type, all_conditions, roi_names,
    e1_label='Block7-11', e4_label='Block22-26',
    x_min=-0.5, x_max=0.5,
    percentile=95,
    label='Epoch4-Epoch1 Section'
):
    """
    跨 condition × ROI 計算 Epoch4-vs-Epoch1 比較的全域 vmax。
    用於讓 Section 3（Ep4-Ep1）的所有投影片共用同一個 colorbar。

    Parameters
    ----------
    all_conditions : list of str
        e.g. ['regular_high', 'random_high', 'random_low']
    """
    cond_vals, diff_vals = [], []
    n_loaded = 0

    for roi_name in roi_names:
        roi_lower = roi_name.lower()
        for cond in all_conditions:
            a_e1, _, times, ids_e1, _, _ = _load_group_data(
                subject_ids, pkl_dir, lock_type, 'learning',
                None, e1_label, cond, roi_lower, h5_dir=h5_dir,
                silent=True
            )
            a_e4, _, _, ids_e4, _, _ = _load_group_data(
                subject_ids, pkl_dir, lock_type, 'learning',
                None, e4_label, cond, roi_lower, h5_dir=h5_dir,
                silent=True
            )
            if a_e1 is None or a_e4 is None:
                continue
            common = [s for s in ids_e1 if s in ids_e4]
            if not common:
                continue
            a_e1c = a_e1[[ids_e1.index(s) for s in common]]
            a_e4c = a_e4[[ids_e4.index(s) for s in common]]
            rm   = a_e4c.mean(axis=0)   # Epoch4
            rn   = a_e1c.mean(axis=0)   # Epoch1
            diff = rm - rn
            t_mask = (times >= x_min) & (times <= x_max)
            cond_vals.append(np.abs(rm[:, t_mask]).ravel())
            cond_vals.append(np.abs(rn[:, t_mask]).ravel())
            diff_vals.append(np.abs(diff[:, t_mask]).ravel())
            n_loaded += 1

    if not cond_vals:
        print(f"  ⚠ [SECTION COLORBAR] [{label}] 無有效資料，各圖獨立計算")
        return None, None

    vmax_cond = np.percentile(np.concatenate(cond_vals), percentile)
    vmax_diff = np.percentile(np.concatenate(diff_vals), percentile)

    print(f"\n{'═'*62}")
    print(f"  SECTION COLORBAR LOG  >>>  {label}")
    print(f"  掃描: {len(roi_names)} ROI × {len(all_conditions)} 條件  (n_matrix={n_loaded})")
    print(f"  vmax_cond = ±{vmax_cond:.4f} dB   ← 套用至 Epoch4 / Epoch1 面板")
    print(f"  vmax_diff = ±{vmax_diff:.4f} dB   ← 套用至 Epoch4-Epoch1 Diff 面板")
    print(f"{'═'*62}\n")

    return vmax_cond, vmax_diff


def _compute_allblocks_testing_vmax(
    subject_ids, pkl_dir, h5_dir,
    lock_type, test_type, condition_pairs, roi_names,
    x_min=-0.5, x_max=0.5,
    percentile=95,
    label=''
):
    """
    計算 AllBlocks Testing 全域 vmax。
    用於讓同一 test_type 的所有 AllBlocks 投影片共用 colorbar。
    """
    cond_vals, diff_vals = [], []
    n_loaded = 0
    lock_cap  = lock_type.capitalize()
    cond_name = 'MotorTest' if test_type == 'motor' else 'PerceptualTest'
    allblocks_dir = Path(pkl_dir) if lock_type == 'stimulus' else Path(h5_dir)

    for roi_name in roi_names:
        roi_lower = roi_name.lower()
        for (c1, c2) in condition_pairs:
            arr_l_list, arr_r_list = [], []
            times_ref = None
            for sid in subject_ids:
                fp_l = allblocks_dir / f'{sid}_{lock_cap}_{cond_name}_AllBlocks_{c1}_ERSP.h5'
                fp_r = allblocks_dir / f'{sid}_{lock_cap}_{cond_name}_AllBlocks_{c2}_ERSP.h5'
                if not fp_l.exists() or not fp_r.exists():
                    continue
                try:
                    el, _, t, _ = _load_h5_response(fp_l, roi_lower)
                    er, _, _, _ = _load_h5_response(fp_r, roi_lower)
                    arr_l_list.append(el)
                    arr_r_list.append(er)
                    if times_ref is None:
                        times_ref = t
                except Exception:
                    continue
            if not arr_l_list or times_ref is None:
                continue
            rm   = np.array(arr_l_list).mean(axis=0)
            rn   = np.array(arr_r_list).mean(axis=0)
            diff = rm - rn
            t_mask = (times_ref >= x_min) & (times_ref <= x_max)
            cond_vals.append(np.abs(rm[:, t_mask]).ravel())
            cond_vals.append(np.abs(rn[:, t_mask]).ravel())
            diff_vals.append(np.abs(diff[:, t_mask]).ravel())
            n_loaded += 1

    if not cond_vals:
        print(f"  ⚠ [SECTION COLORBAR] [{label}] 無有效資料，各圖獨立計算")
        return None, None

    vmax_cond = np.percentile(np.concatenate(cond_vals), percentile)
    vmax_diff = np.percentile(np.concatenate(diff_vals), percentile)

    print(f"\n{'═'*62}")
    print(f"  SECTION COLORBAR LOG  >>>  {label}")
    print(f"  掃描: {len(roi_names)} ROI × {len(condition_pairs)} 條件對  (n_matrix={n_loaded})")
    print(f"  vmax_cond = ±{vmax_cond:.4f} dB   ← 套用至所有 Condition 面板")
    print(f"  vmax_diff = ±{vmax_diff:.4f} dB   ← 套用至所有 Difference 面板")
    print(f"{'═'*62}\n")

    return vmax_cond, vmax_diff


def _compute_testing_pair_vmax(
    subject_ids, pkl_dir, h5_dir,
    lock_type, test_type, condition_pairs, roi_names,
    x_min=-0.5, x_max=0.5,
    percentile=95,
    label='Testing Early/Late Section'
):
    """
    計算 Testing Early/Late 比較的全域 vmax。

    使用 _load_subject_testing_pair 正確載入 first/second pair 資料，
    同時涵蓋 Early + Late × 所有 condition pair × 所有 ROI，
    確保同一 Testing section 的所有投影片共用同一 colorbar。
    """
    cond_vals, diff_vals = [], []
    n_loaded = 0

    for roi_name in roi_names:
        roi_lower = roi_name.lower()
        for (c1, c2) in condition_pairs:
            for pair in ('first', 'second'):
                arr_l, arr_r = [], []
                times_ref = None
                for sid in subject_ids:
                    try:
                        e1, _, t, _, _ = _load_subject_testing_pair(
                            pkl_dir, sid, lock_type, test_type, c1, roi_lower,
                            pair=pair, h5_dir=h5_dir
                        )
                        arr_l.append(e1)
                        if times_ref is None:
                            times_ref = t
                    except Exception:
                        pass
                    try:
                        e2, _, _, _, _ = _load_subject_testing_pair(
                            pkl_dir, sid, lock_type, test_type, c2, roi_lower,
                            pair=pair, h5_dir=h5_dir
                        )
                        arr_r.append(e2)
                    except Exception:
                        pass

                if not arr_l or not arr_r or times_ref is None:
                    continue

                rm   = np.array(arr_l).mean(axis=0)
                rn   = np.array(arr_r).mean(axis=0)
                diff = rm - rn
                t_mask = (times_ref >= x_min) & (times_ref <= x_max)
                cond_vals.append(np.abs(rm[:, t_mask]).ravel())
                cond_vals.append(np.abs(rn[:, t_mask]).ravel())
                diff_vals.append(np.abs(diff[:, t_mask]).ravel())
                n_loaded += 1

    if not cond_vals:
        print(f"  ⚠ [SECTION COLORBAR] [{label}] 無有效資料，各圖獨立計算")
        return None, None

    vmax_cond = np.percentile(np.concatenate(cond_vals), percentile)
    vmax_diff = np.percentile(np.concatenate(diff_vals), percentile)

    print(f"\n{'═'*62}")
    print(f"  SECTION COLORBAR LOG  >>>  {label}")
    print(f"  掃描: {len(roi_names)} ROI × {len(condition_pairs)} 條件對 × 2 pair  (n_matrix={n_loaded})")
    print(f"  vmax_cond = ±{vmax_cond:.4f} dB   ← Early/Late Condition 面板共用")
    print(f"  vmax_diff = ±{vmax_diff:.4f} dB   ← Early/Late Difference 面板共用")
    print(f"{'═'*62}\n")

    return vmax_cond, vmax_diff


def _load_h5_single_electrode(filepath, electrode_name):
    """讀取 MNE AverageTFR h5，提取單一電極 ERSP (n_freqs, n_times)。"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tfr_list = mne.time_frequency.read_tfrs(str(filepath))
    tfr = tfr_list[0] if isinstance(tfr_list, list) else tfr_list

    ch_upper = [ch.upper() for ch in tfr.ch_names]
    elec_upper = electrode_name.upper()
    if elec_upper not in ch_upper:
        raise ValueError(f"電極 '{electrode_name}' 不在檔案中。可用: {tfr.ch_names}")
    idx = ch_upper.index(elec_upper)
    return tfr.data[idx], tfr.freqs, tfr.times, int(tfr.nave)


def _plot_single_electrode_comparison(arr_left, arr_right, freqs, times,
                                       common_ids, suptitle, output_path,
                                       electrode_name,
                                       label_left='Regular', label_right='Random',
                                       nave_list_left=None, nave_list_right=None,
                                       vmax_cond=None, vmax_diff=None,
                                       do_permutation=False, n_permutations=1000):
    """產生單一電極群體比較圖：left | right | Difference。

    vmax_cond / vmax_diff:
        若提供，使用外部統一 colorbar（由電極 section 預掃描提供），
        確保同一電極的三對 triplet 投影片共用同一色軸。
        若為 None，各自從資料計算。
    """
    n_sub      = len(common_ids)
    left_mean  = arr_left.mean(axis=0)
    right_mean = arr_right.mean(axis=0)
    diff       = left_mean - right_mean

    # 建立 trial 數標注字串
    def _nave_str(nave_list):
        if not nave_list or all(n < 0 for n in nave_list):
            return ""
        valid = [n for n in nave_list if n >= 0]
        m = int(np.mean(valid))
        lo, hi = min(valid), max(valid)
        return f"\n(avg {m} trials/sub, range {lo}–{hi})" if lo != hi else f"\n({m} trials/sub)"

    x_min, x_max = -0.5, 0.5
    t_mask = (times >= x_min) & (times <= x_max)

    # 外部 section vmax 優先；否則各自計算
    if vmax_cond is None:
        combined  = np.concatenate([left_mean[:, t_mask].ravel(), right_mean[:, t_mask].ravel()])
        vmax_cond = np.percentile(np.abs(combined), 95)
    if vmax_diff is None:
        vmax_diff = np.percentile(np.abs(diff[:, t_mask].ravel()), 95)
    if vmax_cond < 1e-10: vmax_cond = 1e-10
    if vmax_diff < 1e-10: vmax_diff = 1e-10

    # ── Permutation test ───────────────────────────────────────
    sig_mask = None
    n_sig = n_total = 0
    if do_permutation and n_sub >= 3:
        try:
            diff_per_sub = arr_left - arr_right
            _, clusters, cl_pv, _ = permutation_cluster_1samp_test(
                diff_per_sub, n_permutations=n_permutations,
                threshold=None, tail=0, n_jobs=1,
                verbose=False, out_type='mask')
            sig_mask = np.zeros_like(diff, dtype=bool)
            n_total = len(clusters)
            for c, pv in zip(clusters, cl_pv):
                if pv < 0.05:
                    sig_mask |= c
                    n_sig += 1
            print(f"    [ElecPerm] {electrode_name} Sig clusters: {n_sig}/{n_total}")
        except Exception as _e:
            print(f"    ⚠ Perm test failed ({electrode_name}): {_e}")
    # ─────────────────────────────────────────────────────────────

    lv_c = np.linspace(-vmax_cond, vmax_cond, 20)
    lv_d = np.linspace(-vmax_diff, vmax_diff, 20)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        (axes[0], left_mean,  f'{label_left}\n(N={n_sub}){_nave_str(nave_list_left)}',   lv_c, vmax_cond, 'Power (dB)',           None),
        (axes[1], right_mean, f'{label_right}\n(N={n_sub}){_nave_str(nave_list_right)}', lv_c, vmax_cond, 'Power (dB)',           None),
        (axes[2], diff,       f'Difference ({label_left} - {label_right})',               lv_d, vmax_diff, 'Power Difference (dB)', sig_mask),
    ]
    for ax, data, title, lv, vm, cb_lbl, smask in panels:
        im = ax.contourf(times, freqs, data, levels=lv,
                         cmap='RdBu_r', vmin=-vm, vmax=vm, extend='both')
        if smask is not None and smask.any():
            ax.contour(times, freqs, smask.astype(float),
                       levels=[0.5], colors='black', linewidths=1.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax.axhline(8,  color='white', linestyle=':', linewidth=1, alpha=0.6)
        ax.axhline(13, color='white', linestyle=':', linewidth=1, alpha=0.6)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim([x_min, x_max])
        plt.colorbar(im, ax=ax, label=cb_lbl)
    if do_permutation and sig_mask is not None:
        axes[2].set_title(axes[2].get_title() + '\n(black outline: p<0.05)', fontsize=10)

    fig.suptitle(f'Group ERSP Comparison\n{suptitle} | Electrode: {electrode_name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Saved: {output_path}")
    return dict(n_sig=n_sig, n_total=n_total)


def run_single_electrode_group_analysis(subject_ids, electrodes,
                                         h5_dir, output_dir,
                                         label_left='Regular', label_right='Random',
                                         condition_left='Regular', condition_right='Random',
                                         stim_h5_dir=None,
                                         do_permutation_test=False,
                                         n_permutations=1000,
                                         _triplet_expanded=False,
                                         _electrode_vmaxes=None):
    """
    對指定電極執行群體單一電極 ERSP 分析。
    同時處理 Response-locked（h5 全通道）和 Stimulus-locked（h5 全通道，需先跑 option 15）。

    Triplet 模式（condition_left='regular_high' 等）時，自動跑三對比較：
      regular_high vs random_low、regular_high vs random_high、random_high vs random_low
    """
    # Triplet 模式：自動展開為三對（只展開一次，由下方 _triplet_expanded 控制）

    h5_path      = Path(h5_dir)
    stim_h5_path = Path(stim_h5_dir) if stim_h5_dir else h5_path
    output_path  = Path(output_dir)

    # ── Triplet 模式：自動展開為三對，遞迴呼叫（只展開一次）──
    _TRIPLET_CONDITIONS = {'regular_high', 'random_high', 'random_low'}
    if not _triplet_expanded and (
        condition_left in _TRIPLET_CONDITIONS or condition_right in _TRIPLET_CONDITIONS
    ):
        _pairs = [
            ('regular_high', 'random_low',  'Regular High', 'Random Low'),
            ('regular_high', 'random_high', 'Regular High', 'Random High'),
            ('random_high',  'random_low',  'Random High',  'Random Low'),
        ]

        # ── 電極 Section Colorbar 預掃描 ──
        # 原則：「出現在同一張投影片上的面板才共用 colorbar」
        #
        # key: (roi_group, lock_key, phase) → (vmax_cond, vmax_diff)
        #   phase = 'learning' | 'motor' | 'perceptual' | 'mp_diff'
        #
        # 同一 sub-ROI 的電極（如 Fz+FCz）共用 vmax，讓空間比較有意義。
        # Response 和 Stimulus 分開掃描，因為兩者為不同投影片組。
        # Learning、Testing Motor、Testing Perceptual、MP Diff 各自獨立掃描。

        # 電極 → 最小包含 sub-ROI
        _ELECTRODE_TO_ROI = {}
        for _e in electrodes:
            _best_roi, _best_size = None, 9999
            for _roi, _elecs in ROI_GROUPS.items():
                if _e in _elecs and len(_elecs) < _best_size:
                    _best_roi = _roi.lower()
                    _best_size = len(_elecs)
            _ELECTRODE_TO_ROI[_e] = _best_roi if _best_roi else _e.lower()

        _ROI_TO_ELECTRODES = {}
        for _e, _roi in _ELECTRODE_TO_ROI.items():
            _ROI_TO_ELECTRODES.setdefault(_roi, []).append(_e)

        _electrode_vmaxes = {}   # key: (electrode, lock_key, phase) → (vc, vd)
        x_min, x_max = -0.5, 0.5
        _LEARNING_GROUPS_LABELS = ['Block7-11', 'Block12-16', 'Block17-21', 'Block22-26']
        _all_pair_tuples = [(cl, cr) for cl, cr, _, _ in _pairs]
        _sp = stim_h5_path if stim_h5_dir else h5_path
        _lock_paths = [('Response', h5_path), ('Stimulus', _sp)]

        for _roi_group, _roi_electrodes in _ROI_TO_ELECTRODES.items():
            for _lk_key, _lk_path in _lock_paths:

                # ── Phase 1: Learning ──
                _c_l, _d_l = [], []
                for electrode in _roi_electrodes:
                    for blk_label in _LEARNING_GROUPS_LABELS:
                        for _cl_use, _cr_use in _all_pair_tuples:
                            for sid in subject_ids:
                                fp_l = _lk_path / f'{sid}_{_lk_key}_Learning_{blk_label}_{_cl_use}_ERSP.h5'
                                fp_r = _lk_path / f'{sid}_{_lk_key}_Learning_{blk_label}_{_cr_use}_ERSP.h5'
                                try:
                                    if fp_l.exists():
                                        el, _, t, _ = _load_h5_single_electrode(fp_l, electrode)
                                        t_mask = (t >= x_min) & (t <= x_max)
                                        _c_l.append(np.abs(el[:, t_mask]).ravel())
                                        if fp_r.exists():
                                            er, _, _, _ = _load_h5_single_electrode(fp_r, electrode)
                                            _c_l.append(np.abs(er[:, t_mask]).ravel())
                                            _d_l.append(np.abs((el - er)[:, t_mask]).ravel())
                                except Exception:
                                    pass
                _vc_l = float(np.percentile(np.concatenate(_c_l), 95)) if _c_l else None
                _vd_l = float(np.percentile(np.concatenate(_d_l), 95)) if _d_l else (_vc_l * 0.4 if _vc_l else None)

                # ── Phase 2+3: Testing Motor / Perceptual ──
                _testing_vmaxes = {}
                for _phase, _cn in [('motor', 'MotorTest'), ('perceptual', 'PerceptualTest')]:
                    _c_t, _d_t = [], []
                    for electrode in _roi_electrodes:
                        for _cl_use, _cr_use in _all_pair_tuples:
                            for sid in subject_ids:
                                fp_l = _lk_path / f'{sid}_{_lk_key}_{_cn}_AllBlocks_{_cl_use}_ERSP.h5'
                                fp_r = _lk_path / f'{sid}_{_lk_key}_{_cn}_AllBlocks_{_cr_use}_ERSP.h5'
                                try:
                                    if fp_l.exists():
                                        el, _, t, _ = _load_h5_single_electrode(fp_l, electrode)
                                        t_mask = (t >= x_min) & (t <= x_max)
                                        _c_t.append(np.abs(el[:, t_mask]).ravel())
                                        if fp_r.exists():
                                            er, _, _, _ = _load_h5_single_electrode(fp_r, electrode)
                                            _c_t.append(np.abs(er[:, t_mask]).ravel())
                                            _d_t.append(np.abs((el - er)[:, t_mask]).ravel())
                                except Exception:
                                    pass
                                # AllBlocks 掃完，繼續下一個 sid
                            # ── Early/Late blocks（同一條件對，不嵌套在 sid 迴圈內）──
                            for sid in subject_ids:
                                for _fp_glob in _lk_path.glob(f'{sid}_{_lk_key}_{_cn}_Block*_{_cl_use}_ERSP.h5'):
                                    _fp_r2 = _lk_path / _fp_glob.name.replace(_cl_use, _cr_use)
                                    try:
                                        el2, _, t2, _ = _load_h5_single_electrode(_fp_glob, electrode)
                                        t_mask2 = (t2 >= x_min) & (t2 <= x_max)
                                        _c_t.append(np.abs(el2[:, t_mask2]).ravel())
                                        if _fp_r2.exists():
                                            er2, _, _, _ = _load_h5_single_electrode(_fp_r2, electrode)
                                            _c_t.append(np.abs(er2[:, t_mask2]).ravel())
                                            _d_t.append(np.abs((el2 - er2)[:, t_mask2]).ravel())
                                    except Exception:
                                        pass
                    _vc_t = float(np.percentile(np.concatenate(_c_t), 95)) if _c_t else None
                    _vd_t = float(np.percentile(np.concatenate(_d_t), 95)) if _d_t else (_vc_t * 0.4 if _vc_t else None)
                    _testing_vmaxes[_phase] = (_vc_t, _vd_t)

                # ── Motor ↔ Perceptual colorbar 統一（同 lock type）────────
                # 讓同一 lock type 的 Motor Test 和 Perceptual Test 共用 colorbar，
                # 才能直觀對照兩個測驗條件的 ERSP 幅度差異。
                _vc_m, _vd_m = _testing_vmaxes.get('motor',      (None, None))
                _vc_p, _vd_p = _testing_vmaxes.get('perceptual', (None, None))
                _vc_unified = max(v for v in [_vc_m, _vc_p] if v is not None) if any(v is not None for v in [_vc_m, _vc_p]) else None
                _vd_unified = max(v for v in [_vd_m, _vd_p] if v is not None) if any(v is not None for v in [_vd_m, _vd_p]) else None
                if _vc_unified is not None:
                    _testing_vmaxes['motor']      = (_vc_unified, _vd_unified)
                    _testing_vmaxes['perceptual'] = (_vc_unified, _vd_unified)
                    print(f"    [Motor↔Perceptual 統一] vmax_cond=±{_vc_unified:.4f} vmax_diff=±{_vd_unified:.4f} dB")
                # ─────────────────────────────────────────────────────────────

                # ── Phase 4: Motor-Perceptual Diff ──
                _c_mp, _d_mp = [], []
                for electrode in _roi_electrodes:
                    for _cl_use, _cr_use in _all_pair_tuples:
                        for sid in subject_ids:
                            try:
                                fp_ml = _lk_path / f'{sid}_{_lk_key}_MotorTest_AllBlocks_{_cl_use}_ERSP.h5'
                                fp_pl = _lk_path / f'{sid}_{_lk_key}_PerceptualTest_AllBlocks_{_cl_use}_ERSP.h5'
                                fp_mr = _lk_path / f'{sid}_{_lk_key}_MotorTest_AllBlocks_{_cr_use}_ERSP.h5'
                                fp_pr = _lk_path / f'{sid}_{_lk_key}_PerceptualTest_AllBlocks_{_cr_use}_ERSP.h5'
                                if fp_ml.exists() and fp_pl.exists():
                                    eml, _, t, _ = _load_h5_single_electrode(fp_ml, electrode)
                                    epl, _, _, _ = _load_h5_single_electrode(fp_pl, electrode)
                                    t_mask = (t >= x_min) & (t <= x_max)
                                    mp_l = eml - epl
                                    _c_mp.append(np.abs(mp_l[:, t_mask]).ravel())
                                    if fp_mr.exists() and fp_pr.exists():
                                        emr, _, _, _ = _load_h5_single_electrode(fp_mr, electrode)
                                        epr, _, _, _ = _load_h5_single_electrode(fp_pr, electrode)
                                        mp_r = emr - epr
                                        _c_mp.append(np.abs(mp_r[:, t_mask]).ravel())
                                        _d_mp.append(np.abs((mp_l - mp_r)[:, t_mask]).ravel())
                            except Exception:
                                pass
                _vc_mp = float(np.percentile(np.concatenate(_c_mp), 95)) if _c_mp else None
                _vd_mp = float(np.percentile(np.concatenate(_d_mp), 95)) if _d_mp else (_vc_mp * 0.4 if _vc_mp else None)

                # Log
                print(f"\n  {'═'*62}")
                print(f"  SECTION COLORBAR LOG  >>>  ROI [{_roi_group}] | {_lk_key}-locked")
                print(f"  電極: {_roi_electrodes}")
                print(f"  Learning:      ±{_vc_l:.4f} / ±{_vd_l:.4f} dB" if _vc_l else "  Learning:      無資料")
                for _ph in ('motor', 'perceptual'):
                    _vc_t, _vd_t = _testing_vmaxes[_ph]
                    print(f"  Testing {_ph:10s}: ±{_vc_t:.4f} / ±{_vd_t:.4f} dB" if _vc_t else f"  Testing {_ph}: 無資料")
                print(f"  MP Diff:       ±{_vc_mp:.4f} / inter=±{_vd_mp:.4f} dB" if _vc_mp else "  MP Diff:       無資料")
                print(f"  {'═'*62}")

                # 儲存：同 ROI 的每個電極都拿到同一組 vmax
                for electrode in _roi_electrodes:
                    _electrode_vmaxes[(electrode, _lk_key, 'learning')]    = (_vc_l, _vd_l)
                    _electrode_vmaxes[(electrode, _lk_key, 'motor')]       = _testing_vmaxes['motor']
                    _electrode_vmaxes[(electrode, _lk_key, 'perceptual')]  = _testing_vmaxes['perceptual']
                    _electrode_vmaxes[(electrode, _lk_key, 'mp_diff')]     = (_vc_mp, _vd_mp)

        # ── Cross-lock Colorbar 統一（僅 Learning 階段）────────────────
        # 學習階段需要跨 lock type 比較（Response vs Stimulus 同一 ROI），
        # 才能論述「Theta ERD 只在 Response-locked 出現，Stimulus-locked 沒有」。
        # 測驗階段不統一：測驗階段的比較是同 lock type 內 Motor Test vs Perceptual Test。
        print(f"\n  {'─'*60}")
        print(f"  Cross-lock vmax 統一（Response ↔ Stimulus，僅 learning phase）")
        for electrode in electrodes:
            vc_r, vd_r = _electrode_vmaxes.get((electrode, 'Response', 'learning'), (None, None))
            vc_s, vd_s = _electrode_vmaxes.get((electrode, 'Stimulus', 'learning'), (None, None))
            if vc_r is None and vc_s is None:
                continue
            vc_c = max(v for v in [vc_r, vc_s] if v is not None)
            vd_c = max(v for v in [vd_r, vd_s] if v is not None)
            _electrode_vmaxes[(electrode, 'Response', 'learning')] = (vc_c, vd_c)
            _electrode_vmaxes[(electrode, 'Stimulus', 'learning')] = (vc_c, vd_c)
            print(f"    {electrode} | learning: ±{vc_c:.4f} / ±{vd_c:.4f} dB"
                  f"  (R=±{vc_r or 0:.4f}, S=±{vc_s or 0:.4f})")
        print(f"  {'─'*60}")

        # Motor-Perceptual Diff 也統一 Response ↔ Stimulus
        # 論述「Response-locked 出現 Theta ERD、Stimulus-locked 出現相反 Alpha ERS」，
        # 統一後可直接比較兩個 lock type 的效果幅度。
        print(f"\n  {'─'*60}")
        print(f"  Cross-lock vmax 統一（Response ↔ Stimulus，mp_diff phase）")
        for electrode in electrodes:
            vc_r, vd_r = _electrode_vmaxes.get((electrode, 'Response', 'mp_diff'), (None, None))
            vc_s, vd_s = _electrode_vmaxes.get((electrode, 'Stimulus', 'mp_diff'), (None, None))
            if vc_r is None and vc_s is None:
                continue
            vc_c = max(v for v in [vc_r, vc_s] if v is not None)
            vd_c = max(v for v in [vd_r, vd_s] if v is not None)
            _electrode_vmaxes[(electrode, 'Response', 'mp_diff')] = (vc_c, vd_c)
            _electrode_vmaxes[(electrode, 'Stimulus', 'mp_diff')] = (vc_c, vd_c)
            print(f"    {electrode} | mp_diff: ±{vc_c:.4f} / ±{vd_c:.4f} dB"
                  f"  (R=±{vc_r or 0:.4f}, S=±{vc_s or 0:.4f})")
        print(f"  {'─'*60}")
        for _cl, _cr, _ll, _lr in _pairs:
            print(f"\n{'─'*60}")
            print(f"  單一電極群體分析：{_ll} vs {_lr}")
            print(f"{'─'*60}")
            # 每個條件對使用獨立子資料夾，避免不同條件對的圖片混在一起
            run_single_electrode_group_analysis(
                subject_ids=subject_ids, electrodes=electrodes,
                h5_dir=h5_dir, output_dir=output_dir,
                label_left=_ll, label_right=_lr,
                condition_left=_cl, condition_right=_cr,
                stim_h5_dir=stim_h5_dir,
                do_permutation_test=do_permutation_test,
                n_permutations=n_permutations,
                _triplet_expanded=True,
                _electrode_vmaxes=_electrode_vmaxes,
            )
        return

    for electrode in electrodes:
        print(f"\n{'='*60}")
        print(f"  Single Electrode: {electrode}")
        print(f"{'='*60}")

        elec_out = output_path / f'electrode_{electrode}'
        pair_label = f'{condition_left}_vs_{condition_right}'

        for lock_type in ('Response', 'Stimulus'):
            print(f"\n  [{lock_type}-locked]")
            search_path = stim_h5_path if lock_type == 'Stimulus' else h5_path
            _lk_lower = lock_type.lower()

            # 各 phase 的子資料夾（在電極資料夾內依 phase × lock_type 分類）
            _sec_learning  = elec_out / f'learning_{_lk_lower}'
            _sec_mp_diff   = elec_out / 'motor_perceptual_diff'
            _sec_learning.mkdir(parents=True, exist_ok=True)
            _sec_mp_diff.mkdir(parents=True, exist_ok=True)

            # 取得此電極此 lock_type 的 section vmax（依 phase 分別取用）
            def _get_vmax(phase):
                if not _electrode_vmaxes:
                    return None, None
                return _electrode_vmaxes.get((electrode, lock_type, phase), (None, None))

            _evc_l,  _evd_l  = _get_vmax('learning')
            _evc_m,  _evd_m  = _get_vmax('motor')
            _evc_p,  _evd_p  = _get_vmax('perceptual')
            _emp_vc, _emp_vd = _get_vmax('mp_diff')

            if _evc_l is not None:
                print(f"  [colorbar] {electrode}|{lock_type} Learning=±{_evc_l:.4f} "
                      f"Motor=±{_evc_m:.4f} Percept=±{_evc_p:.4f} MPDiff=±{_emp_vc:.4f} dB")



            # ── Learning ──
            for (bs, be) in LEARNING_GROUPS:
                gl = f'Block{bs}-{be}'
                arr_l, arr_r, ids_found = [], [], []
                freqs = times = None

                nave_l_list, nave_r_list = [], []
                for sid in subject_ids:
                    fp_l = search_path / f'{sid}_{lock_type}_Learning_{gl}_{condition_left}_ERSP.h5'
                    fp_r = search_path / f'{sid}_{lock_type}_Learning_{gl}_{condition_right}_ERSP.h5'
                    if not fp_l.exists() or not fp_r.exists():
                        print(f"    ✗ {sid} {gl}: 檔案不存在（{lock_type}）")
                        continue
                    try:
                        el, f, t, nave_l = _load_h5_single_electrode(fp_l, electrode)
                        er, _, _, nave_r = _load_h5_single_electrode(fp_r, electrode)
                        arr_l.append(el); arr_r.append(er)
                        nave_l_list.append(nave_l); nave_r_list.append(nave_r)
                        if freqs is None: freqs, times = f, t
                        ids_found.append(sid)
                    except Exception as e:
                        print(f"    ✗ {sid}: {e}")

                if not arr_l:
                    continue

                suptitle = f'Learning | {gl} | {lock_type}-locked | {label_left} vs {label_right}'
                out_name = f'group_learning_{lock_type.lower()}_{electrode}_{gl}_{condition_left}_vs_{condition_right}_comparison.png'
                _elec_result_l = _plot_single_electrode_comparison(
                    np.array(arr_l), np.array(arr_r), freqs, times,
                    ids_found, suptitle, _sec_learning / out_name, electrode,
                    label_left=label_left, label_right=label_right,
                    nave_list_left=nave_l_list, nave_list_right=nave_r_list,
                    vmax_cond=_evc_l, vmax_diff=_evd_l,
                    do_permutation=do_permutation_test, n_permutations=n_permutations)
                if do_permutation_test:
                    _log_perm('SingleElec', electrode, lock_type, 'learning', gl,
                              f'{condition_left}_vs_{condition_right}',
                              len(ids_found), _elec_result_l.get('n_sig', 0), _elec_result_l.get('n_total', 0))

            # ── Epoch 4 vs Epoch 1：跨條件比較 ──────────────────────
            # 結構：左 = condition_left (Ep4−Ep1)，右 = condition_right (Ep4−Ep1)
            # 差值 = [cond_left (Ep4−Ep1)] − [cond_right (Ep4−Ep1)]
            _E1_GL = 'Block7-11'
            _E4_GL = 'Block22-26'
            _ep_data = {}  # cond → list of (e4 - e1) per subject
            _ep_nave = {}  # cond → list of nave
            _ep_ids  = {}  # cond → list of sid
            _ep_freq = _ep_time = None

            for _cond, _lbl in [(condition_left, label_left), (condition_right, label_right)]:
                _ep_data[_cond] = []
                _ep_nave[_cond] = []
                _ep_ids[_cond]  = []
                for sid in subject_ids:
                    fp_e1 = search_path / f'{sid}_{lock_type}_Learning_{_E1_GL}_{_cond}_ERSP.h5'
                    fp_e4 = search_path / f'{sid}_{lock_type}_Learning_{_E4_GL}_{_cond}_ERSP.h5'
                    if not fp_e1.exists() or not fp_e4.exists():
                        continue
                    try:
                        e1, fe, te, nv1 = _load_h5_single_electrode(fp_e1, electrode)
                        e4, _,  _,  nv4  = _load_h5_single_electrode(fp_e4, electrode)
                        _ep_data[_cond].append(e4 - e1)
                        _ep_nave[_cond].append((nv1 + nv4) // 2)
                        _ep_ids[_cond].append(sid)
                        if _ep_freq is None:
                            _ep_freq, _ep_time = fe, te
                    except Exception as _ex:
                        print(f'    ✗ {sid} Epoch4vsEpoch1 {_cond}: {_ex}')

            _ids_common_ep = sorted(set(_ep_ids.get(condition_left, [])) &
                                    set(_ep_ids.get(condition_right, [])))
            if _ids_common_ep:
                def _ep_arr(cond):
                    idx = [_ep_ids[cond].index(s) for s in _ids_common_ep]
                    return np.array([_ep_data[cond][i] for i in idx])

                _arr_ep_l = _ep_arr(condition_left)
                _arr_ep_r = _ep_arr(condition_right)
                _suptitle_e = (
                    f'Learning | Epoch 4 − Epoch 1 | {lock_type}-locked | '
                    f'{label_left} vs {label_right}'
                )
                _out_e = (
                    f'group_learning_{lock_type.lower()}_{electrode}'
                    f'_epoch4_minus_epoch1_{condition_left}_vs_{condition_right}_comparison.png'
                )
                _elec_result_ep = _plot_single_electrode_comparison(
                    _arr_ep_l, _arr_ep_r, _ep_freq, _ep_time,
                    _ids_common_ep, _suptitle_e, _sec_learning / _out_e, electrode,
                    label_left=f'{label_left}\n(Ep4−Ep1)',
                    label_right=f'{label_right}\n(Ep4−Ep1)',
                    vmax_cond=_evc_l, vmax_diff=_evd_l,
                    do_permutation=do_permutation_test, n_permutations=n_permutations)
                if do_permutation_test:
                    _log_perm('SingleElec-Ep4E1', electrode, lock_type, 'learning', 'Ep4-Ep1',
                              f'{condition_left}_vs_{condition_right}',
                              len(_ids_common_ep), _elec_result_ep.get('n_sig', 0), _elec_result_ep.get('n_total', 0))

            # ── Testing ──
            for test_type in ('motor', 'perceptual'):
                cond_name = 'MotorTest' if test_type == 'motor' else 'PerceptualTest'
                _sec_testing = elec_out / f'testing_{test_type}_{_lk_lower}'
                _sec_testing.mkdir(parents=True, exist_ok=True)

                # 取得每位受試者的 block 對應（用 condition_left 掃描）
                sub_blocks = {}
                for sid in subject_ids:
                    files = sorted(
                        search_path.glob(f'{sid}_{lock_type}_{cond_name}_Block*_{condition_left}_ERSP.h5'),
                        key=lambda fp: int(''.join(filter(str.isdigit,
                            fp.stem.split('Block')[1].split('_')[0].split('-')[0])))
                    )
                    if files:
                        n = len(files); half = max(n // 2, 1)
                        sub_blocks[sid] = {'first': files[:half], 'second': files[half:]}

                pair_labels = {
                    'first':  f'Early {test_type.capitalize()}',
                    'second': f'Late {test_type.capitalize()}',
                }

                for pair_key, pair_desc in pair_labels.items():
                    arr_l, arr_r, ids_found = [], [], []
                    nave_list_l, nave_list_r = [], []
                    freqs = times = None

                    for sid in subject_ids:
                        if sid not in sub_blocks or not sub_blocks[sid].get(pair_key):
                            continue
                        try:
                            els, ers, nls, nrs = [], [], [], []
                            for fp_l in sub_blocks[sid][pair_key]:
                                fp_r = Path(str(fp_l).replace(
                                    f'_{condition_left}_', f'_{condition_right}_'))
                                if not fp_r.exists():
                                    continue
                                el, f, t, nave_l = _load_h5_single_electrode(fp_l, electrode)
                                er, _, _, nave_r = _load_h5_single_electrode(fp_r, electrode)
                                els.append(el); ers.append(er)
                                nls.append(nave_l); nrs.append(nave_r)
                                if freqs is None: freqs, times = f, t
                            if els and ers:
                                arr_l.append(np.mean(els, axis=0))
                                arr_r.append(np.mean(ers, axis=0))
                                # 跨 block 加總 nave（同一受試者多個 block 的 trial 數加總）
                                nave_list_l.append(sum(n for n in nls if n >= 0))
                                nave_list_r.append(sum(n for n in nrs if n >= 0))
                                ids_found.append(sid)
                        except Exception as e:
                            print(f"    ✗ {sid}: {e}")

                    if not arr_l:
                        continue

                    suptitle = f'Testing | {test_type.capitalize()} | {pair_desc} | {lock_type}-locked | {label_left} vs {label_right}'
                    out_name = f'group_testing_{lock_type.lower()}_{electrode}_{test_type}_{pair_key}_{condition_left}_vs_{condition_right}_comparison.png'
                    _elec_result_t = _plot_single_electrode_comparison(
                        np.array(arr_l), np.array(arr_r), freqs, times,
                        ids_found, suptitle, _sec_testing / out_name, electrode,
                        label_left=label_left, label_right=label_right,
                        nave_list_left=nave_list_l, nave_list_right=nave_list_r,
                        vmax_cond=_evc_m if test_type == "motor" else _evc_p, vmax_diff=_evd_m if test_type == "motor" else _evd_p,
                        do_permutation=do_permutation_test, n_permutations=n_permutations)
                    if do_permutation_test:
                        _log_perm('SingleElec', electrode, lock_type, f'testing_{test_type}', pair_key,
                                  f'{condition_left}_vs_{condition_right}',
                                  len(ids_found), _elec_result_t.get('n_sig', 0), _elec_result_t.get('n_total', 0))

                # ── 各別 Block 群體圖 ──
                all_block_labels = set()
                for sid in subject_ids:
                    for fp in search_path.glob(f'{sid}_{lock_type}_{cond_name}_Block*_{condition_left}_ERSP.h5'):
                        all_block_labels.add(fp.stem.split('_')[3])
                all_block_labels = sorted(all_block_labels,
                    key=lambda b: int(''.join(filter(str.isdigit, b.split('-')[0]))))

                for blk_label in all_block_labels:
                    arr_l, arr_r, ids_found = [], [], []
                    nave_list_l, nave_list_r = [], []
                    freqs = times = None
                    for sid in subject_ids:
                        fp_l = search_path / f'{sid}_{lock_type}_{cond_name}_{blk_label}_{condition_left}_ERSP.h5'
                        fp_r = search_path / f'{sid}_{lock_type}_{cond_name}_{blk_label}_{condition_right}_ERSP.h5'
                        if not fp_l.exists() or not fp_r.exists():
                            continue
                        try:
                            el, f, t, nave_l = _load_h5_single_electrode(fp_l, electrode)
                            er, _, _, nave_r = _load_h5_single_electrode(fp_r, electrode)
                            arr_l.append(el); arr_r.append(er)
                            nave_list_l.append(nave_l); nave_list_r.append(nave_r)
                            if freqs is None: freqs, times = f, t
                            ids_found.append(sid)
                        except Exception as e:
                            print(f"    ✗ {sid} {blk_label}: {e}")
                    if not arr_l:
                        continue
                    suptitle = f'Testing | {test_type.capitalize()} | {blk_label} | {lock_type}-locked | {label_left} vs {label_right}'
                    out_name = f'group_testing_{lock_type.lower()}_{electrode}_{test_type}_{blk_label}_{condition_left}_vs_{condition_right}_comparison.png'
                    _plot_single_electrode_comparison(
                        np.array(arr_l), np.array(arr_r), freqs, times,
                        ids_found, suptitle, _sec_testing / out_name, electrode,
                        label_left=label_left, label_right=label_right,
                        nave_list_left=nave_list_l, nave_list_right=nave_list_r,
                        vmax_cond=_evc_m if test_type == "motor" else _evc_p, vmax_diff=_evd_m if test_type == "motor" else _evd_p)

            # ── Pooled Testing (AllBlocks) 單一電極 ──
            for _test_type in ('motor', 'perceptual'):
                _cond_name = 'MotorTest' if _test_type == 'motor' else 'PerceptualTest'
                # AllBlocks 有自己的 _sec_testing（不沿用 Early/Late 迴圈的值）
                _sec_testing = elec_out / f'testing_{_test_type}_{_lk_lower}'
                _sec_testing.mkdir(parents=True, exist_ok=True)

                arr_l, arr_r, ids_found = [], [], []
                nave_list_l, nave_list_r = [], []
                freqs = times = None

                for sid in subject_ids:
                    fp_l = search_path / f'{sid}_{lock_type}_{_cond_name}_AllBlocks_{condition_left}_ERSP.h5'
                    fp_r = search_path / f'{sid}_{lock_type}_{_cond_name}_AllBlocks_{condition_right}_ERSP.h5'
                    print(f"    [{sid}] 搜尋路徑: {search_path}")
                    print(f"    [{sid}] fp_l: {fp_l.name} → {'✓ 存在' if fp_l.exists() else '✗ 不存在'}")
                    print(f"    [{sid}] fp_r: {fp_r.name} → {'✓ 存在' if fp_r.exists() else '✗ 不存在'}")
                    if not fp_l.exists() or not fp_r.exists():
                        print(f"    ✗ {sid} {_cond_name} AllBlocks: 檔案不存在（{lock_type}）")
                        continue
                    try:
                        el, f, t, nave_l = _load_h5_single_electrode(fp_l, electrode)
                        er, _, _, nave_r = _load_h5_single_electrode(fp_r, electrode)
                        arr_l.append(el); arr_r.append(er)
                        nave_list_l.append(nave_l); nave_list_r.append(nave_r)
                        if freqs is None: freqs, times = f, t
                        ids_found.append(sid)
                        print(f"    ✓ {sid} {_cond_name} AllBlocks 讀取成功")
                    except Exception as e:
                        print(f"    ✗ {sid} 讀取失敗: {e}")

                if not arr_l:
                    continue

                suptitle = f'Testing Pooled | {_test_type.capitalize()} | AllBlocks | {lock_type}-locked | {label_left} vs {label_right}'
                out_name = f'group_pooled_{lock_type.lower()}_{electrode}_{_test_type}_{condition_left}_vs_{condition_right}_comparison.png'
                _elec_result_ab = _plot_single_electrode_comparison(
                    np.array(arr_l), np.array(arr_r), freqs, times,
                    ids_found, suptitle, _sec_testing / out_name, electrode,
                    label_left=label_left, label_right=label_right,
                    nave_list_left=nave_list_l, nave_list_right=nave_list_r,
                    vmax_cond=_evc_m if _test_type == "motor" else _evc_p, vmax_diff=_evd_m if _test_type == "motor" else _evd_p,
                    do_permutation=do_permutation_test, n_permutations=n_permutations)
                if do_permutation_test:
                    _log_perm('SingleElec-AllBlocks', electrode, lock_type, f'testing_{_test_type}', 'allblocks',
                              f'{condition_left}_vs_{condition_right}',
                              len(ids_found), _elec_result_ab.get('n_sig', 0), _elec_result_ab.get('n_total', 0))

            # ── Motor-Perceptual Diff 單一電極 ──
            _motor_reg, _motor_ran = {}, {}
            _percept_reg, _percept_ran = {}, {}

            for sid in subject_ids:
                for _cn, _store_l, _store_r in [
                    ('MotorTest', _motor_reg, _motor_ran),
                    ('PerceptualTest', _percept_reg, _percept_ran),
                ]:
                    fp_l = search_path / f'{sid}_{lock_type}_{_cn}_AllBlocks_{condition_left}_ERSP.h5'
                    fp_r = search_path / f'{sid}_{lock_type}_{_cn}_AllBlocks_{condition_right}_ERSP.h5'
                    if fp_l.exists():
                        try: _store_l[sid], _, _, _ = _load_h5_single_electrode(fp_l, electrode)
                        except Exception: pass
                    if fp_r.exists():
                        try: _store_r[sid], _, _, _ = _load_h5_single_electrode(fp_r, electrode)
                        except Exception: pass

            _common_mp = [s for s in subject_ids
                          if s in _motor_reg and s in _motor_ran
                          and s in _percept_reg and s in _percept_ran]
            if _common_mp:
                _fp_ref = search_path / f'{_common_mp[0]}_{lock_type}_MotorTest_AllBlocks_{condition_left}_ERSP.h5'
                _freqs_mp = _times_mp = None
                if _fp_ref.exists():
                    try: _, _freqs_mp, _times_mp, _ = _load_h5_single_electrode(_fp_ref, electrode)
                    except Exception: pass
                if _freqs_mp is not None:
                    _arr_reg_diff = np.array([_motor_reg[s] - _percept_reg[s] for s in _common_mp])
                    _arr_ran_diff = np.array([_motor_ran[s] - _percept_ran[s] for s in _common_mp])
                    _mp_out = _sec_mp_diff / f'group_motor_perceptual_diff_{lock_type.lower()}_{electrode}_{condition_left}_vs_{condition_right}.png'
                    _suptitle_mp = f'Motor-Perceptual Diff | {lock_type}-locked | Electrode: {electrode}'
                    _n = len(_common_mp)
                    _reg_m = _arr_reg_diff.mean(axis=0)
                    _ran_m = _arr_ran_diff.mean(axis=0)
                    _inter = _reg_m - _ran_m
                    _x_min, _x_max = -0.5, 0.5
                    _tm = (_times_mp >= _x_min) & (_times_mp <= _x_max)

                    # 使用預掃描的跨 triplet pair 共用 vmax；無則各自計算
                    if _emp_vc is not None:
                        _vc = _emp_vc
                        _vi = _emp_vd if _emp_vd is not None else np.percentile(np.abs(_inter[:, _tm].ravel()), 95)
                        print(f"    [colorbar] Motor-Perceptual Diff 使用 section vmax: ±{_vc:.4f} / inter=±{_vi:.4f} dB")
                    else:
                        _vc = np.percentile(np.abs(np.concatenate([_reg_m[:, _tm].ravel(), _ran_m[:, _tm].ravel()])), 95)
                        _vi = np.percentile(np.abs(_inter[:, _tm].ravel()), 95)
                    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
                    for _ax, _dat, _tit, _vm, _lv, _cbl in [
                        (_axes[0], _reg_m, f'{label_left} Motor-Perceptual\n(N={_n})', _vc, np.linspace(-_vc,_vc,20), 'Power diff (dB)'),
                        (_axes[1], _ran_m, f'{label_right} Motor-Perceptual\n(N={_n})', _vc, np.linspace(-_vc,_vc,20), 'Power diff (dB)'),
                        (_axes[2], _inter, f'Interaction\n({label_left} M-P) - ({label_right} M-P)', _vi, np.linspace(-_vi,_vi,20), 'Power diff (dB)'),
                    ]:
                        _im = _ax.contourf(_times_mp, _freqs_mp, _dat, levels=_lv,
                                           cmap='RdBu_r', vmin=-_vm, vmax=_vm, extend='both')
                        _ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
                        _ax.axhline(8,  color='white', linestyle=':', linewidth=1, alpha=0.6)
                        _ax.axhline(13, color='white', linestyle=':', linewidth=1, alpha=0.6)
                        _ax.set_xlabel('Time (s)', fontsize=11)
                        _ax.set_ylabel('Frequency (Hz)', fontsize=11)
                        _ax.set_title(_tit, fontsize=11, fontweight='bold')
                        _ax.set_xlim([_x_min, _x_max])
                        plt.colorbar(_im, ax=_ax, label=_cbl)
                    _fig.suptitle(f'Group {_suptitle_mp}', fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(str(_mp_out), dpi=300, bbox_inches='tight')
                    plt.close(_fig)
                    print(f"    ✓ Saved: {_mp_out}")

    print(f"\n✓ 單一電極群體分析完成，輸出: {output_path}")



def _plot_group_motor_perceptual_diff(arr_reg_diff, arr_ran_diff, freqs, times,
                                       common_ids, suptitle, output_path,
                                       label_left='Regular', label_right='Random',
                                       vmax_cond=None, vmax_inter=None,
                                       do_permutation=False, n_permutations=1000):
    """
    群體 Motor-Perceptual 差值比較圖。
    arr_reg_diff[i] = motor_reg[i] - percept_reg[i]  (n_sub, n_freqs, n_times)
    arr_ran_diff[i] = motor_ran[i] - percept_ran[i]

    Left:   mean(arr_reg_diff) = label_left Motor - label_left Perceptual
    Center: mean(arr_ran_diff) = label_right Motor - label_right Perceptual
    Right:  Interaction = Left - Center

    vmax_cond / vmax_inter:
        若提供，使用外部統一 colorbar（三對 triplet pair 預掃描提供）。
        若為 None，各自從資料計算。
    """
    n_sub      = len(common_ids)
    reg_mean   = arr_reg_diff.mean(axis=0)
    ran_mean   = arr_ran_diff.mean(axis=0)
    interaction = reg_mean - ran_mean

    x_min, x_max = -0.5, 0.5
    t_mask = (times >= x_min) & (times <= x_max)

    if vmax_cond is None:
        combined  = np.concatenate([reg_mean[:, t_mask].ravel(), ran_mean[:, t_mask].ravel()])
        vmax_cond = np.percentile(np.abs(combined), 95)
    vmax_int = vmax_inter
    if vmax_int is None:
        vmax_int = np.percentile(np.abs(interaction[:, t_mask].ravel()), 95)
    if vmax_cond < 1e-10: vmax_cond = 1e-10
    if vmax_int  < 1e-10: vmax_int  = 1e-10

    # ── Permutation tests ──────────────────────────────────────────
    sig_reg = sig_ran = sig_inter = None
    n_sig_reg = n_sig_ran = n_sig_inter = 0
    n_tot_reg = n_tot_ran = n_tot_inter = 0
    if do_permutation and n_sub >= 3:
        print(f"    Running Cluster Permutation Test (MP Diff, n_permutations={n_permutations})...")
        for arr, label_p in [(arr_reg_diff, label_left), (arr_ran_diff, label_right),
                              (arr_reg_diff - arr_ran_diff, 'Interaction')]:
            try:
                _, cls, cl_pv, _ = permutation_cluster_1samp_test(
                    arr, n_permutations=n_permutations,
                    threshold=None, tail=0, n_jobs=1,
                    verbose=False, out_type='mask')
                mask = np.zeros(arr.shape[1:], dtype=bool)
                n_s = sum(1 for c, pv in zip(cls, cl_pv) if pv < 0.05)
                for c, pv in zip(cls, cl_pv):
                    if pv < 0.05:
                        mask |= c
                if label_p == label_left:
                    sig_reg, n_sig_reg, n_tot_reg = mask, n_s, len(cls)
                elif label_p == label_right:
                    sig_ran, n_sig_ran, n_tot_ran = mask, n_s, len(cls)
                else:
                    sig_inter, n_sig_inter, n_tot_inter = mask, n_s, len(cls)
                print(f"      [{label_p}] Sig clusters: {n_s}/{len(cls)}")
            except Exception as e:
                print(f"      ⚠ Perm test failed ({label_p}): {e}")
    # ─────────────────────────────────────────────────────────────────

    lv_c = np.linspace(-vmax_cond, vmax_cond, 20)
    lv_i = np.linspace(-vmax_int,  vmax_int,  20)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        (axes[0], reg_mean,    f'{label_left} Motor-Perceptual\n(N={n_sub})', lv_c, vmax_cond, 'Power diff (dB)', sig_reg),
        (axes[1], ran_mean,    f'{label_right} Motor-Perceptual\n(N={n_sub})', lv_c, vmax_cond, 'Power diff (dB)', sig_ran),
        (axes[2], interaction, f'Interaction\n({label_left} M-P) - ({label_right} M-P)', lv_i, vmax_int, 'Power diff (dB)', sig_inter),
    ]
    for ax, data, title, lv, vm, cbl, sig_mask in panels:
        im = ax.contourf(times, freqs, data, levels=lv,
                         cmap='RdBu_r', vmin=-vm, vmax=vm, extend='both')
        if sig_mask is not None and sig_mask.any():
            ax.contour(times, freqs, sig_mask.astype(float),
                       levels=[0.5], colors='black', linewidths=1.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax.axhline(8,  color='white', linestyle=':', linewidth=1, alpha=0.6)
        ax.axhline(13, color='white', linestyle=':', linewidth=1, alpha=0.6)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim([x_min, x_max])
        plt.colorbar(im, ax=ax, label=cbl)
    if do_permutation:
        axes[2].set_title(axes[2].get_title() + '\n(black outline: p<0.05, cluster-corrected)', fontsize=10)

    fig.suptitle(f'Group Motor vs Perceptual Diff\n{suptitle}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'    ✓ Saved: {output_path}')
    return dict(n_sig_reg=n_sig_reg, n_tot_reg=n_tot_reg,
                n_sig_ran=n_sig_ran, n_tot_ran=n_tot_ran,
                n_sig_inter=n_sig_inter, n_tot_inter=n_tot_inter)


def _draw_ersp_panel(ax, ersp_2d, freqs, times, title, vmin, vmax, x_min=-0.5, x_max=0.2):
    if abs(vmax - vmin) < 1e-10: vmin, vmax = -1e-10, 1e-10
    levels = np.linspace(vmin, vmax, 20)
    im = ax.contourf(
        times, freqs, ersp_2d,
        levels=levels, cmap='RdBu_r',
        vmin=vmin, vmax=vmax, extend='both'
    )
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax.axhline(8,  color='white', linestyle=':', linewidth=1, alpha=0.6)
    ax.axhline(13, color='white', linestyle=':', linewidth=1, alpha=0.6)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim([x_min, x_max])
    return im


def _plot_group_block(arr_reg, arr_ran, freqs, times,
                      common_ids, suptitle, output_path,
                      do_permutation, n_permutations, lock_type='response',
                      vmax_cond=None, vmax_diff=None,
                      label_left='Regular', label_right='Random',
                      nave_list_left=None, nave_list_right=None):
    """
    產生單一 Block 組的群體比較圖：
      [Regular grand avg | Random grand avg | Difference (+ cluster outline)]

    Parameters
    ----------
    arr_reg, arr_ran : np.ndarray  (n_subjects, n_freqs, n_times)
    """
    n_sub    = len(common_ids)
    reg_mean = arr_reg.mean(axis=0)
    ran_mean = arr_ran.mean(axis=0)
    diff     = reg_mean - ran_mean   # Regular - Random

    # ── 建立 trial 數標注字串 ──
    def _nave_str(nave_list):
        if not nave_list or all(n < 0 for n in nave_list):
            return ""
        valid = [n for n in nave_list if n >= 0]
        m = int(np.mean(valid))
        lo, hi = min(valid), max(valid)
        return f"\n(avg {m} trials/sub, range {lo}–{hi})" if lo != hi else f"\n({m} trials/sub)"

    label_left_full  = f'{label_left}\n(N={n_sub}){_nave_str(nave_list_left)}'
    label_right_full = f'{label_right}\n(N={n_sub}){_nave_str(nave_list_right)}'

    # ── xlim ──
    x_min, x_max = -0.5, 0.5

    t_mask = (times >= x_min) & (times <= x_max)

    # ── Cluster Permutation Test ──
    sig_mask = None
    n_sig    = 0
    n_total  = 0

    if do_permutation and n_sub >= 3:
        print(f"    Running Cluster Permutation Test (n_permutations={n_permutations})...")
        try:
            diff_per_sub = arr_reg - arr_ran   # Regular - Random（跟 diff 一致）
            _, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
                diff_per_sub,
                n_permutations=n_permutations,
                threshold=None,
                tail=0,
                n_jobs=1,
                verbose=False,
                out_type='mask'
            )
            n_total  = len(clusters)
            sig_mask = np.zeros_like(diff, dtype=bool)
            for c, pv in zip(clusters, cluster_pv):
                if pv < 0.05:
                    sig_mask |= c
                    n_sig += 1
            print(f"    ✓ Significant clusters: {n_sig}/{n_total}")
        except Exception as e:
            print(f"    ⚠ Permutation test failed: {e}")

    elif do_permutation and n_sub < 3:
        print(f"    ⚠ Too few subjects ({n_sub} < 3), skipping Permutation Test")

    # ── 決定 colorbar 範圍（95th percentile，只從顯示範圍計算）──
    if vmax_cond is None:
        combined = np.concatenate([
            reg_mean[:, t_mask].ravel(),
            ran_mean[:, t_mask].ravel()
        ])
        vmax_cond = np.percentile(np.abs(combined), 95)
    if vmax_diff is None:
        vmax_diff = np.percentile(np.abs(diff[:, t_mask].ravel()), 95)
    vmin_cond = -vmax_cond
    vmin_diff = -vmax_diff

    # ── 繪圖 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = _draw_ersp_panel(axes[0], reg_mean, freqs, times,
                           label_left_full, vmin_cond, vmax_cond,
                           x_min=x_min, x_max=x_max)
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')

    im2 = _draw_ersp_panel(axes[1], ran_mean, freqs, times,
                           label_right_full, vmin_cond, vmax_cond,
                           x_min=x_min, x_max=x_max)
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')

    diff_title = f'Difference ({label_left} - {label_right})'
    if sig_mask is not None and np.any(sig_mask):
        diff_title += '\n(black outline: p<0.05, cluster-corrected)'
    im3 = _draw_ersp_panel(axes[2], diff, freqs, times,
                           diff_title, vmin_diff, vmax_diff,
                           x_min=x_min, x_max=x_max)
    if sig_mask is not None and np.any(sig_mask):
        axes[2].contour(times, freqs, sig_mask,
                        levels=[0.5], colors='black',
                        linewidths=2.0, linestyles='solid')
    plt.colorbar(im3, ax=axes[2], label='Power Difference (dB)')

    fig.suptitle(f'Group ERSP Comparison\n{suptitle}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Saved: {output_path}")

    return {
        'n_subjects'   : n_sub,
        'reg_mean'     : reg_mean,
        'ran_mean'     : ran_mean,
        'diff'         : diff,
        'sig_mask'     : sig_mask,
        'n_sig_clusters': n_sig,
        'n_clusters'    : n_total,
    }


# ============================================================
# 5. 主分析函數（對外介面）
# ============================================================

def group_ersp_analysis(subject_ids,
                        condition1='Regular',
                        condition2='Random',
                        phase='testing',
                        lock_type='response',
                        roi_name='theta',
                        pkl_dir=r'C:\Experiment\Result\triplet',
                        h5_dir=r'C:\Experiment\Result\triplet\h5',
                        output_dir='./group_ersp_results',
                        do_permutation_test=True,
                        n_permutations=1000,
                        test_type=None,
                        data_dir=None,
                        unified_colorbar=False,
                        display_label1=None,
                        display_label2=None,
                        external_vmax_cond=None,
                        external_vmax_diff=None,
                        external_vmax_ep4e1_cond=None,
                        external_vmax_ep4e1_diff=None):
    """
    群體 ERSP 分析。

    external_vmax_cond / external_vmax_diff:
        從 auto_group_ersp_analysis 的 section 預掃描傳入，
        覆蓋內部 unified_colorbar 掃描，確保跨 ROI × 跨 condition pair 一致。
    external_vmax_ep4e1_cond / external_vmax_ep4e1_diff:
        專用於 Epoch4-vs-Epoch1 比較圖的外部 colorbar。

    Learning：按 Block 分組（Block7-11, 12-16, 17-21, 22-26），每組一張圖。

    Testing：因反平衡設計，不同受試者的 Motor/Perceptual block 位置不同，
             改為每位受試者先把自己所有 Motor（或 Perceptual）block 平均，
             再跨受試者做群體分析，共產一張圖。

    Returns
    -------
    dict
    """
    # ── 自動轉換 display label ──
    _DISP = {
        'regular_high': 'Regular High',
        'random_high':  'Random High',
        'random_low':   'Random Low',
        'Regular':      'Regular',
        'Random':       'Random',
        'high':         'High',
        'low':          'Low',
    }
    if display_label1 is None:
        display_label1 = _DISP.get(condition1, condition1)
    if display_label2 is None:
        display_label2 = _DISP.get(condition2, condition2)
    # 向後相容：舊的 data_dir 參數若有傳入，同時當作 pkl_dir 和 h5_dir
    if data_dir is not None:
        pkl_dir = data_dir
        h5_dir  = data_dir

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    roi_lower = roi_name.lower()
    roi_cap   = roi_name.capitalize()

    print("\n" + "=" * 70)
    print("Group ERSP Analysis  v2.1")
    print("=" * 70)
    print(f"Subjects (N={len(subject_ids)}): {subject_ids}")
    print(f"Conditions: {condition1} vs {condition2}")
    print(f"Phase: {phase}  |  Lock: {lock_type}  |  ROI: {roi_cap}")
    if test_type:
        print(f"Test type: {test_type} (counterbalanced design → per-subject block average)")
    print(f"Data source (stim/pkl): {pkl_dir}")
    print(f"Data source (resp/h5):  {h5_dir}")
    print(f"Output dir: {output_path}")
    print("=" * 70)

    all_results = {}

    # ============================================================
    # Learning：按 Block 分組
    # ============================================================
    if phase.lower() == 'learning':

        # ── Colorbar 決策：外部優先，其次內部 unified_colorbar ──
        global_vmax_cond = None
        global_vmax_diff = None
        if external_vmax_cond is not None and external_vmax_diff is not None:
            global_vmax_cond = external_vmax_cond
            global_vmax_diff = external_vmax_diff
            print(f"  [colorbar] 使用外部 section-level colorbar:")
            print(f"             vmax_cond=±{global_vmax_cond:.4f} dB  vmax_diff=±{global_vmax_diff:.4f} dB")
        elif unified_colorbar:
            _xmin, _xmax = -0.5, 0.5
            _gvc, _gvd = 0.0, 0.0
            for (_bs, _be) in LEARNING_GROUPS:
                _gl = f"Block{_bs}-{_be}"
                _a1, _, _t, _i1, _, _nv1 = _load_group_data(subject_ids, pkl_dir, lock_type, phase, None, _gl, condition1, roi_lower, h5_dir=h5_dir)
                _a2, _, _, _i2, _, _nv2  = _load_group_data(subject_ids, pkl_dir, lock_type, phase, None, _gl, condition2, roi_lower, h5_dir=h5_dir)
                if _a1 is None or _a2 is None:
                    continue
                _c = [s for s in _i1 if s in _i2]
                if not _c:
                    continue
                _a1 = _a1[[_i1.index(s) for s in _c]]
                _a2 = _a2[[_i2.index(s) for s in _c]]
                _rm, _rn = _a1.mean(axis=0), _a2.mean(axis=0)
                _d = _rm - _rn
                _tm = (_t >= _xmin) & (_t <= _xmax)
                _gvc = max(_gvc, np.percentile(np.abs(np.concatenate([_rm[:, _tm].ravel(), _rn[:, _tm].ravel()])), 95))
                _gvd = max(_gvd, np.percentile(np.abs(_d[:, _tm].ravel()), 95))
            global_vmax_cond = _gvc if _gvc > 0 else None
            global_vmax_diff = _gvd if _gvd > 0 else None
            if global_vmax_cond is not None and global_vmax_diff is not None:
                print(f"  [colorbar] 內部掃描: vmax_cond={global_vmax_cond:.4f} dB, vmax_diff={global_vmax_diff:.4f} dB")
            else:
                print("  [colorbar] 內部掃描: 無有效資料，各圖獨立計算")

        for (blk_start, blk_end) in LEARNING_GROUPS:
            group_label = f"Block{blk_start}-{blk_end}"

            print(f"\n{'─'*60}")
            print(f"  {group_label}")
            print(f"{'─'*60}")

            print(f"\n  Loading {condition1}...")
            arr1, freqs, times, ids1, _, nave1 = _load_group_data(
                subject_ids, pkl_dir, lock_type, phase,
                None, group_label, condition1, roi_lower,
                h5_dir=h5_dir
            )

            print(f"\n  Loading {condition2}...")
            arr2, _, _, ids2, _, nave2 = _load_group_data(
                subject_ids, pkl_dir, lock_type, phase,
                None, group_label, condition2, roi_lower,
                h5_dir=h5_dir
            )

            if arr1 is None or arr2 is None:
                print(f"  ⚠  {group_label}: insufficient data, skipped")
                continue

            common_ids = [s for s in ids1 if s in ids2]
            if not common_ids:
                print(f"  ⚠  {group_label}: no common subjects, skipped")
                continue

            if len(common_ids) < len(ids1):
                arr1 = arr1[[ids1.index(s) for s in common_ids]]
                arr2 = arr2[[ids2.index(s) for s in common_ids]]
            # 對齊 nave 到 common_ids
            nave1_common = [nave1[ids1.index(s)] for s in common_ids]
            nave2_common = [nave2[ids2.index(s)] for s in common_ids]

            print(f"\n  Common subjects: {len(common_ids)}")

            suptitle = (
                f"Learning | {group_label} | "
                f"{lock_type.capitalize()}-locked | {roi_cap} ROI"
            )
            out_name  = (
                f"group_learning_{lock_type}_{roi_lower}_{group_label}_comparison.png"
            )
            block_result = _plot_group_block(
                arr1, arr2, freqs, times,
                common_ids, suptitle, output_path / out_name,
                do_permutation_test, n_permutations, lock_type=lock_type,
                vmax_cond=global_vmax_cond,
                vmax_diff=global_vmax_diff,
                label_left=display_label1 or condition1,
                label_right=display_label2 or condition2,
                nave_list_left=nave1_common,
                nave_list_right=nave2_common,
            )
            # ── 寫入全域摘要 ──
            if do_permutation_test:
                _log_perm(
                    label='ROI-Group', roi=roi_cap, lock=lock_type,
                    phase='learning', pair=group_label,
                    cond_pair=f'{condition1}_vs_{condition2}',
                    n_sub=len(common_ids),
                    n_sig=block_result.get('n_sig_clusters', 0),
                    n_total=block_result.get('n_clusters', 0),
                )
            all_results[group_label] = {
                **block_result,
                'subject_ids': common_ids,
                'freqs': freqs, 'times': times,
            }

        # ── Epoch 4 vs Epoch 1 群體比較（跨條件）──────────────────────────
        # 結構：左 = condition1 (Ep4−Ep1)，右 = condition2 (Ep4−Ep1)
        # 差值 = [cond1 (Ep4−Ep1)] − [cond2 (Ep4−Ep1)]
        _E1_LABEL = 'Block7-11'
        _E4_LABEL = 'Block22-26'
        _EPOCH_DISP = {
            'regular_high': 'Regular High', 'random_high': 'Random High',
            'random_low':   'Random Low',   'Regular':     'Regular',
            'Random':       'Random',       'high':        'High',
            'low':          'Low',
        }
        print(f"\n{'─'*60}")
        print(f"  Group Learning: Epoch 4 − Epoch 1（跨條件比較）")
        print(f"{'─'*60}")
        try:
            # 載入四組資料
            _ep_data = {}
            _ep_ids  = {}
            _ep_nave = {}
            _ep_freqs = _ep_times = None
            for _cond in [condition1, condition2]:
                _disp_c = _EPOCH_DISP.get(_cond, _cond)
                print(f"\n  Loading Epoch1 ({_E1_LABEL}) – {_cond}...")
                _a1, _f, _t, _ids1, _, _nv1 = _load_group_data(
                    subject_ids, pkl_dir, lock_type, phase,
                    None, _E1_LABEL, _cond, roi_lower, h5_dir=h5_dir)
                print(f"  Loading Epoch4 ({_E4_LABEL}) – {_cond}...")
                _a4, _,  _,  _ids4, _, _nv4 = _load_group_data(
                    subject_ids, pkl_dir, lock_type, phase,
                    None, _E4_LABEL, _cond, roi_lower, h5_dir=h5_dir)
                if _a1 is None or _a4 is None:
                    print(f"  ⚠ {_cond}: Epoch1 或 Epoch4 資料不足，跳過")
                    continue
                _common = [s for s in _ids1 if s in _ids4]
                if not _common:
                    print(f"  ⚠ {_cond}: 無共同受試者，跳過")
                    continue
                _a1c = _a1[[_ids1.index(s) for s in _common]]
                _a4c = _a4[[_ids4.index(s) for s in _common]]
                _ep_data[_cond] = _a4c - _a1c   # 時間維度的差值
                _ep_ids[_cond]  = _common
                _ep_nave[_cond] = [(_nv1[_ids1.index(s)] + _nv4[_ids4.index(s)]) // 2
                                   for s in _common]
                if _ep_freqs is None:
                    _ep_freqs, _ep_times = _f, _t

            if condition1 in _ep_data and condition2 in _ep_data:
                # 找兩個條件共同的受試者
                _ids_both = sorted(set(_ep_ids[condition1]) & set(_ep_ids[condition2]))
                if _ids_both:
                    def _ep_slice(cond):
                        idx = [_ep_ids[cond].index(s) for s in _ids_both]
                        return (_ep_data[cond][idx],
                                [_ep_nave[cond][i] for i in idx])

                    _arr_l, _nv_l = _ep_slice(condition1)
                    _arr_r, _nv_r = _ep_slice(condition2)
                    _disp1 = _EPOCH_DISP.get(condition1, condition1)
                    _disp2 = _EPOCH_DISP.get(condition2, condition2)
                    suptitle_e = (
                        f"Learning: Epoch 4 − Epoch 1 | "
                        f"{lock_type.capitalize()}-locked | {roi_cap} ROI | "
                        f"{_disp1} vs {_disp2}"
                    )
                    out_name_e = (
                        f"group_learning_{lock_type}_{roi_lower}"
                        f"_epoch4_minus_epoch1_{condition1}_vs_{condition2}_comparison.png"
                    )
                    _ep4e1_vc = external_vmax_ep4e1_cond
                    _ep4e1_vd = external_vmax_ep4e1_diff
                    if _ep4e1_vc is not None:
                        print(f"  [colorbar] Ep4-Ep1 使用外部 colorbar: ±{_ep4e1_vc:.4f} / ±{_ep4e1_vd:.4f} dB")
                    _ep4e1_result = _plot_group_block(
                        _arr_l, _arr_r, _ep_freqs, _ep_times,
                        _ids_both, suptitle_e, output_path / out_name_e,
                        do_permutation_test, n_permutations, lock_type=lock_type,
                        vmax_cond=_ep4e1_vc,
                        vmax_diff=_ep4e1_vd,
                        label_left=f'{_disp1}\n(Ep4−Ep1)',
                        label_right=f'{_disp2}\n(Ep4−Ep1)',
                        nave_list_left=_nv_l,
                        nave_list_right=_nv_r,
                    )
                    if do_permutation_test:
                        _log_perm(
                            label='ROI-Group', roi=roi_cap, lock=lock_type,
                            phase='learning', pair='Ep4-Ep1',
                            cond_pair=f'{condition1}_vs_{condition2}',
                            n_sub=len(_ids_both),
                            n_sig=_ep4e1_result.get('n_sig_clusters', 0),
                            n_total=_ep4e1_result.get('n_clusters', 0),
                        )
                    all_results[f'epoch4_minus_epoch1_{condition1}_vs_{condition2}'] = {
                        'freqs': _ep_freqs, 'times': _ep_times}
                    print(f"  ✓ Ep4-Ep1 saved: {out_name_e}")
                else:
                    print(f"  ⚠ Ep4-Ep1: 兩個條件無共同受試者")
            else:
                print(f"  ⚠ Ep4-Ep1: 資料不完整，跳過")
        except Exception as _e4e:
            print(f"  ✗ Epoch4 vs Epoch1: {_e4e}")
            import traceback; traceback.print_exc()

    # ============================================================
    # Testing：反平衡設計，按「第一對 block / 第二對 block」分兩張圖
    # ============================================================
    else:
        if not test_type:
            raise ValueError("Testing phase requires test_type ('motor' or 'perceptual')")

        tt_cap = test_type.capitalize()
        pair_labels = {
            'first'    : f'(Early {tt_cap} blocks)',
            'second'   : f'(Late {tt_cap} blocks)',
            'allblocks': f'(AllBlocks {tt_cap})',
        }

        # ── Colorbar 決策：外部優先，其次內部 unified_colorbar ──
        global_vmax_cond = None
        global_vmax_diff = None
        if external_vmax_cond is not None and external_vmax_diff is not None:
            global_vmax_cond = external_vmax_cond
            global_vmax_diff = external_vmax_diff
            print(f"  [colorbar] 使用外部 section-level colorbar:")
            print(f"             vmax_cond=±{global_vmax_cond:.4f} dB  vmax_diff=±{global_vmax_diff:.4f} dB")
        elif unified_colorbar:
            _xmin, _xmax = -0.5, 0.5
            _gvc, _gvd = 0.0, 0.0
            for _pair in ('first', 'second'):
                _el1, _el2, _times = [], [], None
                for sid in subject_ids:
                    try:
                        _e1, _f, _t, _, _ = _load_subject_testing_pair(
                            pkl_dir, sid, lock_type, test_type, condition1, roi_lower,
                            pair=_pair, h5_dir=h5_dir)
                        _el1.append(_e1)
                        if _times is None:
                            _times = _t
                    except Exception:
                        pass
                    try:
                        _e2, _f, _t, _, _ = _load_subject_testing_pair(
                            pkl_dir, sid, lock_type, test_type, condition2, roi_lower,
                            pair=_pair, h5_dir=h5_dir)
                        _el2.append(_e2)
                        if _times is None:
                            _times = _t
                    except Exception:
                        pass
                if not _el1 or not _el2 or _times is None:
                    continue
                _tm = (_times >= _xmin) & (_times <= _xmax)
                _rm = np.array(_el1).mean(axis=0)
                _rn = np.array(_el2).mean(axis=0)
                _d = _rm - _rn
                _gvc = max(_gvc, np.percentile(np.abs(np.concatenate([_rm[:, _tm].ravel(), _rn[:, _tm].ravel()])), 95))
                _gvd = max(_gvd, np.percentile(np.abs(_d[:, _tm].ravel()), 95))
            global_vmax_cond = _gvc if _gvc > 0 else None
            global_vmax_diff = _gvd if _gvd > 0 else None
            if global_vmax_cond is not None and global_vmax_diff is not None:
                print(f"  [colorbar] 內部掃描: vmax_cond={global_vmax_cond:.4f} dB, vmax_diff={global_vmax_diff:.4f} dB")
            else:
                print("  [colorbar] 內部掃描: 無有效資料，各圖獨立計算")
        for pair_key, pair_desc in pair_labels.items():

            print(f"\n{'─'*60}")
            print(f"  Testing / {tt_cap} / {pair_desc}")
            print(f"{'─'*60}")

            def _load_pair(trial_type, _pair=pair_key):
                ersp_list  = []
                freqs = times = None
                loaded_ids = []
                sub_block_info = {}
                nave_list = []
                for sid in subject_ids:
                    try:
                        ersp, f, t, blk_names, sub_nave = _load_subject_testing_pair(
                            pkl_dir, sid, lock_type,
                            test_type, trial_type, roi_lower,
                            pair=_pair, h5_dir=h5_dir
                        )
                        ersp_list.append(ersp)
                        loaded_ids.append(sid)
                        sub_block_info[sid] = blk_names
                        # 평균 nave（若同一受試者有多個 block，取平均代表）
                        valid_nave = [n for n in sub_nave if n >= 0]
                        nave_list.append(int(np.mean(valid_nave)) if valid_nave else -1)
                        if freqs is None:
                            freqs, times = f, t
                        _sub_nave = int(np.mean(valid_nave)) if valid_nave else -1
                        print(f"    ✓ {sid}  blocks={[b for b in blk_names]}  shape={ersp.shape}  n_trials={_sub_nave}")
                    except FileNotFoundError as e:
                        print(f"    ✗ {sid}: {e}")
                    except Exception as e:
                        print(f"    ✗ {sid}: load failed ({e})")
                if not ersp_list:
                    return None, None, None, [], {}, []
                return np.array(ersp_list), freqs, times, loaded_ids, sub_block_info, nave_list

            print(f"\n  Loading {condition1}...")
            arr1, freqs, times, ids1, info1, nave1 = _load_pair(condition1)

            print(f"\n  Loading {condition2}...")
            arr2, _, _, ids2, info2, nave2 = _load_pair(condition2)

            if arr1 is None or arr2 is None:
                print(f"  ⚠  {pair_desc}: insufficient data, skipped")
                continue

            common_ids = [s for s in ids1 if s in ids2]
            if not common_ids:
                print(f"  ⚠  {pair_desc}: no common subjects, skipped")
                continue

            if len(common_ids) < len(ids1):
                arr1 = arr1[[ids1.index(s) for s in common_ids]]
                arr2 = arr2[[ids2.index(s) for s in common_ids]]
            nave1_common = [nave1[ids1.index(s)] for s in common_ids]
            nave2_common = [nave2[ids2.index(s)] for s in common_ids]

            print(f"\n  Common subjects: {len(common_ids)}")

            suptitle = (
                f"Testing | {tt_cap} | {pair_desc} | "
                f"{lock_type.capitalize()}-locked | {roi_cap} ROI"
            )
            out_name = (
                f"group_testing_{lock_type}_{roi_lower}_{test_type}_{pair_key}_comparison.png"
            )
            block_result = _plot_group_block(
                arr1, arr2, freqs, times,
                common_ids, suptitle, output_path / out_name,
                do_permutation_test, n_permutations, lock_type=lock_type,
                vmax_cond=global_vmax_cond,
                vmax_diff=global_vmax_diff,
                label_left=display_label1 or condition1,
                label_right=display_label2 or condition2,
                nave_list_left=nave1_common,
                nave_list_right=nave2_common,
            )
            # ── 寫入全域摘要 ──
            if do_permutation_test:
                _log_perm(
                    label='ROI-Group', roi=roi_cap, lock=lock_type,
                    phase=f'testing_{test_type}', pair=pair_key,
                    cond_pair=f'{condition1}_vs_{condition2}',
                    n_sub=len(common_ids),
                    n_sig=block_result.get('n_sig_clusters', 0),
                    n_total=block_result.get('n_clusters', 0),
                )
            all_results[pair_key] = {
                **block_result,
                'subject_ids'  : common_ids,
                'freqs'        : freqs,
                'times'        : times,
                'block_info_c1': {s: info1[s] for s in common_ids if s in info1},
                'block_info_c2': {s: info2[s] for s in common_ids if s in info2},
            }

    # ── 摘要 ──
    print(f"\n{'='*70}")
    print(f"✓ Group analysis complete! {len(all_results)} unit(s) processed")
    for gl, res in all_results.items():
        if 'subject_ids' not in res:
            continue
        n_sig = res.get('n_sig_clusters', 0)
        print(f"  {gl}: N={len(res['subject_ids'])}  sig. clusters={n_sig}")
    print(f"✓ Results saved to: {output_path}")
    print(f"{'='*70}")

    return all_results


# ============================================================
# 6. 全自動群體分析（一次跑完所有組合）
# ============================================================

def auto_group_ersp_analysis(subject_ids,
                             pkl_dir=r'C:\Experiment\Result\triplet',
                             h5_dir=r'C:\Experiment\Result\triplet\h5',
                             output_dir=r'C:\Experiment\Result\group_ersp',
                             do_permutation_test=True,
                             n_permutations=1000,
                             data_dir=None,
                             unified_colorbar=False,
                             display_label1=None,
                             display_label2=None):
    """
    全自動群體 ERSP 分析。

    自動跑完以下所有組合：
      phases     : learning, testing
      test_types : motor, perceptual  （僅 testing）
      lock_types : stimulus, response
      roi_names  : theta, alpha

    共 12 個組合，每個組合產生 4 張 Block 比較圖。
    """

    combos = []
    for lock in ['stimulus', 'response']:
        for roi in [r.lower() for r in ROI_GROUPS.keys()]:
            combos.append({'phase': 'learning', 'test_type': None,
                           'lock_type': lock, 'roi_name': roi})
    for tt in ['motor', 'perceptual']:
        for lock in ['stimulus', 'response']:
            for roi in [r.lower() for r in ROI_GROUPS.keys()]:
                combos.append({'phase': 'testing', 'test_type': tt,
                               'lock_type': lock, 'roi_name': roi})

    total = len(combos)

    # 向後相容：舊的 data_dir 參數若有傳入，同時當作 pkl_dir 和 h5_dir
    if data_dir is not None:
        pkl_dir = data_dir
        h5_dir  = data_dir

    # 自動偵測 trial_type 條件名稱（從 h5_dir 掃描 Response 檔案）
    import glob as _glob
    detected_types = []
    sample_files = _glob.glob(os.path.join(h5_dir, f'{subject_ids[0]}_Response_*.h5'))
    for f in sample_files:
        parts = os.path.basename(f).replace('_ERSP.h5', '').split('_')
        # triplet 名稱是 random_high / random_low / regular_high（兩段），一般是 high / low（一段）
        if len(parts) >= 2 and parts[-2] in ('random', 'regular'):
            tt = f"{parts[-2]}_{parts[-1]}"
        else:
            tt = parts[-1]
        if tt not in detected_types:
            detected_types.append(tt)

    # Triplet 模式偵測：若含 regular_high / random_high / random_low，展開為三對
    _TRIPLET_SET = {'regular_high', 'random_high', 'random_low'}
    if _TRIPLET_SET.issubset(set(detected_types)):
        condition_pairs = [
            ('regular_high', 'random_low'),
            ('regular_high', 'random_high'),
            ('random_high',  'random_low'),
        ]
    elif len(detected_types) >= 2:
        if detected_types[0] in ('Regular', 'high'):
            condition_pairs = [(detected_types[1], detected_types[0])]
        else:
            condition_pairs = [(detected_types[0], detected_types[1])]
    else:
        condition_pairs = [('Random', 'Regular')]  # fallback

    condition1, condition2 = condition_pairs[0]  # 向後相容顯示用

    print("\n" + "=" * 70)
    print("Group ERSP Auto Analysis  v2.0")
    print("=" * 70)
    print(f"Subjects (N={len(subject_ids)}): {subject_ids}")
    print(f"Condition pairs: {condition_pairs}")
    print(f"Data source (stim/pkl): {pkl_dir}")
    print(f"Data source (resp/h5):  {h5_dir}")
    print(f"Output root: {output_dir}")
    print(f"Permutation Test: {'Yes' if do_permutation_test else 'No'}")
    print(f"Total {total} combinations × {len(condition_pairs)} pairs")
    print("=" * 70)

    # ============================================================
    # Section-level colorbar 預掃描
    # ============================================================
    # 設計原則：「出現在同一張投影片上的面板才共用 colorbar」
    #
    # 學習階段：Motor（上）+ 對應 Perceptual（下）配對顯示
    #   Motor          ↔ Perceptual
    #   Motor_Frontal  ↔ Perceptual_Frontal
    #   Motor_Central  ↔ Perceptual_Central
    #   Motor_Parietal ↔ Perceptual_Parietal
    #   Motor_Occipital↔ Perceptual_Occipital
    #
    # 測驗階段：Early（上）+ Late（下）為同一 ROI 的不同時段
    #   → 每個 ROI 各自獨立掃描，不同 ROI 不共用
    # ─────────────────────────────────────────────────────────────
    _ALL_ROIS   = [r.lower() for r in ROI_GROUPS.keys()]
    _ALL_BLOCKS = [f"Block{bs}-{be}" for bs, be in LEARNING_GROUPS]
    _all_conds  = list({c for pair in condition_pairs for c in pair})

    _LEARNING_ROI_PAIRS = [
        ('motor',           'perceptual'),
        ('motor_frontal',   'perceptual_frontal'),
        ('motor_central',   'perceptual_central'),
        ('motor_parietal',  'perceptual_parietal'),
        ('motor_occipital', 'perceptual_occipital'),
    ]

    # key: (phase_or_test_type, lock_type, roi_name) → (vmax_cond, vmax_diff)
    _section_vmaxes   = {}
    _ep4e1_vmaxes     = {}
    # key: (lock_type, test_type, roi_name) → (vmax_cond, vmax_diff)
    _allblocks_vmaxes = {}

    if unified_colorbar:
        print(f"\n{'█'*62}")
        print(f"  正在預掃描 Section-level Colorbar（請稍候...）")
        print(f"{'█'*62}")

        # ── 學習階段：每個配對合掃 ──
        for _lk in ['stimulus', 'response']:
            for (_roi_m, _roi_p) in _LEARNING_ROI_PAIRS:
                _roi_pair = [_roi_m, _roi_p]
                _lbl = (f"Learning | {_lk.capitalize()}-locked | "
                        f"{_roi_m} + {_roi_p}")
                _vc, _vd = _compute_block_section_vmax(
                    subject_ids, pkl_dir, h5_dir,
                    _lk, 'learning', _ALL_BLOCKS,
                    condition_pairs, _roi_pair,
                    test_type=None, label=_lbl
                )
                _section_vmaxes[('learning', _lk, _roi_m)] = (_vc, _vd)
                _section_vmaxes[('learning', _lk, _roi_p)] = (_vc, _vd)

                _lbl_e = (f"Epoch4-Epoch1 | {_lk.capitalize()}-locked | "
                          f"{_roi_m} + {_roi_p}")
                _vc_e, _vd_e = _compute_epoch_diff_section_vmax(
                    subject_ids, pkl_dir, h5_dir,
                    _lk, _all_conds, _roi_pair,
                    label=_lbl_e
                )
                _ep4e1_vmaxes[(_lk, _roi_m)] = (_vc_e, _vd_e)
                _ep4e1_vmaxes[(_lk, _roi_p)] = (_vc_e, _vd_e)

        # ── 測驗階段：每個 ROI 各自獨立掃 ──
        for _lk in ['stimulus', 'response']:
            for _tt in ['motor', 'perceptual']:
                for _roi in _ALL_ROIS:
                    _lbl_pair = (f"Testing {_tt.capitalize()} Early/Late | "
                                 f"{_lk.capitalize()}-locked | {_roi}")
                    _vc_pair, _vd_pair = _compute_testing_pair_vmax(
                        subject_ids, pkl_dir, h5_dir,
                        _lk, _tt, condition_pairs, [_roi],
                        label=_lbl_pair
                    )
                    _section_vmaxes[(_tt, _lk, _roi)] = (_vc_pair, _vd_pair)

                    _lbl_ab = (f"Testing AllBlocks {_tt.capitalize()} | "
                               f"{_lk.capitalize()}-locked | {_roi}")
                    _vc_ab, _vd_ab = _compute_allblocks_testing_vmax(
                        subject_ids, pkl_dir, h5_dir,
                        _lk, _tt, condition_pairs, [_roi],
                        label=_lbl_ab
                    )
                    _allblocks_vmaxes[(_lk, _tt, _roi)] = (_vc_ab, _vd_ab)

        # ── AllBlocks colorbar 統一：Motor Test ↔ Perceptual Test（同 lock type）──
        # 論述「Motor Test 有 Theta ERD 但 Perceptual Test 沒有」，
        # 兩個測驗條件必須在同一個 colorbar 下才有視覺意義。
        print(f"\n  {'─'*60}")
        print(f"  AllBlocks vmax 統一 Step 1：Motor Test ↔ Perceptual Test（per lock type × ROI）")
        _all_rois_ab = set(roi for (lk, tt, roi) in _allblocks_vmaxes)
        for _lk in ['stimulus', 'response']:
            for _roi in _all_rois_ab:
                vc_m, vd_m = _allblocks_vmaxes.get((_lk, 'motor',      _roi), (None, None))
                vc_p, vd_p = _allblocks_vmaxes.get((_lk, 'perceptual', _roi), (None, None))
                if vc_m is None and vc_p is None:
                    continue
                vc_c = max(v for v in [vc_m, vc_p] if v is not None)
                vd_c = max(v for v in [vd_m, vd_p] if v is not None)
                _allblocks_vmaxes[(_lk, 'motor',      _roi)] = (vc_c, vd_c)
                _allblocks_vmaxes[(_lk, 'perceptual', _roi)] = (vc_c, vd_c)

        # ── AllBlocks colorbar 統一 Step 2：Motor ROI ↔ Perceptual ROI 配對（同 lock type × test type）──
        # 論述「Motor Test Response-locked 中，Motor ROI 有 Theta ERD 但 Perceptual ROI 沒有」，
        # 同一測驗條件同一 lock type 下的 Motor ROI 和 Perceptual ROI 需共用 colorbar。
        print(f"  AllBlocks vmax 統一 Step 2：Motor ROI ↔ Perceptual ROI 配對（per lock type × test type）")
        _AB_ROI_PAIRS = [
            ('motor',          'perceptual'),
            ('motor_frontal',  'perceptual_frontal'),
            ('motor_central',  'perceptual_central'),
            ('motor_parietal', 'perceptual_parietal'),
            ('motor_occipital','perceptual_occipital'),
        ]
        for _lk in ['stimulus', 'response']:
            for _tt in ['motor', 'perceptual']:
                for _roi_m, _roi_p in _AB_ROI_PAIRS:
                    vc_m, vd_m = _allblocks_vmaxes.get((_lk, _tt, _roi_m), (None, None))
                    vc_p, vd_p = _allblocks_vmaxes.get((_lk, _tt, _roi_p), (None, None))
                    if vc_m is None and vc_p is None:
                        continue
                    vc_c = max(v for v in [vc_m, vc_p] if v is not None)
                    vd_c = max(v for v in [vd_m, vd_p] if v is not None)
                    _allblocks_vmaxes[(_lk, _tt, _roi_m)] = (vc_c, vd_c)
                    _allblocks_vmaxes[(_lk, _tt, _roi_p)] = (vc_c, vd_c)
                    print(f"    {_lk} | {_tt} | {_roi_m}↔{_roi_p}: ±{vc_c:.4f} dB")
        print(f"  {'─'*60}")

        print(f"\n{'█'*62}")
        print(f"  Section Colorbar 預掃描完成")
        print(f"  （在 Log 中搜尋「SECTION COLORBAR LOG」可驗證各節 vmax）")
        print(f"{'█'*62}\n")

    all_combo_results = {}
    done = 0
    skipped = 0

    for i, combo in enumerate(combos, 1):
        phase     = combo['phase']
        test_type = combo['test_type']
        lock_type = combo['lock_type']
        roi_name  = combo['roi_name']

        tt_str  = f"_{test_type}" if test_type else ""
        key     = f"{phase}{tt_str}_{lock_type}_{roi_name}"
        sub_dir = str(Path(output_dir) / key)

        print(f"\n{'#'*70}")
        print(f"  [{i}/{total}]  {key}")
        print(f"{'#'*70}")

        # 從預掃描結果取得本 section 的 vmax
        if unified_colorbar:
            _sect_key = (test_type if test_type else 'learning', lock_type, roi_name)
            _ext_vc, _ext_vd = _section_vmaxes.get(_sect_key, (None, None))
            _ext_e1_vc, _ext_e1_vd = _ep4e1_vmaxes.get((lock_type, roi_name), (None, None))
            if _ext_vc is not None:
                print(f"  → section colorbar [{roi_name}]: vmax_cond=±{_ext_vc:.4f} dB  vmax_diff=±{_ext_vd:.4f} dB")
        else:
            _ext_vc = _ext_vd = _ext_e1_vc = _ext_e1_vd = None

        for _c1, _c2 in condition_pairs:
            _pair_label = f"{_c1}_vs_{_c2}"
            _sub_dir_pair = str(Path(sub_dir) / _pair_label)
            try:
                result = group_ersp_analysis(
                    subject_ids              = subject_ids,
                    condition1               = _c1,
                    condition2               = _c2,
                    phase                    = phase,
                    lock_type                = lock_type,
                    roi_name                 = roi_name,
                    pkl_dir                  = pkl_dir,
                    h5_dir                   = h5_dir,
                    output_dir               = _sub_dir_pair,
                    do_permutation_test      = do_permutation_test,
                    n_permutations           = n_permutations,
                    test_type                = test_type,
                    unified_colorbar         = unified_colorbar,
                    display_label1           = display_label1,
                    display_label2           = display_label2,
                    external_vmax_cond       = _ext_vc,
                    external_vmax_diff       = _ext_vd,
                    external_vmax_ep4e1_cond = _ext_e1_vc,
                    external_vmax_ep4e1_diff = _ext_e1_vd,
                )
                all_combo_results[f"{key}_{_pair_label}"] = result
                n_blocks = len(result)
                if n_blocks > 0:
                    done += 1
                    print(f"  ✓ [{i}/{total}] {key} | {_pair_label}: {n_blocks} block group(s) done")
                else:
                    skipped += 1
                    print(f"  ⚠  [{i}/{total}] {key} | {_pair_label}: no valid data")
            except Exception as e:
                skipped += 1
                print(f"  ✗ [{i}/{total}] {key} | {_pair_label}: error → {e}")
                import traceback; traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"✓ Auto group analysis complete")
    print(f"  Success: {done}/{total} combinations")
    if skipped:
        print(f"  Skipped/failed: {skipped}/{total}  (data not yet generated or error)")
    print(f"  Images saved to: {output_dir}")
    print(f"{'='*70}")

    # ── Pooled Testing + Motor-Perceptual Diff 群體分析 ──────────────
    print(f"\n{'#'*70}")
    print(f"  Pooled Testing + Motor-Perceptual Diff 群體分析")
    print(f"{'#'*70}")

    # ── Motor-Perceptual Diff vmax 獨立預掃描（統一 Response ↔ Stimulus colorbar）──
    # 因為 MP diff 的論述需要比較 Response-locked 和 Stimulus-locked 的效果幅度，
    # 兩個 lock type 必須共用 colorbar 才有視覺意義。
    # 預掃描在主迴圈前獨立完成，主迴圈直接使用統一後的 vmax。
    print(f"\n  {'█'*62}")
    print(f"  Motor-Perceptual Diff vmax 預掃描（Response ↔ Stimulus 統一）")
    print(f"  {'█'*62}")
    _mp_vmaxes_pre = {}
    for _lk_mp in ['stimulus', 'response']:
        _lk_cap_mp = _lk_mp.capitalize()
        _adir_mp   = Path(pkl_dir) if _lk_mp == 'stimulus' else Path(h5_dir)
        for _roi_mp in [r.lower() for r in ROI_GROUPS.keys()]:
            _mc_vals, _mi_vals = [], []
            for _c1p, _c2p in condition_pairs:
                _mr_p, _mn_p, _pr_p, _pn_p = {}, {}, {}, {}
                for sid in subject_ids:
                    for _cn_p, _sr_p, _sn_p in [
                        ('MotorTest',     _mr_p, _mn_p),
                        ('PerceptualTest', _pr_p, _pn_p),
                    ]:
                        for _cd_p, _st_p in [(_c1p, _sr_p), (_c2p, _sn_p)]:
                            fp_p = _adir_mp / f'{sid}_{_lk_cap_mp}_{_cn_p}_AllBlocks_{_cd_p}_ERSP.h5'
                            if fp_p.exists():
                                try:
                                    _ap, _, _, _ = _load_h5_response(fp_p, _roi_mp)
                                    _st_p[sid] = _ap
                                except Exception:
                                    pass
                _cm_p = [s for s in subject_ids
                         if s in _mr_p and s in _mn_p and s in _pr_p and s in _pn_p]
                if not _cm_p:
                    continue
                fp_ref_p = _adir_mp / f'{_cm_p[0]}_{_lk_cap_mp}_MotorTest_AllBlocks_{_c1p}_ERSP.h5'
                if not fp_ref_p.exists():
                    continue
                try:
                    _, _, _tm_p, _ = _load_h5_response(fp_ref_p, _roi_mp)
                except Exception:
                    continue
                _t_mask_p = (_tm_p >= -0.5) & (_tm_p <= 0.5)
                _rd_p = np.array([_mr_p[s] - _pr_p[s] for s in _cm_p]).mean(axis=0)
                _nd_p = np.array([_mn_p[s] - _pn_p[s] for s in _cm_p]).mean(axis=0)
                _mc_vals.append(np.abs(_rd_p[:, _t_mask_p]).ravel())
                _mc_vals.append(np.abs(_nd_p[:, _t_mask_p]).ravel())
                _mi_vals.append(np.abs((_rd_p - _nd_p)[:, _t_mask_p]).ravel())
            _vc_mp = float(np.percentile(np.concatenate(_mc_vals), 95)) if _mc_vals else None
            _vi_mp = float(np.percentile(np.concatenate(_mi_vals), 95)) if _mi_vals else None
            _mp_vmaxes_pre[(_roi_mp, _lk_mp)] = (_vc_mp, _vi_mp)

    # Harmonize Step 1：Response ↔ Stimulus（同 ROI）
    print(f"  {'─'*60}")
    print(f"  MP Diff vmax harmonization Step 1（Response ↔ Stimulus per ROI）:")
    for _roi_h in [r.lower() for r in ROI_GROUPS.keys()]:
        _vc_r, _vi_r = _mp_vmaxes_pre.get((_roi_h, 'response'), (None, None))
        _vc_s, _vi_s = _mp_vmaxes_pre.get((_roi_h, 'stimulus'), (None, None))
        if _vc_r is None and _vc_s is None:
            continue
        _vc_c = max(v for v in [_vc_r, _vc_s] if v is not None)
        _vi_c = max(v for v in [_vi_r, _vi_s] if v is not None)
        _mp_vmaxes_pre[(_roi_h, 'response')] = (_vc_c, _vi_c)
        _mp_vmaxes_pre[(_roi_h, 'stimulus')] = (_vc_c, _vi_c)

    # Harmonize Step 2：Motor ROI ↔ Perceptual ROI 配對（同 lock type）
    # 論述「MP diff Response-locked 中 Motor ROI 和 Perceptual ROI 的效果方向」，
    # 兩個 ROI 需共用 colorbar 才能比較幅度。
    print(f"  MP Diff vmax harmonization Step 2（Motor ROI ↔ Perceptual ROI per lock type）:")
    _MP_ROI_PAIRS = [
        ('motor',          'perceptual'),
        ('motor_frontal',  'perceptual_frontal'),
        ('motor_central',  'perceptual_central'),
        ('motor_parietal', 'perceptual_parietal'),
        ('motor_occipital','perceptual_occipital'),
    ]
    for _lk_mp_h in ['stimulus', 'response']:
        for _roi_m_h, _roi_p_h in _MP_ROI_PAIRS:
            _vc_m_h, _vi_m_h = _mp_vmaxes_pre.get((_roi_m_h, _lk_mp_h), (None, None))
            _vc_p_h, _vi_p_h = _mp_vmaxes_pre.get((_roi_p_h, _lk_mp_h), (None, None))
            if _vc_m_h is None and _vc_p_h is None:
                continue
            _vc_c2 = max(v for v in [_vc_m_h, _vc_p_h] if v is not None)
            _vi_c2 = max(v for v in [_vi_m_h, _vi_p_h] if v is not None)
            _mp_vmaxes_pre[(_roi_m_h, _lk_mp_h)] = (_vc_c2, _vi_c2)
            _mp_vmaxes_pre[(_roi_p_h, _lk_mp_h)] = (_vc_c2, _vi_c2)
            print(f"    {_lk_mp_h} | {_roi_m_h}↔{_roi_p_h}: ±{_vc_c2:.4f} / inter=±{_vi_c2:.4f} dB")
    print(f"  {'─'*60}\n")

    for lock_type in ['stimulus', 'response']:
        for roi_name in [r.lower() for r in ROI_GROUPS.keys()]:
            roi_lower = roi_name
            roi_cap   = roi_name.capitalize()

            # ── Pooled Motor / Pooled Perceptual ──
            _DISP_AUTO = {
                'regular_high': 'Regular High',
                'random_high':  'Random High',
                'random_low':   'Random Low',
                'Regular':      'Regular',
                'Random':       'Random',
            }
            for condition1, condition2 in condition_pairs:
                display_label1 = _DISP_AUTO.get(condition1, condition1)
                display_label2 = _DISP_AUTO.get(condition2, condition2)
                for test_type in ['motor', 'perceptual']:
                    cond_name = 'MotorTest' if test_type == 'motor' else 'PerceptualTest'
                    lock_cap  = lock_type.capitalize()

                    # stimulus-locked AllBlocks 存在 pkl_dir（stim h5 目錄），
                    # response-locked AllBlocks 存在 h5_dir（triplet h5 目錄）
                    _allblocks_dir = Path(pkl_dir) if lock_type == 'stimulus' else Path(h5_dir)

                    arr_left_list, arr_right_list, ids_found = [], [], []
                    nave_left_list, nave_right_list = [], []
                    freqs_p = times_p = None

                    for sid in subject_ids:
                        fp_l = _allblocks_dir / f'{sid}_{lock_cap}_{cond_name}_AllBlocks_{condition1}_ERSP.h5'
                        fp_r = _allblocks_dir / f'{sid}_{lock_cap}_{cond_name}_AllBlocks_{condition2}_ERSP.h5'
                        if not fp_l.exists() or not fp_r.exists():
                            continue
                        try:
                            el, f, t, _nave_l = _load_h5_response(fp_l, roi_name)
                            er, _, _, _nave_r = _load_h5_response(fp_r, roi_name)
                            arr_left_list.append(el); arr_right_list.append(er)
                            nave_left_list.append(_nave_l); nave_right_list.append(_nave_r)
                            if freqs_p is None: freqs_p, times_p = f, t
                            ids_found.append(sid)
                            print(f"    ✓ {sid} {test_type} AllBlocks  n_trials_left={_nave_l}  n_trials_right={_nave_r}")
                        except Exception as _e:
                            print(f"    ✗ {sid} {test_type} AllBlocks: {_e}")

                    if len(arr_left_list) == 0:
                        continue

                    arr_l = np.array(arr_left_list)
                    arr_r = np.array(arr_right_list)
                    common = ids_found
                    sub_dir_p = str(Path(output_dir) / f'testing_pooled_{test_type}_{lock_type}_{roi_lower}')
                    Path(sub_dir_p).mkdir(parents=True, exist_ok=True)
                    suptitle_p = (
                        f'Testing Pooled | {test_type.capitalize()} | AllBlocks | '
                        f'{lock_cap}-locked | {roi_cap} ROI'
                    )
                    out_name_p = f'group_pooled_{test_type}_{lock_type}_{roi_lower}_{condition1}_vs_{condition2}_comparison.png'
                    # AllBlocks 使用預掃描的 section vmax
                    _ab_vc, _ab_vd = _allblocks_vmaxes.get((lock_type, test_type, roi_lower), (None, None)) if unified_colorbar else (None, None)
                    if _ab_vc is not None:
                        print(f"  [colorbar] AllBlocks 使用 section colorbar: ±{_ab_vc:.4f} / ±{_ab_vd:.4f} dB")
                    _ab_result = _plot_group_block(
                        arr_l, arr_r, freqs_p, times_p,
                        common, suptitle_p, Path(sub_dir_p) / out_name_p,
                        do_permutation_test, n_permutations, lock_type=lock_type,
                        vmax_cond=_ab_vc,
                        vmax_diff=_ab_vd,
                        label_left=display_label1 or condition1,
                        label_right=display_label2 or condition2,
                        nave_list_left=nave_left_list,
                        nave_list_right=nave_right_list,
                    )
                    if do_permutation_test:
                        _log_perm(
                            label='ROI-AllBlocks', roi=roi_name, lock=lock_type,
                            phase=f'testing_{test_type}', pair='allblocks',
                            cond_pair=f'{condition1}_vs_{condition2}',
                            n_sub=len(common),
                            n_sig=_ab_result.get('n_sig_clusters', 0),
                            n_total=_ab_result.get('n_clusters', 0),
                        )

                # ── Motor-Perceptual Diff（三對）──
                lock_cap = lock_type.capitalize()
                _allblocks_dir_mp = Path(pkl_dir) if lock_type == 'stimulus' else Path(h5_dir)

                # MP vmax 從預掃描結果取得（已在主迴圈前完成 harmonization）
                _mp_vmax_cond, _mp_vmax_inter = _mp_vmaxes_pre.get((roi_name, lock_type), (None, None))
                if _mp_vmax_cond is not None:
                    print(f"\n  {'═'*58}")
                    print(f"  SECTION COLORBAR LOG  >>>  Motor-Percept Diff | {lock_type} | {roi_lower}")
                    print(f"  vmax_cond=±{_mp_vmax_cond:.4f} dB  vmax_inter=±{_mp_vmax_inter:.4f} dB  [harmonized]")
                    print(f"  {'═'*58}")

                for _c1, _c2 in condition_pairs:
                    motor_reg, motor_ran = {}, {}
                    percept_reg, percept_ran = {}, {}

                    for sid in subject_ids:
                        for cond, store_reg, store_ran in [
                            ('MotorTest', motor_reg, motor_ran),
                            ('PerceptualTest', percept_reg, percept_ran),
                        ]:
                            fp_l = _allblocks_dir_mp / f'{sid}_{lock_cap}_{cond}_AllBlocks_{_c1}_ERSP.h5'
                            fp_r = _allblocks_dir_mp / f'{sid}_{lock_cap}_{cond}_AllBlocks_{_c2}_ERSP.h5'
                            if fp_l.exists():
                                try:
                                    store_reg[sid], _, _, _ = _load_h5_response(fp_l, roi_name)
                                except Exception:
                                    pass
                            if fp_r.exists():
                                try:
                                    store_ran[sid], _, _, _ = _load_h5_response(fp_r, roi_name)
                                except Exception:
                                    pass

                    common_mp = [s for s in subject_ids
                                 if s in motor_reg and s in motor_ran
                                 and s in percept_reg and s in percept_ran]
                    if not common_mp:
                        continue

                    freqs_mp = times_mp = None
                    fp_ref = _allblocks_dir_mp / f'{common_mp[0]}_{lock_cap}_MotorTest_AllBlocks_{_c1}_ERSP.h5'
                    if fp_ref.exists():
                        try:
                            _, freqs_mp, times_mp, _ = _load_h5_response(fp_ref, roi_name)
                        except Exception:
                            pass
                    if freqs_mp is None:
                        continue

                    arr_reg_diff = np.array([motor_reg[s] - percept_reg[s] for s in common_mp])
                    arr_ran_diff = np.array([motor_ran[s] - percept_ran[s] for s in common_mp])

                    _DISP_MP = {
                        'regular_high': 'Regular High',
                        'random_high':  'Random High',
                        'random_low':   'Random Low',
                        'Regular':      'Regular',
                        'Random':       'Random',
                    }
                    _lbl_c1   = _DISP_MP.get(_c1, _c1)
                    _lbl_c2   = _DISP_MP.get(_c2, _c2)
                    _pair_lbl = f'{_c1}_vs_{_c2}'
                    sub_dir_mp = str(Path(output_dir) / f'testing_motor_perceptual_diff_{lock_type}_{roi_lower}')
                    Path(sub_dir_mp).mkdir(parents=True, exist_ok=True)
                    suptitle_mp = f'Motor-Perceptual Diff | {_lbl_c1} vs {_lbl_c2} | {lock_cap}-locked | {roi_cap} ROI'
                    out_name_mp = f'group_motor_perceptual_diff_{lock_type}_{roi_lower}_{_pair_lbl}.png'
                    _mp_diff_result = _plot_group_motor_perceptual_diff(
                        arr_reg_diff, arr_ran_diff, freqs_mp, times_mp,
                        common_mp, suptitle_mp, Path(sub_dir_mp) / out_name_mp,
                        label_left=_lbl_c1, label_right=_lbl_c2,
                        vmax_cond=_mp_vmax_cond, vmax_inter=_mp_vmax_inter,
                        do_permutation=do_permutation_test,
                        n_permutations=n_permutations,
                    )
                    if do_permutation_test and _c1 == condition1 and _c2 == condition2:
                        _log_perm(
                            label='MP-Diff-RegLeft', roi=roi_name, lock=lock_type,
                            phase='testing_motor_vs_perceptual', pair='allblocks',
                            cond_pair=f'{_c1}_vs_{_c2}',
                            n_sub=len(common_mp),
                            n_sig=_mp_diff_result.get('n_sig_reg', 0),
                            n_total=_mp_diff_result.get('n_tot_reg', 0),
                        )
                        _log_perm(
                            label='MP-Diff-RanRight', roi=roi_name, lock=lock_type,
                            phase='testing_motor_vs_perceptual', pair='allblocks',
                            cond_pair=f'{_c1}_vs_{_c2}',
                            n_sub=len(common_mp),
                            n_sig=_mp_diff_result.get('n_sig_ran', 0),
                            n_total=_mp_diff_result.get('n_tot_ran', 0),
                        )
                        _log_perm(
                            label='MP-Diff-Interaction', roi=roi_name, lock=lock_type,
                            phase='testing_motor_vs_perceptual', pair='allblocks',
                            cond_pair=f'{_c1}_vs_{_c2}',
                            n_sub=len(common_mp),
                            n_sig=_mp_diff_result.get('n_sig_inter', 0),
                            n_total=_mp_diff_result.get('n_tot_inter', 0),
                        )

        if any(c['lock_type'] == 'stimulus' for c in combos):
            print(f"\n⚠  Stimulus-locked data source:")
            print(f"   Run option 13 first, select 'Save for group analysis: y'")
            print(f"   pkl will be saved automatically to data_dir")

        print("\n" + "="*60)
        print("輸出 ERSP 摘要 CSV 供 R 分析用...")
        print("="*60)

    def export_ersp_to_csv(data_dir, subject_ids, output_csv_dir=r'C:\Experiment\ersp_csv'):
        """
        從 h5 檔案讀取 ERSP 結果，
        對指定頻率和時間窗口取平均，輸出成 CSV 供 R 分析用。

        Response-locked：Motor ROI，Theta（4-8Hz），-300 to +50ms
        Stimulus-locked：Perceptual ROI，Alpha（8-13Hz），+100 to +300ms
        """
        import pandas as pd
        import glob
        import warnings

        WINDOWS = {
            'response': {
                'rois':      ['Motor', 'Motor_Frontal', 'Motor_Central',
                              'Motor_Parietal', 'Motor_Occipital',
                              'Perceptual', 'Perceptual_Parietal', 'Perceptual_Occipital',
                              'Perceptual_Frontal', 'Perceptual_Central'],
                'freq_band': 'theta',
                'freq_range': (4, 8),
                'time_range': (-0.300, 0.050),
                'out_dir':   os.path.join(output_csv_dir, 'response_lock'),
            },
            'stimulus': {
                'rois':      ['Motor', 'Motor_Frontal', 'Motor_Central',
                              'Motor_Parietal', 'Motor_Occipital',
                              'Perceptual', 'Perceptual_Parietal', 'Perceptual_Occipital',
                              'Perceptual_Frontal', 'Perceptual_Central'],
                'freq_band': 'alpha',
                'freq_range': (8, 13),
                'time_range': (0.100, 0.300),
                'out_dir':   os.path.join(output_csv_dir, 'stimulus_lock'),
            },
        }

        rows = []

        for lock_type, cfg in WINDOWS.items():
            os.makedirs(cfg['out_dir'], exist_ok=True)

            # 掃描 h5 檔案
            if lock_type == 'response':
                h5_files = glob.glob(os.path.join(data_dir, '*_Response_*.h5'))
            else:
                h5_files = glob.glob(os.path.join(data_dir, '*_Stimulus_*.h5'))

            for fpath in h5_files:
                fname = os.path.basename(fpath)
                parts = fname.replace('_ERSP.h5', '').split('_')

                # 解析檔名：sub0001_Response_Learning_Block7-11_Regular_ERSP.h5
                try:
                    sid         = parts[0]
                    phase       = parts[2]   # Learning / MotorTest / PerceptualTest
                    block_group = parts[3]   # Block7-11
                    trial_type  = parts[4]   # Regular / Random / high / low
                except IndexError:
                    print(f"  ⚠ 無法解析檔名：{fname}")
                    continue

                if sid not in subject_ids:
                    continue

                # 判斷 tasktype
                if 'Motor' in phase and 'Test' in phase:
                    tasktype = 'motor'
                elif 'Perceptual' in phase and 'Test' in phase:
                    tasktype = 'percept'
                else:
                    tasktype = 'none'

                # 讀取 h5
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        tfr_list = mne.time_frequency.read_tfrs(str(fpath))
                    tfr = tfr_list[0] if isinstance(tfr_list, list) else tfr_list
                except Exception as e:
                    print(f"  ⚠ 讀取失敗：{fname}：{e}")
                    continue

                freqs = tfr.freqs
                times = tfr.times

                # 頻率和時間 mask
                freq_mask = (freqs >= cfg['freq_range'][0]) & (freqs <= cfg['freq_range'][1])
                time_mask = (times >= cfg['time_range'][0]) & (times <= cfg['time_range'][1])

                # 對每個 ROI 取平均
                for roi_name in cfg['rois']:
                    roi_channels = ROI_GROUPS.get(roi_name)
                    if roi_channels is None:
                        continue

                    ch_names_upper = [ch.upper() for ch in tfr.ch_names]
                    roi_idx = [ch_names_upper.index(ch.upper())
                               for ch in roi_channels
                               if ch.upper() in ch_names_upper]

                    if not roi_idx:
                        continue

                    # data shape: (n_ch, n_freqs, n_times)
                    roi_data = tfr.data[roi_idx]           # (n_roi_ch, n_freqs, n_times)
                    roi_mean = roi_data.mean(axis=0)       # (n_freqs, n_times)
                    ersp_mean = roi_mean[freq_mask][:, time_mask].mean()

                    rows.append({
                        'sid':         sid,
                        'lock_type':   lock_type,
                        'phase':       phase,
                        'block_group': block_group,
                        'trial_type':  trial_type,
                        'tasktype':    tasktype,
                        'roi':         roi_name,
                        'freq_band':   cfg['freq_band'],
                        'freq_min':    cfg['freq_range'][0],
                        'freq_max':    cfg['freq_range'][1],
                        'time_min_ms': int(cfg['time_range'][0] * 1000),
                        'time_max_ms': int(cfg['time_range'][1] * 1000),
                        'ersp_mean':   float(ersp_mean),
                    })

            # 輸出 CSV
            if rows:
                df = pd.DataFrame(rows)
                df_lock = df[df['lock_type'] == lock_type]
                if not df_lock.empty:
                    out_path = os.path.join(cfg['out_dir'], 'ersp_summary.csv')
                    df_lock.to_csv(out_path, index=False)
                    print(f"  ✓ {lock_type} CSV 已儲存：{out_path}（{len(df_lock)} 行）")

    export_ersp_to_csv(
        data_dir=h5_dir,
        subject_ids=subject_ids,
        output_csv_dir=r'C:\Experiment\ersp_csv',
    )

    # ════════════════════════════════════════════════════════════
    # 全域 Permutation Test 摘要
    # ════════════════════════════════════════════════════════════
    print_perm_summary()

    # ════════════════════════════════════════════════════════════
    # G*Power 樣本數估算
    # ════════════════════════════════════════════════════════════
    print("\n" + "█"*70)
    print("  G*Power 樣本數估算（基於 10 人 pilot data）")
    print("  比較：regular_high vs random_low")
    print("  Theta (4–8 Hz, −300 to +50 ms)  ─  Response-locked, Motor ROI")
    print("  Alpha (8–13 Hz, +100 to +300 ms) ─  Stimulus-locked, Perceptual ROI")
    print("█"*70)
    compute_power_analysis(
        h5_dir=h5_dir,
        pkl_dir=pkl_dir,
        subject_ids=subject_ids,
        condition_left='regular_high',
        condition_right='random_low',
        alpha=0.05,
        power_target=0.80,
    )

    return all_combo_results


    # ============================================================
    # 7. 輸出 ERSP 摘要 CSV 供 R 分析用
    # ============================================================


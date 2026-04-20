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


# ============================================================
# 1. 儲存個別受試者 ERSP（由 ersp.py 的 save_for_group 呼叫）
# ============================================================

def save_subject_ersp(ersp_data, subject_id, condition, phase, lock_type,
                      freqs, times, roi_name,
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
    """讀取 stimulus lock 的 .pkl，回傳 (ersp_2d, freqs, times)。"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['ersp'], data['freqs'], data['times']


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

    return power, freqs, times


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
            raise FileNotFoundError(f"File not found: {filepath}")
        return _load_pkl(filepath)

    else:  # response lock
        filename = (
            f"{subject_id}_Response_Learning_{group_label}_{trial_type}_ERSP.h5"
        )
        filepath = data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return _load_h5_response(filepath, roi_name)


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
    block_names = []
    freqs = times = None

    for fp in files_to_use:
        try:
            if lock_type == 'stimulus':
                ersp, f, t = _load_pkl(fp)
            else:
                ersp, f, t = _load_h5_response(fp, roi_name)
            ersp_list.append(ersp)
            block_names.append(fp.name)
            if freqs is None:
                freqs, times = f, t
        except Exception as e:
            print(f"      ⚠ Skipping {fp.name}: {e}")

    if not ersp_list:
        raise FileNotFoundError(f"pair={pair}: all blocks failed to load for {subject_id}")

    return np.mean(ersp_list, axis=0), freqs, times, block_names


# ============================================================
# 3. 批次載入多位受試者
# ============================================================

def _load_group_data(subject_ids, pkl_dir, lock_type, phase,
                     test_type, group_label, trial_type, roi_name,
                     h5_dir=None):
    """
    載入所有受試者在某個條件（某個 Block 組）的 ERSP。

    Returns
    -------
    ersp_array : np.ndarray  (n_loaded, n_freqs, n_times) or None
    freqs, times : np.ndarray or None
    loaded_ids, missing_ids : list
    """
    ersp_list  = []
    freqs = times = None
    loaded_ids = []
    missing_ids = []

    for sid in subject_ids:
        try:
            ersp, f, t = _find_and_load(
                pkl_dir, sid, lock_type, phase,
                test_type, group_label, trial_type, roi_name,
                h5_dir=h5_dir
            )
            ersp_list.append(ersp)
            loaded_ids.append(sid)
            if freqs is None:
                freqs, times = f, t
            print(f"    ✓ {sid}  shape={ersp.shape}")
        except FileNotFoundError as e:
            print(f"    ✗ {sid}: {e}")
            missing_ids.append(sid)
        except Exception as e:
            print(f"    ✗ {sid}: load failed ({e})")
            missing_ids.append(sid)

    if not ersp_list:
        return None, None, None, [], missing_ids

    return np.array(ersp_list), freqs, times, loaded_ids, missing_ids


# ============================================================
# 4. 繪圖
# ============================================================

def _draw_ersp_panel(ax, ersp_2d, freqs, times, title, vmin, vmax, x_min=-0.5, x_max=0.2):
    levels = np.linspace(vmin, vmax, 20)   # ← 改這行
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
                      label_left='Regular', label_right='Random'):
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
    diff     = reg_mean - ran_mean   # Regular − Random

    # ── xlim 依 lock_type ──
    if lock_type == 'stimulus':
        x_min, x_max = -0.5, 0.3
    else:
        x_min, x_max = -0.5, 0.2

    t_mask = (times >= x_min) & (times <= x_max)

    # ── Cluster Permutation Test ──
    sig_mask = None
    n_sig    = 0

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
            sig_mask = np.zeros_like(diff, dtype=bool)
            for c, pv in zip(clusters, cluster_pv):
                if pv < 0.05:
                    sig_mask |= c
                    n_sig += 1
            print(f"    ✓ Significant clusters: {n_sig}/{len(clusters)}")
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
                           f'{label_left}\n(N={n_sub})', vmin_cond, vmax_cond,
                           x_min=x_min, x_max=x_max)
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')

    im2 = _draw_ersp_panel(axes[1], ran_mean, freqs, times,
                           f'{label_right}\n(N={n_sub})', vmin_cond, vmax_cond,
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
                        display_label2=None):
    """
    群體 ERSP 分析。

    Learning：按 Block 分組（Block7-11, 12-16, 17-21, 22-26），每組一張圖。

    Testing：因反平衡設計，不同受試者的 Motor/Perceptual block 位置不同，
             改為每位受試者先把自己所有 Motor（或 Perceptual）block 平均，
             再跨受試者做群體分析，共產一張圖。

    Returns
    -------
    dict
    """
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

        # 統一 colorbar 掃描
        global_vmax_cond = None
        global_vmax_diff = None
        if unified_colorbar:
            if lock_type == 'stimulus':
                _xmin, _xmax = -0.5, 0.3
            else:
                _xmin, _xmax = -0.5, 0.2
            _gvc, _gvd = 0.0, 0.0
            for (_bs, _be) in LEARNING_GROUPS:
                _gl = f"Block{_bs}-{_be}"
                _a1, _, _t, _i1, _ = _load_group_data(subject_ids, pkl_dir, lock_type, phase, None, _gl, condition1, roi_lower, h5_dir=h5_dir)
                _a2, _, _, _i2, _ = _load_group_data(subject_ids, pkl_dir, lock_type, phase, None, _gl, condition2, roi_lower, h5_dir=h5_dir)
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
            print(f"  統一 colorbar: vmax_cond={global_vmax_cond:.4f} dB, vmax_diff={global_vmax_diff:.4f} dB")

        for (blk_start, blk_end) in LEARNING_GROUPS:
            group_label = f"Block{blk_start}-{blk_end}"

            print(f"\n{'─'*60}")
            print(f"  {group_label}")
            print(f"{'─'*60}")

            print(f"\n  Loading {condition1}...")
            arr1, freqs, times, ids1, _ = _load_group_data(
                subject_ids, pkl_dir, lock_type, phase,
                None, group_label, condition1, roi_lower,
                h5_dir=h5_dir
            )

            print(f"\n  Loading {condition2}...")
            arr2, _, _, ids2, _ = _load_group_data(
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

            print(f"\n  Common subjects: {len(common_ids)}")

            suptitle = (
                f"Learning | {group_label} | "
                f"{lock_type.capitalize()}-locked | {roi_cap} ROI"
            )
            out_name  = (
                f"group_learning_{lock_type}_{roi_lower}_{group_label}_comparison.png"
            )
            block_result = _plot_group_block(
                arr2, arr1, freqs, times,
                common_ids, suptitle, output_path / out_name,
                do_permutation_test, n_permutations, lock_type=lock_type,
                vmax_cond=global_vmax_cond,
                vmax_diff=global_vmax_diff,
                label_left=display_label2 or condition2,
                label_right=display_label1 or condition1,
            )
            all_results[group_label] = {
                **block_result,
                'subject_ids': common_ids,
                'freqs': freqs, 'times': times,
            }

    # ============================================================
    # Testing：反平衡設計，按「第一對 block / 第二對 block」分兩張圖
    # ============================================================
    else:
        if not test_type:
            raise ValueError("Testing phase requires test_type ('motor' or 'perceptual')")

        tt_cap = test_type.capitalize()
        pair_labels = {
            'first' : f'(Early {tt_cap} blocks)',
            'second': f'(Late {tt_cap} blocks)',
        }

        # 統一 colorbar 掃描
        global_vmax_cond = None
        global_vmax_diff = None
        if unified_colorbar:
            if lock_type == 'stimulus':
                _xmin, _xmax = -0.5, 0.3
            else:
                _xmin, _xmax = -0.5, 0.2
            _gvc, _gvd = 0.0, 0.0
            for _pair in ('first', 'second'):
                _el1, _el2, _times = [], [], None
                for sid in subject_ids:
                    try:
                        _e1, _f, _t, _ = _load_subject_testing_pair(
                            pkl_dir, sid, lock_type, test_type, condition1, roi_lower,
                            pair=_pair, h5_dir=h5_dir)
                        _el1.append(_e1)
                        if _times is None:
                            _times = _t
                    except Exception:
                        pass
                    try:
                        _e2, _f, _t, _ = _load_subject_testing_pair(
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
            print(f"  統一 colorbar: vmax_cond={global_vmax_cond:.4f} dB, vmax_diff={global_vmax_diff:.4f} dB")

        for pair_key, pair_desc in pair_labels.items():

            print(f"\n{'─'*60}")
            print(f"  Testing / {tt_cap} / {pair_desc}")
            print(f"{'─'*60}")

            def _load_pair(trial_type, _pair=pair_key):
                ersp_list  = []
                freqs = times = None
                loaded_ids = []
                sub_block_info = {}
                for sid in subject_ids:
                    try:
                        ersp, f, t, blk_names = _load_subject_testing_pair(
                            pkl_dir, sid, lock_type,
                            test_type, trial_type, roi_lower,
                            pair=_pair, h5_dir=h5_dir
                        )
                        ersp_list.append(ersp)
                        loaded_ids.append(sid)
                        sub_block_info[sid] = blk_names
                        if freqs is None:
                            freqs, times = f, t
                        print(f"    ✓ {sid}  blocks={[b for b in blk_names]}  shape={ersp.shape}")
                    except FileNotFoundError as e:
                        print(f"    ✗ {sid}: {e}")
                    except Exception as e:
                        print(f"    ✗ {sid}: load failed ({e})")
                if not ersp_list:
                    return None, None, None, [], {}
                return np.array(ersp_list), freqs, times, loaded_ids, sub_block_info

            print(f"\n  Loading {condition1}...")
            arr1, freqs, times, ids1, info1 = _load_pair(condition1)

            print(f"\n  Loading {condition2}...")
            arr2, _, _, ids2, info2 = _load_pair(condition2)

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

            print(f"\n  Common subjects: {len(common_ids)}")

            suptitle = (
                f"Testing | {tt_cap} | {pair_desc} | "
                f"{lock_type.capitalize()}-locked | {roi_cap} ROI"
            )
            out_name = (
                f"group_testing_{lock_type}_{roi_lower}_{test_type}_{pair_key}_comparison.png"
            )
            block_result = _plot_group_block(
                arr2, arr1, freqs, times,
                common_ids, suptitle, output_path / out_name,
                do_permutation_test, n_permutations, lock_type=lock_type,
                vmax_cond=global_vmax_cond,
                vmax_diff=global_vmax_diff,
                label_left=display_label2 or condition2,
                label_right=display_label1 or condition1,
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
        tt = parts[-1]
        if tt not in detected_types:
            detected_types.append(tt)
    if len(detected_types) >= 2:
        if detected_types[0] in ('Regular', 'high'):
            condition1, condition2 = detected_types[1], detected_types[0]
        else:
            condition1, condition2 = detected_types[0], detected_types[1]
    else:
        condition1, condition2 = 'Random', 'Regular'  # fallback

    print("\n" + "=" * 70)
    print("Group ERSP Auto Analysis  v2.0")
    print("=" * 70)
    print(f"Subjects (N={len(subject_ids)}): {subject_ids}")
    print(f"Conditions detected: {condition1} vs {condition2}")
    print(f"Data source (stim/pkl): {pkl_dir}")
    print(f"Data source (resp/h5):  {h5_dir}")
    print(f"Output root: {output_dir}")
    print(f"Permutation Test: {'Yes' if do_permutation_test else 'No'}")
    print(f"Total {total} combinations")
    print("=" * 70)

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

        try:
            result = group_ersp_analysis(
                subject_ids         = subject_ids,
                condition1          = condition1,
                condition2          = condition2,
                phase               = phase,
                lock_type           = lock_type,
                roi_name            = roi_name,
                pkl_dir             = pkl_dir,
                h5_dir              = h5_dir,
                output_dir          = sub_dir,
                do_permutation_test = do_permutation_test,
                n_permutations      = n_permutations,
                test_type           = test_type,
                unified_colorbar    = unified_colorbar,
                display_label1      = display_label1,
                display_label2      = display_label2,
            )
            all_combo_results[key] = result
            n_blocks = len(result)
            if n_blocks > 0:
                done += 1
                print(f"  ✓ [{i}/{total}] {key}: {n_blocks} block group(s) done")
            else:
                skipped += 1
                print(f"  ⚠  [{i}/{total}] {key}: no valid data (pkl/h5 not yet generated?)")
        except Exception as e:
            skipped += 1
            print(f"  ✗ [{i}/{total}] {key}: error → {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"✓ Auto group analysis complete")
    print(f"  Success: {done}/{total} combinations")
    if skipped:
        print(f"  Skipped/failed: {skipped}/{total}  (data not yet generated or error)")
    print(f"  Images saved to: {output_dir}")
    print(f"{'='*70}")

    if any(c['lock_type'] == 'stimulus' for c in combos):
        print(f"\n⚠  Stimulus-locked data source:")
        print(f"   Run option 13 first, select 'Save for group analysis: y'")
        print(f"   pkl will be saved automatically to data_dir")

    print("\n" + "="*60)
    print("輸出 ERSP 摘要 CSV 供 R 分析用...")
    print("="*60)
    export_ersp_to_csv(
        data_dir=h5_dir,
        subject_ids=subject_ids,
        output_csv_dir=r'C:\Experiment\ersp_csv',
    )

    return all_combo_results


# ============================================================
# 7. 輸出 ERSP 摘要 CSV 供 R 分析用
# ============================================================

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

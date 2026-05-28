import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path

def plot_ersp_lum2023_style(power_dict, subject_id, output_dir):
    """
    繪製 ERSP 圖（Lum et al. 2023 風格）
    
    特色：
    - Theta 和 Alpha ROI 並排顯示
    - dB 轉換後的功率
    - 時頻 heatmap with contours
    - 事件時間點標記（time=0）
    """
    
    n_rois = len(power_dict)
    
    fig, axes = plt.subplots(1, n_rois, figsize=(6*n_rois, 5))
    if n_rois == 1:
        axes = [axes]
    
    for idx, (roi_name, data) in enumerate(power_dict.items()):
        ax = axes[idx]
        
        power = data['power']
        times = data['times']
        freqs = data['freqs']
        
        # === Heatmap with contours ===
        im = ax.contourf(
            times * 1000,  # 轉成毫秒
            freqs,
            power,
            levels=20,
            cmap='RdBu_r',
            extend='both'
        )
        
        # 標記 time=0（刺激或反應時間點）
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Event onset')
        
        # 設定座標軸
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title(f'{roi_name} ROI', fontsize=14, fontweight='bold')
        ax.set_ylim([freqs[0], 30])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=11)
    
    plt.suptitle(f'ERSP - {subject_id} (Dillian\'s Parameters)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # === 儲存圖片 ===
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{subject_id}_ersp_lum2023_style.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ ERSP 圖片已儲存: {output_path}")
    plt.close(fig)

def plot_ersp_comparison(power_by_condition, subject_id, condition_labels, output_dir):
    """
    繪製條件比較圖（例如 Regular vs Random）
    
    Layout:
    Row 1: Condition 1 (Theta, Alpha)
    Row 2: Condition 2 (Theta, Alpha)
    Row 3: Difference (Theta, Alpha)
    """
    from matplotlib.gridspec import GridSpec
    
    conditions = list(power_by_condition.keys())
    roi_names = list(power_by_condition[conditions[0]].keys())
    n_rois = len(roi_names)
    
    fig = plt.figure(figsize=(6*n_rois, 12))
    gs = GridSpec(3, n_rois, figure=fig, hspace=0.3, wspace=0.3)
    
    for col_idx, roi_name in enumerate(roi_names):
        
        # === Row 1: Condition 1 ===
        ax1 = fig.add_subplot(gs[0, col_idx])
        data1 = power_by_condition[conditions[0]][roi_name]
        im1 = ax1.contourf(data1['times'] * 1000, data1['freqs'], data1['power'],
                          levels=20, cmap='RdBu_r', extend='both')
        ax1.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax1.set_title(f'{condition_labels[0]}\n{roi_name} ROI', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency (Hz)', fontsize=11)
        ax1.set_ylim([data1['freqs'][0], 30])
        plt.colorbar(im1, ax=ax1, label='Power (dB)')
        
        # === Row 2: Condition 2 ===
        ax2 = fig.add_subplot(gs[1, col_idx])
        data2 = power_by_condition[conditions[1]][roi_name]
        im2 = ax2.contourf(data2['times'] * 1000, data2['freqs'], data2['power'],
                          levels=20, cmap='RdBu_r', extend='both')
        ax2.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax2.set_title(f'{condition_labels[1]}\n{roi_name} ROI', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency (Hz)', fontsize=11)
        ax2.set_ylim([data2['freqs'][0], 30])
        plt.colorbar(im2, ax=ax2, label='Power (dB)')
        
        # === Row 3: Difference (Condition 2 - Condition 1) ===
        ax3 = fig.add_subplot(gs[2, col_idx])
        diff_power = data2['power'] - data1['power']
        im3 = ax3.contourf(data1['times'] * 1000, data1['freqs'], diff_power,
                          levels=20, cmap='RdBu_r', extend='both')
        ax3.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax3.set_title(f'Difference\n{roi_name} ROI', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time (ms)', fontsize=11)
        ax3.set_ylabel('Frequency (Hz)', fontsize=11)
        ax3.set_ylim([data1['freqs'][0], 30])
        plt.colorbar(im3, ax=ax3, label='Power diff (dB)')
    
    plt.suptitle(f'ERSP Comparison - {subject_id}', fontsize=16, fontweight='bold', y=0.995)
    
    # 儲存
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{subject_id}_ersp_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 比較圖已儲存: {output_path}")
    plt.close(fig)


def plot_learning_comparison(results, subject_id, lock_type, output_dir, block_label=None,
                              trial_counts=None):
    """
    繪製 Learning 階段比較圖 (Regular vs Random)，每個 ROI 各自產一張圖。

    Layout: figsize=(18,5)，3 個 subplot 橫排：Regular | Random | Difference
    Regular 和 Random 共用 colorbar 範圍；Difference 獨立 colorbar。
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    _block_suffix = f"_{block_label}" if block_label else ""
    _block_str    = f" | {block_label}" if block_label else ""

    cond_keys = list(results.keys())
    roi_names = list(results[cond_keys[0]].keys())

    for roi_name in roi_names:
        # 新標籤：regular_high vs random_low（主比較）
        if 'regular_high' in cond_keys and 'random_low' in cond_keys:
            data_left  = results['regular_high'][roi_name]
            data_right = results['random_low'][roi_name]
            label_left, label_right = 'Regular High', 'Random Low'
        elif 'high' in cond_keys:
            data_left  = results['high'][roi_name]
            data_right = results['low'][roi_name]
            label_left, label_right = 'High', 'Low'
        else:
            data_left  = results['Regular'][roi_name]
            data_right = results['Random'][roi_name]
            label_left, label_right = 'Regular', 'Random'

        diff_power = data_left['power'] - data_right['power']

        times = data_left['times'] * 1000
        freqs = data_left['freqs']

        x_min, x_max = -500, 500

        t_mask = (times >= x_min) & (times <= x_max)

        # 共用 colorbar 範圍（cond1 / cond2）
        combined = np.concatenate([
            data_left['power'][:, t_mask].ravel(),
            data_right['power'][:, t_mask].ravel()
        ])
        vmax_cond = np.percentile(np.abs(combined), 95)
        vmin_cond = -vmax_cond
        vmax_diff = np.percentile(np.abs(diff_power[:, t_mask].ravel()), 95)
        vmin_diff = -vmax_diff

        # 取得 trial 數標注（若有提供 trial_counts）
        def _n_str(key):
            if trial_counts and key in trial_counts:
                return f"\n(n={trial_counts[key]} trials)"
            return ""

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        def _panel(ax, power, title, vmin, vmax, xlabel=False):
            levels = np.linspace(vmin, vmax, 20)
            im = ax.contourf(times, freqs, power,
                             levels=levels, cmap='RdBu_r',
                             vmin=vmin, vmax=vmax, extend='both')
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency (Hz)', fontsize=11)
            if xlabel:
                ax.set_xlabel('Time (ms)', fontsize=11)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([freqs[0], freqs[-1]])
            return im

        # 決定用哪個 key 查詢 trial_counts
        cond_key_left  = 'regular_high' if 'regular_high' in cond_keys else ('high' if 'high' in cond_keys else 'Regular')
        cond_key_right = 'random_low'   if 'random_low'   in cond_keys else ('low'  if 'low'  in cond_keys else 'Random')

        im1 = _panel(axes[0], data_left['power'],
                     f'{label_left}{_n_str(cond_key_left)}', vmin_cond, vmax_cond)
        plt.colorbar(im1, ax=axes[0], label='Power (dB)')

        im2 = _panel(axes[1], data_right['power'],
                     f'{label_right}{_n_str(cond_key_right)}', vmin_cond, vmax_cond)
        plt.colorbar(im2, ax=axes[1], label='Power (dB)')

        im3 = _panel(axes[2], diff_power,
                     f'Difference ({label_left} - {label_right})', vmin_diff, vmax_diff, xlabel=True)
        plt.colorbar(im3, ax=axes[2], label='Power diff (dB)')

        fig.suptitle(
            f'{subject_id} | {lock_type.capitalize()}-locked | Learning{_block_str} | {roi_name}',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        filename = (
            f'{subject_id}_{lock_type}_lock_learning{_block_suffix}'
            f'_{roi_name}_ersp_comparison.png'
        )
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Learning {roi_name} 比較圖已儲存: {output_path}")
        plt.close(fig)


def plot_testing_comparison(results, subject_id, lock_type, test_type, output_dir, block_label=None,
                             trial_counts=None):
    """
    繪製 Testing 階段比較圖 (Regular vs Random)，每個 ROI 各自產一張圖。

    Layout: figsize=(18,5)，3 個 subplot 橫排：Regular | Random | Difference
    Regular 和 Random 共用 colorbar 範圍；Difference 獨立 colorbar。
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    _block_suffix = f"_{block_label}" if block_label else ""
    _block_str    = f" | {block_label}" if block_label else ""
    phase_label   = f"Testing: {test_type.capitalize()}"

    cond_keys = list(results.keys())
    roi_names = list(results[cond_keys[0]].keys())

    for roi_name in roi_names:
        # 新標籤：regular_high vs random_low（主比較）
        if 'regular_high' in cond_keys and 'random_low' in cond_keys:
            data_left  = results['regular_high'][roi_name]
            data_right = results['random_low'][roi_name]
            label_left, label_right = 'Regular High', 'Random Low'
        elif 'high' in cond_keys:
            data_left  = results['high'][roi_name]
            data_right = results['low'][roi_name]
            label_left, label_right = 'High', 'Low'
        else:
            data_left  = results['Regular'][roi_name]
            data_right = results['Random'][roi_name]
            label_left, label_right = 'Regular', 'Random'

        diff_power = data_left['power'] - data_right['power']

        times = data_left['times'] * 1000
        freqs = data_left['freqs']

        x_min, x_max = -500, 500

        t_mask = (times >= x_min) & (times <= x_max)

        combined = np.concatenate([
            data_left['power'][:, t_mask].ravel(),
            data_right['power'][:, t_mask].ravel()
        ])
        vmax_cond = np.percentile(np.abs(combined), 95)
        vmin_cond = -vmax_cond
        vmax_diff = np.percentile(np.abs(diff_power[:, t_mask].ravel()), 95)
        vmin_diff = -vmax_diff

        # 取得 trial 數標注
        def _n_str_t(key):
            if trial_counts and key in trial_counts:
                return f"\n(n={trial_counts[key]} trials)"
            return ""

        cond_key_left  = 'regular_high' if 'regular_high' in cond_keys else ('high' if 'high' in cond_keys else 'Regular')
        cond_key_right = 'random_low'   if 'random_low'   in cond_keys else ('low'  if 'low'  in cond_keys else 'Random')

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        def _panel(ax, power, title, vmin, vmax, xlabel=False):
            levels = np.linspace(vmin, vmax, 20)
            im = ax.contourf(times, freqs, power,
                             levels=levels, cmap='RdBu_r',
                             vmin=vmin, vmax=vmax, extend='both')
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency (Hz)', fontsize=11)
            if xlabel:
                ax.set_xlabel('Time (ms)', fontsize=11)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([freqs[0], freqs[-1]])
            return im

        im1 = _panel(axes[0], data_left['power'],
                     f'{label_left}{_n_str_t(cond_key_left)}', vmin_cond, vmax_cond)
        plt.colorbar(im1, ax=axes[0], label='Power (dB)')

        im2 = _panel(axes[1], data_right['power'],
                     f'{label_right}{_n_str_t(cond_key_right)}', vmin_cond, vmax_cond)
        plt.colorbar(im2, ax=axes[1], label='Power (dB)')

        im3 = _panel(axes[2], diff_power,
                     f'Difference ({label_left} - {label_right})', vmin_diff, vmax_diff, xlabel=True)
        plt.colorbar(im3, ax=axes[2], label='Power diff (dB)')

        fig.suptitle(
            f'{subject_id} | {lock_type.capitalize()}-locked | {phase_label}{_block_str} | {roi_name}',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        filename = (
            f'{subject_id}_{lock_type}_lock_testing_{test_type}{_block_suffix}'
            f'_{roi_name}_ersp_comparison.png'
        )
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Testing {test_type.capitalize()} {roi_name} 比較圖已儲存: {output_path}")
        plt.close(fig)

def plot_motor_perceptual_comparison(motor_results, perceptual_results,
                                      subject_id, lock_type, output_dir):
    """
    繪製 Motor vs Perceptual 差值比較圖，每個 ROI 各自產一張圖。

    Layout: figsize=(18,5)，3 個 subplot 橫排：
      Left:   label_left Motor - label_left Perceptual
      Center: label_right Motor - label_right Perceptual
      Right:  Interaction = Left - Center
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    cond_keys = list(motor_results.keys())
    if 'regular_high' in cond_keys:
        label_left, label_right = 'regular_high', 'random_low'
        disp_left,  disp_right  = 'Regular High', 'Random Low'
    elif 'high' in cond_keys:
        label_left, label_right = 'high', 'low'
        disp_left,  disp_right  = 'High', 'Low'
    else:
        label_left, label_right = 'Regular', 'Random'
        disp_left,  disp_right  = 'Regular', 'Random'

    if label_left not in motor_results or label_left not in perceptual_results:
        print(f"  ⚠ Motor-Perceptual 差值圖：找不到 {label_left} 資料，跳過")
        return
    if label_right not in motor_results or label_right not in perceptual_results:
        print(f"  ⚠ Motor-Perceptual 差值圖：找不到 {label_right} 資料，跳過")
        return

    roi_names = [r for r in motor_results[label_left].keys()
                 if r in perceptual_results.get(label_left, {})]

    for roi_name in roi_names:
        m_l = motor_results[label_left][roi_name]['power']
        m_r = motor_results[label_right][roi_name]['power']
        p_l = perceptual_results[label_left][roi_name]['power']
        p_r = perceptual_results[label_right][roi_name]['power']
        times = motor_results[label_left][roi_name]['times'] * 1000
        freqs = motor_results[label_left][roi_name]['freqs']

        reg_diff    = m_l - p_l
        ran_diff    = m_r - p_r
        interaction = reg_diff - ran_diff

        x_min, x_max = -500, 500
        t_mask = (times >= x_min) & (times <= x_max)

        combined  = np.concatenate([reg_diff[:, t_mask].ravel(), ran_diff[:, t_mask].ravel()])
        vmax_cond = np.percentile(np.abs(combined), 95)
        vmax_int  = np.percentile(np.abs(interaction[:, t_mask].ravel()), 95)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        def _panel(ax, power, title, vmin, vmax, xlabel=False):
            levels = np.linspace(vmin, vmax, 20)
            im = ax.contourf(times, freqs, power,
                             levels=levels, cmap='RdBu_r',
                             vmin=vmin, vmax=vmax, extend='both')
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency (Hz)', fontsize=11)
            if xlabel:
                ax.set_xlabel('Time (ms)', fontsize=11)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([freqs[0], freqs[-1]])
            return im

        im1 = _panel(axes[0], reg_diff,
                     f'{disp_left} Motor \u2212 {label_left} Perceptual',
                     -vmax_cond, vmax_cond)
        plt.colorbar(im1, ax=axes[0], label='Power diff (dB)')

        im2 = _panel(axes[1], ran_diff,
                     f'{disp_right} Motor \u2212 {label_right} Perceptual',
                     -vmax_cond, vmax_cond)
        plt.colorbar(im2, ax=axes[1], label='Power diff (dB)')

        im3 = _panel(axes[2], interaction,
                     f'Interaction\n({disp_left} M\u2212P) \u2212 ({label_right} M\u2212P)',
                     -vmax_int, vmax_int, xlabel=True)
        plt.colorbar(im3, ax=axes[2], label='Power diff (dB)')

        fig.suptitle(
            f'{subject_id} | {lock_type.capitalize()}-locked | '
            f'Motor vs Perceptual (AllBlocks) | {roi_name}',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        filename = (
            f'{subject_id}_{lock_type}_lock_testing_motor_vs_perceptual'
            f'_{roi_name}_ersp_comparison.png'
        )
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  \u2713 Motor vs Perceptual {roi_name} 比較圖已儲存: {output_path}")
        plt.close(fig)


def plot_triplet_comparison(results, subject_id, lock_type, output_dir,
                            cond1='regular_high', cond2='random_low',
                            phase_label='Learning', block_label=None,
                            trial_counts=None):
    """
    繪製任意兩個 triplet 條件的 ERSP 比較圖。

    預設：Regular High vs Random Low
    也可指定：Regular High vs Random High、Random High vs Random Low 等

    Layout: figsize=(18,5)，3 subplot 橫排：cond1 | cond2 | Difference (cond1 - cond2)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    DISP = {
        'regular_high': 'Regular High',
        'random_high':  'Random High',
        'random_low':   'Random Low',
        'high': 'High', 'low': 'Low',
    }

    if cond1 not in results or cond2 not in results:
        missing = [c for c in [cond1, cond2] if c not in results]
        print(f"  ⚠ plot_triplet_comparison：缺少條件 {missing}，跳過")
        return

    _block_suffix = f"_{block_label}" if block_label else ""
    _block_str    = f" | {block_label}" if block_label else ""
    disp1 = DISP.get(cond1, cond1)
    disp2 = DISP.get(cond2, cond2)

    roi_names = list(results[cond1].keys())

    for roi_name in roi_names:
        d1 = results[cond1][roi_name]
        d2 = results[cond2][roi_name]
        diff_power = d1['power'] - d2['power']
        times = d1['times'] * 1000
        freqs = d1['freqs']
        x_min, x_max = -500, 500
        t_mask = (times >= x_min) & (times <= x_max)

        combined  = np.concatenate([d1['power'][:, t_mask].ravel(),
                                     d2['power'][:, t_mask].ravel()])
        vmax_cond = np.percentile(np.abs(combined), 95)
        vmax_diff = np.percentile(np.abs(diff_power[:, t_mask].ravel()), 95)

        # 取得 trial 數標注
        n_str1 = f"\n(n={trial_counts[cond1]} trials)" if (trial_counts and cond1 in trial_counts) else ""
        n_str2 = f"\n(n={trial_counts[cond2]} trials)" if (trial_counts and cond2 in trial_counts) else ""

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        def _panel(ax, power, title, vmin, vmax, xlabel=False):
            levels = np.linspace(vmin, vmax, 20)
            im = ax.contourf(times, freqs, power,
                             levels=levels, cmap='RdBu_r',
                             vmin=vmin, vmax=vmax, extend='both')
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency (Hz)', fontsize=11)
            if xlabel:
                ax.set_xlabel('Time (ms)', fontsize=11)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([freqs[0], freqs[-1]])
            return im

        im1 = _panel(axes[0], d1['power'], f'{disp1}{n_str1}', -vmax_cond, vmax_cond)
        plt.colorbar(im1, ax=axes[0], label='Power (dB)')

        im2 = _panel(axes[1], d2['power'], f'{disp2}{n_str2}', -vmax_cond, vmax_cond)
        plt.colorbar(im2, ax=axes[1], label='Power (dB)')

        im3 = _panel(axes[2], diff_power,
                     f'Diff ({disp1} − {disp2})',
                     -vmax_diff, vmax_diff, xlabel=True)
        plt.colorbar(im3, ax=axes[2], label='Power diff (dB)')

        fig.suptitle(
            f'{subject_id} | {lock_type.capitalize()}-locked | '
            f'{phase_label}{_block_str} | {roi_name} | {disp1} vs {disp2}',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()

        fname_cond = f"{cond1}_vs_{cond2}"
        filename = (
            f'{subject_id}_{lock_type}_lock_{phase_label.lower()}{_block_suffix}'
            f'_{roi_name}_{fname_cond}_ersp.png'
        )
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {disp1} vs {disp2} | {roi_name} 已儲存: {output_path}")
        plt.close(fig)


def plot_epoch_diff_comparison(data_e1, data_e4, subject_id, lock_type, output_dir,
                                condition_label, trial_type_key,
                                n_e1=None, n_e4=None):
    """
    繪製同一條件 Epoch 4 (Block22-26) vs Epoch 1 (Block7-11) 的 ERSP 比較圖。

    Layout: figsize=(18,5)，3 subplot 橫排：
      Epoch 4 | Epoch 1 | Diff (Epoch 4 − Epoch 1)

    Parameters
    ----------
    data_e1 : dict  {roi_name: power_dict}   Epoch 1 資料（Block7-11）
    data_e4 : dict  {roi_name: power_dict}   Epoch 4 資料（Block22-26）
    subject_id : str
    lock_type : str   'stimulus' 或 'response'
    output_dir : str
    condition_label : str   顯示用名稱，例如 'Regular High'
    trial_type_key : str    檔名用 key，例如 'regular_high'
    n_e1, n_e4 : int or None   trial 數，顯示在標題
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    roi_names = list(data_e1.keys())

    for roi_name in roi_names:
        if roi_name not in data_e4:
            continue

        d1 = data_e1[roi_name]   # Epoch 1
        d4 = data_e4[roi_name]   # Epoch 4
        diff_power = d4['power'] - d1['power']   # Epoch 4 − Epoch 1

        times = d1['times'] * 1000
        freqs = d1['freqs']
        x_min, x_max = -500, 500
        t_mask = (times >= x_min) & (times <= x_max)

        combined  = np.concatenate([d4['power'][:, t_mask].ravel(),
                                     d1['power'][:, t_mask].ravel()])
        vmax_cond = np.percentile(np.abs(combined), 95)
        vmin_cond = -vmax_cond
        vmax_diff = np.percentile(np.abs(diff_power[:, t_mask].ravel()), 95)
        vmin_diff = -vmax_diff
        if vmax_cond < 1e-10: vmax_cond = 1e-10
        if vmax_diff < 1e-10: vmax_diff = 1e-10

        n_str4 = f"\n(n={n_e4} trials)" if n_e4 else ""
        n_str1 = f"\n(n={n_e1} trials)" if n_e1 else ""

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        def _panel(ax, power, title, vmin, vmax, xlabel=False):
            levels = np.linspace(vmin, vmax, 20)
            im = ax.contourf(times, freqs, power,
                             levels=levels, cmap='RdBu_r',
                             vmin=vmin, vmax=vmax, extend='both')
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency (Hz)', fontsize=11)
            if xlabel:
                ax.set_xlabel('Time (ms)', fontsize=11)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([freqs[0], freqs[-1]])
            return im

        im1 = _panel(axes[0], d4['power'],
                     f'Epoch 4 (Block22-26){n_str4}', vmin_cond, vmax_cond)
        plt.colorbar(im1, ax=axes[0], label='Power (dB)')

        im2 = _panel(axes[1], d1['power'],
                     f'Epoch 1 (Block7-11){n_str1}', vmin_cond, vmax_cond)
        plt.colorbar(im2, ax=axes[1], label='Power (dB)')

        im3 = _panel(axes[2], diff_power,
                     'Diff (Epoch 4 − Epoch 1)', vmin_diff, vmax_diff, xlabel=True)
        plt.colorbar(im3, ax=axes[2], label='Power diff (dB)')

        fig.suptitle(
            f'{subject_id} | {lock_type.capitalize()}-locked | '
            f'Learning: Epoch 4 vs Epoch 1 | {condition_label} | {roi_name}',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()

        filename = (
            f'{subject_id}_{lock_type}_lock_learning_epoch4_vs_epoch1'
            f'_{trial_type_key}_{roi_name}_ersp.png'
        )
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Epoch4 vs Epoch1 | {condition_label} | {roi_name} 已儲存: {out_path}")
        plt.close(fig)

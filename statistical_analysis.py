"""
統計分析模組
"""

import numpy as np
from scipy import stats


def compare_regular_vs_random(regular_power, random_power, print_results=True):
    """
    比較 regular 和 random 條件
    
    Parameters
    ----------
    regular_power : ndarray
        Regular 條件的功率
    random_power : ndarray
        Random 條件的功率
    print_results : bool
        是否列印結果
        
    Returns
    -------
    results : dict
        統計結果
    """
    # t-test
    t_stat, p_val = stats.ttest_ind(regular_power, random_power)
    
    # Cohen's d
    pooled_std = np.sqrt((np.std(regular_power)**2 + np.std(random_power)**2) / 2)
    cohens_d = (np.mean(regular_power) - np.mean(random_power)) / pooled_std
    
    results = {
        't_statistic': t_stat,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'regular_mean': np.mean(regular_power),
        'regular_std': np.std(regular_power),
        'random_mean': np.mean(random_power),
        'random_std': np.std(random_power)
    }
    
    if print_results:
        print(f"\nRegular vs Random 比較:")
        print(f"  Regular: M={results['regular_mean']:.4e}, SD={results['regular_std']:.4e}")
        print(f"  Random:  M={results['random_mean']:.4e}, SD={results['random_std']:.4e}")
        print(f"  t({len(regular_power)+len(random_power)-2})={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Cohen's d={cohens_d:.3f}")
        if p_val < 0.05:
            print(f"  *** 顯著差異 (p < 0.05)")
    
    return results


def compare_blocks(block_powers_dict, block_type='learning', print_results=True):
    """
    比較不同 blocks
    
    Parameters
    ----------
    block_powers_dict : dict
        {block_number: power_array}
    block_type : str
        Block 類型描述
    print_results : bool
        是否列印結果
        
    Returns
    -------
    results : dict
        統計結果
    """
    block_numbers = sorted(block_powers_dict.keys())
    block_means = [np.mean(block_powers_dict[b]) for b in block_numbers]
    block_stds = [np.std(block_powers_dict[b]) for b in block_numbers]
    
    # ANOVA
    power_arrays = [block_powers_dict[b] for b in block_numbers]
    f_stat, p_val = stats.f_oneway(*power_arrays)
    
    results = {
        'block_numbers': block_numbers,
        'block_means': block_means,
        'block_stds': block_stds,
        'f_statistic': f_stat,
        'p_value': p_val
    }
    
    if print_results:
        print(f"\n{block_type.capitalize()} Blocks 比較:")
        for i, block_num in enumerate(block_numbers):
            print(f"  Block {block_num}: M={block_means[i]:.4e}, SD={block_stds[i]:.4e}")
        print(f"  F({len(block_numbers)-1}, {sum([len(p) for p in power_arrays])-len(block_numbers)})={f_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            print(f"  *** Blocks 間有顯著差異 (p < 0.05)")
    
    return results


def compute_learning_effect(block_means, block_numbers):
    """
    計算學習效應（線性趨勢）
    
    Parameters
    ----------
    block_means : list or ndarray
        各 block 的平均功率
    block_numbers : list or ndarray
        Block 編號
        
    Returns
    -------
    results : dict
        線性回歸結果
    """
    from scipy.stats import linregress
    
    slope, intercept, r_value, p_value, std_err = linregress(block_numbers, block_means)
    
    results = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }
    
    print(f"\n學習效應分析:")
    print(f"  斜率: {slope:.4e}")
    print(f"  R²: {r_value**2:.3f}")
    print(f"  p={p_value:.4f}")
    if p_value < 0.05:
        direction = "下降" if slope < 0 else "上升"
        print(f"  *** 顯著的{direction}趨勢 (p < 0.05)")
    
    return results


def aggregate_blocks(block_powers_dict, block_groups):
    """
    聚合多個 blocks
    
    Parameters
    ----------
    block_powers_dict : dict
        {block_number: power_array}
    block_groups : dict
        {group_name: [block_numbers]}
        例如 {'learning': [1,2,3,4], 'test': [5,6]}
        
    Returns
    -------
    aggregated : dict
        {group_name: concatenated_power_array}
    """
    aggregated = {}
    
    for group_name, block_list in block_groups.items():
        powers = []
        for block_num in block_list:
            if block_num in block_powers_dict:
                powers.append(block_powers_dict[block_num])
        
        if len(powers) > 0:
            aggregated[group_name] = np.concatenate(powers)
            print(f"{group_name}: {len(aggregated[group_name])} trials (blocks {block_list})")
    
    return aggregated
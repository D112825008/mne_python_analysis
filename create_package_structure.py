"""
EEG分析啟動檔
"""
import shutil
import sys
import os
from pathlib import Path

# 定義項目目錄
PROJECT_ROOT = Path(r'C:\Experiment')

# 定義包名
PACKAGE_NAME = 'mne_python_analysis'

# 定義模組列表
MODULES = [
    '__init__.py',
    'data_io.py',
    'montage.py',
    'signal_processing.py',
    'ica_analysis.py',
    'preprocessing.py',
    'microstate.py',
    'epochs.py',
    'utils.py',
    'spectral_analysis.py',  # ← 新增
    'roi_analysis.py',        # ← 新增
    'asrt_visualization.py',  # ← 新增（如果你要把視覺化分離）
    'main.py'
]

def create_package_structure():
    """建立模組目錄結構"""
    package_dir = PROJECT_ROOT / PACKAGE_NAME
    
    # 建立包目錄（如果不存在）
    package_dir.mkdir(parents=True, exist_ok=True)
    
    # 建立每個模組文件
    for module in MODULES:
        module_path = package_dir / module
        if not module_path.exists():
            module_path.touch()
            print(f"✓ 建立文件: {module_path}")
        else:
            print(f"  文件已存在: {module_path}")
            
def convert_from_exists(force=True):
    """
    從現有的eeg_analysis轉換
    
    Parameters
    ----------
    force : bool
        如果為 True，直接覆蓋不詢問（預設）
    """
    old_package_dir = PROJECT_ROOT / 'eeg_analysis'
    new_package_dir = PROJECT_ROOT / PACKAGE_NAME
    
    # 檢查舊模組是否存在
    if not old_package_dir.exists():
        print(f"⚠️  舊模組目錄不存在: {old_package_dir}")
        print("建立新的模組結構...")
        create_package_structure()
        return
        
    # 如果新模組已存在，根據 force 參數決定是否詢問
    if new_package_dir.exists():
        if not force:
            response = input(f"{new_package_dir} 已存在，是否要覆蓋? (y/n): ")
            if response.lower() != 'y':
                print("❌ 操作取消")
                return
        
        print(f"🗑️  刪除舊目錄: {new_package_dir}")
        shutil.rmtree(new_package_dir)
        
    # 複製舊模組目錄到新模組
    shutil.copytree(old_package_dir, new_package_dir)
    print(f"✓ 複製 {old_package_dir} -> {new_package_dir}")
    
    # 更新所有Python文件中的導入語句
    print("\n更新導入語句...")
    for py_file in new_package_dir.glob('*.py'):
        update_imports(py_file)
    
    print("\n✓ 完成! 請檢查新模組中的文件以確保所有導入語句已正確更新。")

def update_imports(file_path):
    """更新文件中的導入語句"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # 如果 UTF-8 失敗，嘗試其他編碼
        with open(file_path, 'r', encoding='cp950') as f:
            content = f.read()
    
    # 更新導入語句
    updated_content = content.replace('from eeg_analysis', f'from {PACKAGE_NAME}')
    updated_content = updated_content.replace('import eeg_analysis', f'import {PACKAGE_NAME}')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"  ✓ {file_path.name}")

def force_reinstall():
    """
    強制重新安裝套件
    """
    print("\n執行強制重新安裝...")
    print("=" * 60)
    
    import subprocess
    
    try:
        # 先解除安裝
        print("1. 解除安裝舊版本...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", PACKAGE_NAME], 
                      check=False)
        
        # 重新安裝（開發模式）
        print("\n2. 重新安裝（開發模式）...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(PROJECT_ROOT)],
            check=True
        )
        
        if result.returncode == 0:
            print("\n✓ 安裝成功!")
            print(f"可以使用指令: mne-analysis")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 安裝失敗: {e}")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")

if __name__ == "__main__":
    print("="*60)
    print("  MNE Python Analysis 模組結構建立工具")
    print("="*60)
    
    print("\n選擇操作:")
    print("1. 建立新的模組結構")
    print("2. 從現有的 eeg_analysis 轉換（自動覆蓋）")
    print("3. 強制重新安裝套件")
    print("0. 退出")
    
    choice = input("\n請輸入 (0-3): ").strip()
    
    if choice == '1':
        create_package_structure()
        print("\n✓ 目錄結構已建立。請編輯各個模組文件添加必要的程式碼。")
    elif choice == '2':
        convert_from_exists(force=True)  # force=True 直接覆蓋
    elif choice == '3':
        force_reinstall()
    elif choice == '0':
        print("退出")
    else:
        print("❌ 無效的選擇")
        


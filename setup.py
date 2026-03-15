from setuptools import setup, find_packages

setup(
    name="mne_python_analysis",
    version="4.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "mne",
        "scipy",
        "autoreject",
        "mne-icalabel",
        "ipywidgets",
    ],
    entry_points={
        'console_scripts': [
            'mne-analysis=mne_python_analysis.main:main',
        ],
    },
    # 強制覆蓋選項
    zip_safe=False,  # 不使用 zip 格式，直接解壓縮安裝
)
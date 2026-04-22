# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata, collect_dynamic_libs


datas = []
binaries = []


def _safe_collect_data_files(package, includes=None):
    try:
        if includes:
            return collect_data_files(package, includes=includes)
        return collect_data_files(package)
    except TypeError:
        try:
            return collect_data_files(package)
        except Exception:
            return []
    except Exception:
        return []


def _safe_collect_submodules(package):
    try:
        return collect_submodules(package)
    except Exception:
        return []


def _safe_copy_metadata(package):
    try:
        return copy_metadata(package)
    except Exception:
        return []


def _safe_collect_dynamic_libs(package):
    try:
        return collect_dynamic_libs(package)
    except Exception:
        return []

# PaddleOCR v3 delegates named pipelines (for example "OCR") to PaddleX.
# In a frozen app, those YAML/config assets are not discovered automatically;
# without them Paddle raises: "The pipeline (OCR) does not exist".
datas += _safe_collect_data_files(
    'paddlex',
    includes=[
        'configs/**/*.yaml',
        'configs/**/*.yml',
        'configs/**/*.json',
    ],
)
datas += _safe_collect_data_files(
    'paddleocr',
    includes=[
        '**/*.yaml',
        '**/*.yml',
        '**/*.json',
    ],
)

# PaddleX OCR dependency checks use importlib.metadata.version(...) for most extras.
# In a frozen app, code may be bundled while .dist-info metadata is missing, which
# makes PaddleX think OCR extras are unavailable and raises a generic dependency
# error during pipeline creation. Bundle metadata for the OCR extra set explicitly.
for package in [
    'paddlex',
    'paddleocr',
    'openai',
    'google-genai',
    'google-api-core',
    'google-auth',
    'beautifulsoup4',
    'einops',
    'ftfy',
    'imagesize',
    'Jinja2',
    'lxml',
    'opencv-contrib-python',
    'openpyxl',
    'premailer',
    'pyclipper',
    'pypdfium2',
    'python-bidi',
    'regex',
    'safetensors',
    'scikit-learn',
    'scipy',
    'sentencepiece',
    'shapely',
    'tiktoken',
    'tokenizers',
]:
    datas += _safe_copy_metadata(package)

# Paddle's CPU build relies on native DLLs such as mklml.dll/libiomp5md.dll.
# PyInstaller's automatic hooks are not reliably collecting the full paddle/libs
# payload for this app, so bundle paddle dynamic libraries explicitly.
binaries += _safe_collect_dynamic_libs('paddle')

if os.path.isdir('prompts'):
    datas += [
        (os.path.join('prompts', name), 'prompts')
        for name in os.listdir('prompts')
        if os.path.isfile(os.path.join('prompts', name))
    ]

hiddenimports = ['pystray', 'app']
hiddenimports += _safe_collect_submodules('paddlex.inference.pipelines')
hiddenimports += _safe_collect_submodules('paddleocr._pipelines')
hiddenimports += [
    'pyclipper',
    'imagesize',
    'premailer',
    'ftfy',
    'bidi.algorithm',
    'sentencepiece',
    'sentencepiece.sentencepiece_model_pb2',
    'shapely',
    'shapely.geometry',
]


a = Analysis(
    ['client_desktop.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='InvoiceBatchProcessor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='InvoiceBatchProcessor',
)

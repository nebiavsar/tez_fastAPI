# PaddleOCR-VL-1.5 kurulum scripti — Windows PowerShell
#
# Bu script ADR-004'te belirlenen OCR stack'i kurar:
#   - torch (CUDA 11.8 build)
#   - transformers v5+
#   - accelerate
#   - bitsandbytes (4-bit quantize için)
#   - pillow
#
# Tahmini indirme: ~3.3 GB
# Tahmini süre: 10-25 dakika (internet hızına bağlı)
#
# Kullanım:
#   cd C:\Users\nebi\Desktop\tez\tez_fastAPI
#   powershell -ExecutionPolicy Bypass -File scripts\setup_ocr.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== PaddleOCR-VL-1.5 kurulum ===" -ForegroundColor Cyan
Write-Host ""

# Adim 1: Mevcut CPU torch'unu kaldir
Write-Host "[1/4] Mevcut torch (CPU-only) kaldiriliyor..." -ForegroundColor Yellow
pip uninstall -y torch torchvision torchaudio
if (-not $?) { Write-Host "uyari: torch kaldirma basarisiz, devam ediliyor" -ForegroundColor DarkYellow }

# Adim 2: torch CUDA 11.8 build'i kur
Write-Host ""
Write-Host "[2/4] torch+cu118 kuruluyor (~2.7 GB)..." -ForegroundColor Yellow
pip install torch --index-url https://download.pytorch.org/whl/cu118
if (-not $?) { throw "torch kurulumu basarisiz" }

# Adim 3: Diger OCR bagimliliklarini kur
Write-Host ""
Write-Host "[3/4] transformers + accelerate + bitsandbytes + pillow kuruluyor..." -ForegroundColor Yellow
pip install "transformers>=5.0.0" accelerate bitsandbytes pillow
if (-not $?) { throw "transformers/accelerate/bitsandbytes kurulumu basarisiz" }

# Adim 4: CUDA + bitsandbytes dogrulama
Write-Host ""
Write-Host "[4/4] Kurulum dogrulamasi..." -ForegroundColor Yellow
python -c @"
import torch
print(f'torch: {torch.__version__}')
print(f'cuda_available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'cuda_device: {torch.cuda.get_device_name(0)}')
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'vram: {vram_gb:.2f} GB')
else:
    print('UYARI: CUDA bulunamadi! torch GPU build kurulmus mu kontrol et.')

try:
    import transformers
    print(f'transformers: {transformers.__version__}')
except ImportError:
    print('HATA: transformers kurulamadi')

try:
    import bitsandbytes as bnb
    print(f'bitsandbytes: {bnb.__version__}')
except ImportError:
    print('HATA: bitsandbytes kurulamadi')
except Exception as e:
    print(f'UYARI: bitsandbytes import hata verdi: {e}')
"@

Write-Host ""
Write-Host "=== Kurulum tamam ===" -ForegroundColor Green
Write-Host "Sonraki adim: scripts\spike_paddle_vl.py --image <yol> --prompt-strategy s1_raw --quantize 4bit"

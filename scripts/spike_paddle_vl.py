"""
Spike — PaddleOCR-VL-1.5 yerel inference testi.

Amaç: ADR-004'te belirlenen üç hipotezi doğrulamak:
  H1) Türkçe el yazısı doğruluğu kabul edilebilir mi?
  H2) 4-8 GB VRAM GPU'da BF16 ile çalışıyor mu?
  H3) Yerleşik "OCR:" prompt'u dışında custom prompt'larla yapılandırılmış çıkış alınabiliyor mu?

Kullanım:
    # 4-bit quantize (4 GB VRAM için önerilen — ADR-004)
    python scripts/spike_paddle_vl.py --image path/to/exam.jpg --quantize 4bit

    # BF16 (8 GB+ VRAM gerekir)
    python scripts/spike_paddle_vl.py --image path/to/exam.jpg --quantize bf16

    # Custom prompt stratejileri
    python scripts/spike_paddle_vl.py --image exam.jpg --quantize 4bit --prompt-strategy s2_json

    # CPU fallback (yavaş, sadece test)
    python scripts/spike_paddle_vl.py --image exam.jpg --cpu

Bu script production kodu DEĞİL — sadece test/spike. İlerleyen aşamada
app/services/ocr_service.py içine entegre edilecek.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Windows konsolu cp1252 default — Türkçe karakterleri (ı, ğ, ş, vb.) basamaz.
# Hem script'in kendi print'leri hem de OCR çıktısı için UTF-8 şart.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass  # Python < 3.7 fallback

# 4 GB VRAM için bellek fragmantasyonunu önle — torch.cuda import'undan ÖNCE set edilmeli.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL_PATH = "PaddlePaddle/PaddleOCR-VL-1.5"

BUILTIN_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}

CUSTOM_PROMPTS = {
    "s1_raw": "OCR:",
    "s2_json": (
        "Bu bir Türkçe açık uçlu sınav kâğıdıdır. Aşağıdaki JSON şemasına uyarak "
        "soru numaralarını ve öğrencinin el yazısı cevaplarını çıkar. "
        "Üst bilgi (öğrenci ismi, okul), marj notları ve sayfa numaralarını yok say. "
        "Sadece geçerli JSON döndür, açıklama yazma.\n\n"
        '[{"question_number": "1", "question_text": "...", "extracted_answer": "..."}]'
    ),
    "s3_spot": "Spotting:",
}


def load_model(device: str, quantize: str):
    print(f"[load] model: {MODEL_PATH}")
    print(f"[load] device: {device}, quantize: {quantize}")
    t0 = time.perf_counter()

    load_kwargs: dict = {}

    if quantize == "4bit":
        if device != "cuda":
            print("[load] HATA: 4-bit quantize sadece CUDA'da çalışır", file=sys.stderr)
            sys.exit(1)
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        if device != "cuda":
            print("[load] HATA: 8-bit quantize sadece CUDA'da çalışır", file=sys.stderr)
            sys.exit(1)
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif quantize == "fp32":
        load_kwargs["torch_dtype"] = torch.float32
    else:
        raise ValueError(f"bilinmeyen quantize: {quantize}")

    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, **load_kwargs)

    if quantize in {"4bit", "8bit"}:
        model = model.eval()  # quantize edilmişler zaten device_map='auto' veya cuda'ya yerleşir
    else:
        model = model.to(device).eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    elapsed = time.perf_counter() - t0
    print(f"[load] tamamlandı — {elapsed:.1f} sn")

    if device == "cuda" and torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / (1024**3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[load] VRAM kullanılan: {vram_alloc:.2f} GB / toplam: {vram_total:.2f} GB")

    return model, processor


def run_inference(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    task: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # PaddleOCRVLImageProcessor'ün kendi default size'ı var
    # (shortest_edge=112896, longest_edge=1003520 ≈ 1280 patches × 28²).
    # Resmi örnek koddaki min_pixels/max_pixels override'ı bu versiyonda gereksiz.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Degeneration önlemi — ilk denemede model "AİLİAİAİ..." gibi tekrar döngüsüne girmişti.
    # Üç katmanlı koruma:
    #   - repetition_penalty>1: aynı token'ı tekrar etmeyi cezalandırır (1.3 = orta)
    #   - no_repeat_ngram_size: 4 token'lık aynı diziyi ikinci kez yasaklar
    #   - low temperature + sampling: greedy'nin trap'lerinden çıkar, ama hâlâ deterministik gibi
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
    elapsed = time.perf_counter() - t0
    n_input = inputs["input_ids"].shape[-1]
    n_generated = outputs.shape[-1] - n_input
    print(f"[gen] generate süresi: {elapsed:.2f} sn ({n_generated} üretilen / {n_input} girdi token)")

    decoded = processor.decode(outputs[0][inputs["input_ids"].shape[-1] : -1])
    return decoded


def try_parse_json(text: str) -> list | None:
    """S2 stratejisi için: cevap içindeki JSON array'i çıkar."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="PaddleOCR-VL-1.5 spike testi")
    parser.add_argument("--image", required=True, type=Path, help="Sınav kâğıdı görüntüsü")
    parser.add_argument(
        "--task",
        choices=list(BUILTIN_PROMPTS.keys()),
        default="ocr",
        help="Yerleşik task (sadece --prompt-strategy verilmediğinde kullanılır)",
    )
    parser.add_argument(
        "--prompt-strategy",
        choices=list(CUSTOM_PROMPTS.keys()),
        default=None,
        help="Spike stratejisi: s1_raw | s2_json | s3_spot",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=640,
        help="Üretilecek max token sayısı. 4 GB VRAM için 640 önerilen (KV cache sığar).",
    )
    parser.add_argument("--cpu", action="store_true", help="GPU varsa bile CPU kullan")
    parser.add_argument(
        "--quantize",
        choices=["4bit", "8bit", "bf16", "fp32"],
        default="4bit",
        help="Precision/quantization (4 GB VRAM için 4bit önerilir)",
    )

    args = parser.parse_args()

    if not args.image.exists():
        print(f"[hata] görüntü bulunamadı: {args.image}", file=sys.stderr)
        return 1

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cpu" and args.quantize in {"4bit", "8bit"}:
        print(f"[uyarı] CPU'da {args.quantize} desteklenmiyor, fp32'ye düşürülüyor")
        args.quantize = "fp32"

    if args.prompt_strategy:
        prompt_text = CUSTOM_PROMPTS[args.prompt_strategy]
        task_for_size = "spotting" if args.prompt_strategy == "s3_spot" else "ocr"
        print(f"[prompt] strateji: {args.prompt_strategy}")
    else:
        prompt_text = BUILTIN_PROMPTS[args.task]
        task_for_size = args.task
        print(f"[prompt] yerleşik task: {args.task}")

    image = Image.open(args.image).convert("RGB")
    print(f"[image] {args.image.name} — {image.size[0]}x{image.size[1]}")

    model, processor = load_model(device, args.quantize)
    output = run_inference(
        model=model,
        processor=processor,
        image=image,
        prompt_text=prompt_text,
        task=task_for_size,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n" + "=" * 60)
    print("MODEL ÇIKIŞI")
    print("=" * 60)
    print(output)
    print("=" * 60)

    if args.prompt_strategy == "s2_json":
        parsed = try_parse_json(output)
        print("\n[s2_json] JSON parse denemesi:")
        if parsed is not None:
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
            print(f"[s2_json] başarılı — {len(parsed)} kayıt")
        else:
            print("[s2_json] BAŞARISIZ — model JSON üretmedi")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Spike — Qwen2.5-VL-3B-Instruct yerel inference testi.

PaddleOCR-VL-1.5 spike'ı el yazısında zayıf kaldı (spike_output_s1_v2.txt).
Bu spike, Qwen2.5-VL'in Türkçe el yazısında daha iyi olup olmadığını ölçer.

Kullanım:
    python scripts/spike_qwen_vl.py --image ornek3.jpeg --prompt-strategy s1_raw
    python scripts/spike_qwen_vl.py --image ornek3.jpeg --prompt-strategy s2_json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPTS = {
    "s1_raw": (
        "Bu görüntüdeki tüm metni satır satır oku ve yaz. "
        "Hem basılı metni hem de el yazısı kısımları olduğu gibi çıkar. "
        "Türkçe karakterlere dikkat et."
    ),
    "s2_json": (
        "Bu bir Türkçe açık uçlu sınav kâğıdıdır. Aşağıdaki JSON şemasına uyarak "
        "soru numaralarını ve öğrencinin el yazısı cevaplarını çıkar. "
        "Üst bilgi (öğrenci ismi, okul, sınıf, numara), marj notları ve sayfa numaralarını yok say. "
        "Sadece geçerli JSON döndür, açıklama yazma.\n\n"
        'Şema: [{"question_number": "1", "question_text": "...", "extracted_answer": "..."}]'
    ),
    "s3_handwriting_only": (
        "Bu görüntüde basılı (matbu) sorulara öğrenci tarafından elle yazılmış cevaplar var. "
        "Sadece öğrencinin el yazısıyla yazdığı cevapları soru numarasıyla eşleştirip listele. "
        "Format: '1) cevap metni' şeklinde her satır ayrı."
    ),
}


def load_model(quantize: str):
    print(f"[load] model: {MODEL_PATH}")
    print(f"[load] quantize: {quantize}")
    t0 = time.perf_counter()

    load_kwargs: dict = {"device_map": "cuda:0"}

    if quantize == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        raise ValueError(f"bilinmeyen quantize: {quantize}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, **load_kwargs).eval()

    # 4 GB VRAM için image token sayısını agresif sınırla.
    # Default ~3056 token (önceki run OOM). 512 token = ~28x18 patch grid = ~400x500 px.
    # Sınav metni okunabilirlik için yeterli, KV cache çok daha küçük.
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=128 * 28 * 28,   # ~100k pixels min
        max_pixels=512 * 28 * 28,   # ~400k pixels max ≈ 512 image token
    )

    elapsed = time.perf_counter() - t0
    print(f"[load] tamamlandı — {elapsed:.1f} sn")

    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / (1024**3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[load] VRAM kullanılan: {vram_alloc:.2f} GB / toplam: {vram_total:.2f} GB")

    return model, processor


def run_inference(model, processor, image: Image.Image, prompt_text: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    n_input = inputs["input_ids"].shape[-1]
    print(f"[gen] girdi token: {n_input}, max_new: {max_new_tokens}")

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
    elapsed = time.perf_counter() - t0
    n_generated = outputs.shape[-1] - n_input
    print(f"[gen] generate süresi: {elapsed:.2f} sn ({n_generated} üretilen token)")

    decoded = processor.batch_decode(
        outputs[:, n_input:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return decoded


def try_parse_json(text: str) -> list | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-3B spike testi")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--prompt-strategy",
        choices=list(PROMPTS.keys()),
        default="s1_raw",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--quantize", choices=["4bit", "bf16"], default="4bit")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"[hata] görüntü bulunamadı: {args.image}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("[hata] CUDA yok, Qwen2.5-VL CPU'da çok yavaş — durduruluyor", file=sys.stderr)
        return 1

    prompt_text = PROMPTS[args.prompt_strategy]
    print(f"[prompt] strateji: {args.prompt_strategy}")

    image = Image.open(args.image).convert("RGB")
    print(f"[image] {args.image.name} — {image.size[0]}x{image.size[1]}")

    model, processor = load_model(args.quantize)
    output = run_inference(model, processor, image, prompt_text, args.max_new_tokens)

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

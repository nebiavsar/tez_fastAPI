"""parse_s3_markdown için unit testler.

Bu testler GPU/model gerektirmez — gerçek spike çıkışlarını fixture olarak
alıp parse mantığını lokal makinede doğrular.

Çalıştırmak için:
    pytest tests/test_ocr_parser.py -v
"""

from __future__ import annotations

from app.ml.ocr_parser import parse_s3_markdown


# Gerçek spike çıkışı (ornek3.jpeg, Qwen2.5-VL-7B, S3 prompt)
REAL_S3_OUTPUT = """Tabii ki! İşte öğrencinin el ile yazdığı cevapler:

**1)** Pb(NO₃)₂(suda)+2KI(suda)->PbI₂(k)+2KNO₃(suda)

**2)** Suyun buharlaşması: Kondensel
   Kağıdın yanması: Kimyasal
   Tuzun suda çözünmesi: Kimyasall
   Elmanın kararması: Kimyasell

**3**)
- pH değişimi: 4
- Gaz çıkışı: 2
- Enerji değişimı: 6
- Katı oluşumu: 7
- Asallık/Aktiflik: 5
- İletkenlik değişimi:8
- Renk değişimi:9
- Işık oluşumu:

**4**)
- Tebkime Denklemi: Çözelme
- 2Mg(k)+O₂(g)->2MgO(k): Oksidasyon tepkimesi
- NaOH(suda)+HCl(suda)->NaCl(suda)+ H₂O(s): Aşit-Baz tepkimesisi

**5**)
- CaCO₃(k)+2HCI(suda)->CaCI₂(suda)(suda)+H₂O(l)+CO₂(e): Reaksiyo
- Mg(k)+Cu(NO₃)_2(sud)->Mg(NO₃)(suda)+(Cu(k): Reaktifler
"""


def test_parse_real_s3_output_extracts_five_questions() -> None:
    result = parse_s3_markdown(REAL_S3_OUTPUT)
    numbers = [a.question_number for a in result]
    assert numbers == ["1", "2", "3", "4", "5"], f"Beklenen 5 soru, alınan: {numbers}"


def test_parse_real_s3_first_question_has_chemistry_equation() -> None:
    result = parse_s3_markdown(REAL_S3_OUTPUT)
    first = next(a for a in result if a.question_number == "1")
    assert "Pb(NO₃)₂" in first.extracted_answer
    assert "PbI₂" in first.extracted_answer


def test_parse_real_s3_second_question_keeps_multiline_answers() -> None:
    result = parse_s3_markdown(REAL_S3_OUTPUT)
    second = next(a for a in result if a.question_number == "2")
    # 2. soru 4 alt cevap içeriyordu, hepsi tek extracted_answer'da satır satır
    assert "Suyun buharlaşması" in second.extracted_answer
    assert "Kağıdın yanması" in second.extracted_answer
    assert "Tuzun suda çözünmesi" in second.extracted_answer
    assert "Elmanın kararması" in second.extracted_answer


def test_parse_real_s3_question_text_empty() -> None:
    """OCR question_text döndürmez — Spring tarafı dolduracak."""
    result = parse_s3_markdown(REAL_S3_OUTPUT)
    assert all(a.question_text == "" for a in result)


def test_parse_empty_input_returns_empty_list() -> None:
    assert parse_s3_markdown("") == []
    assert parse_s3_markdown("Sadece düz metin, soru numarası yok.") == []


def test_parse_plain_numbered_fallback() -> None:
    """**N)** formatı yoksa düz '1.' / '1)' fallback'i kullan."""
    text = (
        "1) Cevap birinci\n"
        "2. Cevap ikinci\n"
        "3) Cevap üçüncü uzun bir cümle\n"
    )
    result = parse_s3_markdown(text)
    assert [a.question_number for a in result] == ["1", "2", "3"]
    assert result[0].extracted_answer == "Cevap birinci"
    assert result[2].extracted_answer == "Cevap üçüncü uzun bir cümle"


def test_parse_strips_bullet_markers() -> None:
    text = "**1)** - madde A\n- madde B\n* madde C"
    result = parse_s3_markdown(text)
    assert result[0].extracted_answer == "madde A\nmadde B\nmadde C"


# --- 2026-05-22 v2 fix testleri (safety net'ler) ---


def test_parse_truncates_at_leaked_question_number() -> None:
    """Soru 4'ün bloğuna soru 5'in cevabı sızmışsa kes — gerçek 2026-05-22 bug'ı."""
    text = (
        "**4)** Çökelme tepkimesi\n"
        "5)\n"
        "CaCO3 + 2HCl -> CaCl2 + H2O + CO2: tepkime denklemi"
    )
    result = parse_s3_markdown(text)
    by_id = {a.question_number: a for a in result}
    assert "4" in by_id
    # Soru 4'ün cevabında "CaCO3" gibi 5. sorunun içeriği bulunmamalı
    assert "CaCO3" not in by_id["4"].extracted_answer
    assert "Çökelme" in by_id["4"].extracted_answer


def test_parse_filters_hallucination_lines() -> None:
    """Model 'Yanlış cevap:', 'Not:' gibi kendi yorumlarını eklemişse ele."""
    text = (
        "**2)** Suyun buharlaşması: Kondensel\n"
        "Yanlış cevap: Kondensal\n"
        "Not: Yanlış işaretlenmiştir.\n"
        "Kağıdın yanması: Kimyasal"
    )
    result = parse_s3_markdown(text)
    body = result[0].extracted_answer
    assert "Suyun buharlaşması: Kondensel" in body
    assert "Kağıdın yanması: Kimyasal" in body
    assert "Yanlış cevap" not in body
    assert "Not:" not in body


def test_parse_skips_empty_answer_marker() -> None:
    """'**N)** (boş)' → soru atlanır, listede görünmez."""
    text = (
        "**1)** Pb(NO3)2 + 2KI -> PbI2\n"
        "**2)** (boş)\n"
        "**3)** Kısmi cevap burada"
    )
    result = parse_s3_markdown(text)
    numbers = [a.question_number for a in result]
    assert numbers == ["1", "3"]


def test_parse_real_2026_05_22_bug_case() -> None:
    """End-to-end test'te (ornek3.jpeg) gözlenen gerçek hata vakası."""
    # Soru 2 halüsinasyon + soru 4-5 karışıklığı — birlikte
    text = (
        "**1)** Pb(NO₃)₂(suda)+2KI(suda)-> PbI₂(k)+ 2KNO₃(suda)\n"
        "Çözelme tepkimesi\n"
        "\n"
        "**2)** Suyun buharlaşması: Kondensel\n"
        "Yanlış cevap: Kondensal\n"
        "Not: Yanlış işaretlenmiştir.\n"
        "\n"
        "**4)** Çökelme tepkimesi\n"
        "5)\n"
        "CaCO₃(k)+2HCl(suda) tepkime denklemi"
    )
    result = parse_s3_markdown(text)
    by_id = {a.question_number: a for a in result}

    # Soru 1: olduğu gibi
    assert "Pb(NO₃)₂" in by_id["1"].extracted_answer
    assert "Çözelme tepkimesi" in by_id["1"].extracted_answer

    # Soru 2: halüsinasyon temizlenmiş
    assert "Kondensel" in by_id["2"].extracted_answer
    assert "Yanlış cevap" not in by_id["2"].extracted_answer
    assert "Not:" not in by_id["2"].extracted_answer

    # Soru 4: soru 5'in içeriği sızmamış
    assert "Çökelme" in by_id["4"].extracted_answer
    assert "CaCO" not in by_id["4"].extracted_answer

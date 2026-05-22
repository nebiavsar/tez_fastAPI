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

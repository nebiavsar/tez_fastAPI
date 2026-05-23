"""parse_s3_markdown için unit testler (section-based type detection, v3).

VLM tip etiketi yazma denemesi başarısız oldu (canlı testte 2026-05-22), bunun
yerine section başlıklarına dayanan deterministik tip tespitine geçildi.
Pattern heuristic'leri (tek harf MC, "label: value" FB) kaldırıldı.

3 strateji:
    "Çoktan Seçmeli ..." → MULTIPLE_CHOICE
    "Boşluk Doldurma" / "Eşleştirme" / "Matching" → FILL_BLANK
    Başka başlık yok → OPEN_ENDED (default)
"""

from __future__ import annotations

from app.ml.ocr_parser import parse_s3_markdown
from app.schemas import QuestionType


# Gerçek spike çıkışı (ornek3.jpeg, Qwen2.5-VL-7B)
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

**4**)
- Tebkime Denklemi: Çözelme

**5**)
- CaCO₃(k)+2HCI(suda)->CaCI₂(suda)+H₂O(l)+CO₂(e)
"""


def test_parse_real_s3_output_extracts_five_questions() -> None:
    result = parse_s3_markdown(REAL_S3_OUTPUT)
    numbers = [a.question_number for a in result]
    assert numbers == ["1", "2", "3", "4", "5"]


def test_parse_real_s3_first_question_has_chemistry_equation() -> None:
    result = parse_s3_markdown(REAL_S3_OUTPUT)
    first = next(a for a in result if a.question_number == "1")
    assert "Pb(NO₃)₂" in first.extracted_answer


def test_parse_real_s3_default_type_is_open_ended() -> None:
    """Section başlığı yok → tüm sorular OPEN_ENDED."""
    result = parse_s3_markdown(REAL_S3_OUTPUT)
    assert all(a.question_type == QuestionType.OPEN_ENDED for a in result)


def test_parse_real_s3_question_text_empty() -> None:
    result = parse_s3_markdown(REAL_S3_OUTPUT)
    assert all(a.question_text == "" for a in result)


def test_parse_empty_input_returns_empty_list() -> None:
    assert parse_s3_markdown("") == []
    assert parse_s3_markdown("Sadece düz metin, soru numarası yok.") == []


def test_parse_plain_numbered_fallback() -> None:
    text = (
        "1) Cevap birinci\n"
        "2. Cevap ikinci\n"
        "3) Cevap üçüncü uzun bir cümle\n"
    )
    result = parse_s3_markdown(text)
    assert [a.question_number for a in result] == ["1", "2", "3"]


def test_parse_strips_bullet_markers() -> None:
    text = "**1)** - madde A\n- madde B\n* madde C"
    result = parse_s3_markdown(text)
    assert result[0].extracted_answer == "madde A\nmadde B\nmadde C"


def test_parse_filters_hallucination_lines() -> None:
    text = (
        "**2)** Suyun buharlaşması: Kondensel\n"
        "Yanlış cevap: Kondensal\n"
        "Not: Yanlış işaretlenmiştir.\n"
        "Kağıdın yanması: Kimyasal"
    )
    result = parse_s3_markdown(text)
    body = result[0].extracted_answer
    assert "Suyun buharlaşması: Kondensel" in body
    assert "Yanlış cevap" not in body
    assert "Not:" not in body


def test_parse_skips_empty_answer_marker() -> None:
    text = (
        "**1)** Pb(NO3)2 + 2KI -> PbI2\n"
        "**2)** (boş)\n"
        "**3)** Kısmi cevap burada"
    )
    result = parse_s3_markdown(text)
    numbers = [a.question_number for a in result]
    assert numbers == ["1", "3"]


def test_parse_subquestion_grouping() -> None:
    text = (
        "**2a)** Karasal\n"
        "**2b)** Kimyasal\n"
    )
    result = parse_s3_markdown(text)
    assert len(result) == 1
    assert result[0].question_number == "2"
    assert "a) Karasal" in result[0].extracted_answer
    assert "b) Kimyasal" in result[0].extracted_answer


# --- Section-based type detection (v3, 2026-05-23) ---


def test_section_multiple_choice_tr() -> None:
    text = (
        "Çoktan Seçmeli Sorular\n"
        "**1)** C\n"
        "**2)** A\n"
    )
    result = parse_s3_markdown(text)
    assert all(a.question_type == QuestionType.MULTIPLE_CHOICE for a in result)


def test_section_multiple_choice_en() -> None:
    text = (
        "Multiple Choice Questions\n"
        "**1)** C\n"
    )
    result = parse_s3_markdown(text)
    assert result[0].question_type == QuestionType.MULTIPLE_CHOICE


def test_section_fill_blank_bosluk_doldurma() -> None:
    text = (
        "Boşluk Doldurma\n"
        "**1)** fiziksel\n"
        "**2)** kimyasal\n"
    )
    result = parse_s3_markdown(text)
    assert all(a.question_type == QuestionType.FILL_BLANK for a in result)


def test_section_fill_blank_eslestirme() -> None:
    """'Eşleştirme' başlığı da FILL_BLANK'a yönlendirilir."""
    text = (
        "Eşleştirme\n"
        "**3)** a→4, b→2, c→9\n"
    )
    result = parse_s3_markdown(text)
    assert result[0].question_type == QuestionType.FILL_BLANK


def test_section_fill_blank_matching_en() -> None:
    text = (
        "Matching\n"
        "**3)** a-4, b-2\n"
    )
    result = parse_s3_markdown(text)
    assert result[0].question_type == QuestionType.FILL_BLANK


def test_section_no_header_means_open_ended() -> None:
    """Hiç section başlığı yok → her şey OPEN_ENDED."""
    text = (
        "**1)** Bu açık uçlu bir cevap, herhangi bir başlık yok\n"
        "**2)** Yine açık uçlu\n"
    )
    result = parse_s3_markdown(text)
    assert all(a.question_type == QuestionType.OPEN_ENDED for a in result)


def test_section_mixed_full_paper() -> None:
    """Karmaşık senaryo: 3 section, her birinde sorular var."""
    text = (
        "Açık Uçlu Sorular\n"
        "**1)** Pb(NO3)2 + 2KI -> PbI2 + 2KNO3\n"
        "\n"
        "Çoktan Seçmeli Sorular\n"
        "**2)** C\n"
        "**3)** A\n"
        "\n"
        "Boşluk Doldurma\n"
        "**4a)** fiziksel\n"
        "**4b)** kimyasal\n"
        "\n"
        "Eşleştirme\n"
        "**5)** a→4, b→2, c→9\n"
    )
    result = parse_s3_markdown(text)
    by_id = {a.question_number: a.question_type for a in result}

    # 1: section "Açık Uçlu" tanınmıyor → default OPEN_ENDED ✓
    assert by_id["1"] == QuestionType.OPEN_ENDED
    # 2, 3: Çoktan Seçmeli section'ı
    assert by_id["2"] == QuestionType.MULTIPLE_CHOICE
    assert by_id["3"] == QuestionType.MULTIPLE_CHOICE
    # 4 (grouped 4a + 4b): Boşluk Doldurma section'ı
    assert by_id["4"] == QuestionType.FILL_BLANK
    # 5: Eşleştirme section'ı → FILL_BLANK
    assert by_id["5"] == QuestionType.FILL_BLANK


def test_section_header_case_insensitive() -> None:
    text = (
        "ÇOKTAN SEÇMELİ SORULAR\n"
        "**1)** B\n"
    )
    result = parse_s3_markdown(text)
    assert result[0].question_type == QuestionType.MULTIPLE_CHOICE


def test_section_header_with_dashes() -> None:
    """'Multiple-Choice' veya 'Fill-in-the-Blank' gibi tireli varyantlar."""
    text = (
        "Multiple-Choice Questions\n"
        "**1)** C\n"
    )
    result = parse_s3_markdown(text)
    assert result[0].question_type == QuestionType.MULTIPLE_CHOICE


# --- VLM tutarsız header formatları (canlı testte 2026-05-23 gözlemlendi) ---


def test_parse_markdown_h3_header_format() -> None:
    """VLM bazen '### 1)' gibi Markdown H3 başlığı kullanıyor."""
    text = (
        "### 1) Pb(NO₃)₂ + 2KI → PbI₂ + 2KNO₃\n"
        "\n"
        "### 2) Suyun buharlaşması: Fiziksel\n"
    )
    result = parse_s3_markdown(text)
    assert len(result) == 2
    assert result[0].question_number == "1"
    assert result[1].question_number == "2"
    assert "Pb(NO₃)₂" in result[0].extracted_answer


def test_parse_markdown_h2_header_subquestion() -> None:
    """## 4a) gibi H2 + sub-question kombinasyonu."""
    text = (
        "## 4a) Karasal\n"
        "## 4b) Kimyasal\n"
    )
    result = parse_s3_markdown(text)
    assert len(result) == 1
    assert result[0].question_number == "4"
    assert "Karasal" in result[0].extracted_answer
    assert "Kimyasal" in result[0].extracted_answer


def test_parse_mixed_header_formats() -> None:
    """Bazı sorularda '### 1)', bazılarında '**2)**' karışık."""
    text = (
        "### 1) Birinci cevap denklemi\n"
        "**2)** İkinci cevap\n"
        "3) Üçüncü cevap düz format\n"
    )
    result = parse_s3_markdown(text)
    assert [a.question_number for a in result] == ["1", "2", "3"]

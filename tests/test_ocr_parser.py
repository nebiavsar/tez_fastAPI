"""Parser testleri (v4, 2026-05-23 — yıldızlı section + compound ID + puan parse).

Standart sınav formatı:
    *çoktan seçmeli soru
    1) ... (10p)
    A) ... B) ... C) ... D) ...

    *boşluk doldurma
    1) ... (5p)

    1-) açık uçlu (10p)

Compound IDs: mc1, mc2, fb1, fb2, oe1, oe2, oe3
"""

from __future__ import annotations

from app.ml.ocr_parser import (
    _classify_section_text,
    _extract_points,
    parse_s3_markdown,
)
from app.schemas import QuestionType


# --- Helper fonksiyonları ---


def test_classify_section_multiple_choice_tr() -> None:
    assert _classify_section_text("çoktan seçmeli soru") == QuestionType.MULTIPLE_CHOICE
    assert _classify_section_text("Çoktan Seçmeli Sorular") == QuestionType.MULTIPLE_CHOICE
    assert _classify_section_text("ÇOKTAN SEÇMELİ") == QuestionType.MULTIPLE_CHOICE


def test_classify_section_multiple_choice_en() -> None:
    assert _classify_section_text("multiple choice") == QuestionType.MULTIPLE_CHOICE
    assert _classify_section_text("Multiple Choice Question") == QuestionType.MULTIPLE_CHOICE
    assert _classify_section_text("multiple-choice questions") == QuestionType.MULTIPLE_CHOICE


def test_classify_section_fill_blank() -> None:
    assert _classify_section_text("boşluk doldurma") == QuestionType.FILL_BLANK
    assert _classify_section_text("eşleştirme") == QuestionType.FILL_BLANK
    assert _classify_section_text("fill in the blanks") == QuestionType.FILL_BLANK
    assert _classify_section_text("matching") == QuestionType.FILL_BLANK


def test_classify_section_open_ended() -> None:
    assert _classify_section_text("açık uçlu") == QuestionType.OPEN_ENDED
    assert _classify_section_text("klasik") == QuestionType.OPEN_ENDED
    assert _classify_section_text("kısa cevaplı") == QuestionType.OPEN_ENDED
    assert _classify_section_text("uzun cevaplı") == QuestionType.OPEN_ENDED


def test_classify_section_unknown() -> None:
    """Tanınmayan section başlığı → None."""
    assert _classify_section_text("Not:") is None
    assert _classify_section_text("bla bla") is None


def test_extract_points_variants() -> None:
    assert _extract_points("Soru metni (10p)") == 10
    assert _extract_points("(5p)") == 5
    assert _extract_points("(10 puan)") == 10
    assert _extract_points("Cevap (15 pt)") == 15
    assert _extract_points("(20 points)") == 20
    # Yoksa default
    assert _extract_points("Soru metni", default=10) == 10
    assert _extract_points("Soru metni", default=5) == 5


# --- Parser entegrasyon testleri ---


def test_starred_mc_section_assigns_mc_type() -> None:
    text = (
        "*çoktan seçmeli soru\n"
        "1) Türkiye'nin coğrafi konumu? (10p)\n"
        "A) Devşirme B) İskân C) Tımar D) İltizam\n"
    )
    result = parse_s3_markdown(text)
    assert len(result) == 1
    assert result[0].question_number == "mc1"
    assert result[0].question_type == QuestionType.MULTIPLE_CHOICE
    assert result[0].max_score == 10


def test_starred_fb_section_assigns_fb_type() -> None:
    text = (
        "*boşluk doldurma\n"
        "1) Dünya'nın en büyük yüzölçüme sahip ülkesi ...'dir. (5p)\n"
    )
    result = parse_s3_markdown(text)
    assert len(result) == 1
    assert result[0].question_number == "fb1"
    assert result[0].question_type == QuestionType.FILL_BLANK
    assert result[0].max_score == 5


def test_no_section_defaults_to_open_ended() -> None:
    text = (
        "1-) Fotosentez nedir? (10p)\n"
        "Cevap: bitkilerin güneş ışığı kullanarak besin üretmesidir.\n"
    )
    result = parse_s3_markdown(text)
    assert len(result) == 1
    assert result[0].question_number == "oe1"
    assert result[0].question_type == QuestionType.OPEN_ENDED
    assert result[0].max_score == 10


def test_repeated_numbers_get_distinct_compound_ids() -> None:
    """Ahmet Yesevi formatında: çoktan seçmeli 1,2 + boşluk doldurma 1,2 + açık uçlu 1,2,3.

    Parser tekrarlı numaraları section prefix ile ayırır: mc1, mc2, fb1, fb2, oe1, oe2, oe3.
    """
    text = (
        "*çoktan seçmeli soru\n"
        "1) MC soru 1 (10p)\n"
        "*çoktan seçmeli soru\n"
        "2) MC soru 2 (10p)\n"
        "\n"
        "*boşluk doldurma\n"
        "1) FB soru 1 (5p)\n"
        "*boşluk doldurma\n"
        "2) FB soru 2 (5p)\n"
        "\n"
        "1-) OE soru 1 (10p)\n"
        "Cevap: lorem ipsum\n"
        "2-) OE soru 2 (10p)\n"
        "Cevap: dolor sit\n"
        "3-) OE soru 3 (10p)\n"
        "Cevap: amet\n"
    )
    result = parse_s3_markdown(text)
    ids = [a.question_number for a in result]
    # Sıra: mc önce, fb sonra, oe en son
    assert ids == ["mc1", "mc2", "fb1", "fb2", "oe1", "oe2", "oe3"]


def test_per_question_max_score_is_parsed() -> None:
    text = (
        "*çoktan seçmeli soru\n"
        "1) Soru bir (10p)\n"
        "*çoktan seçmeli soru\n"
        "2) Soru iki (15p)\n"
        "*boşluk doldurma\n"
        "1) Boşluk (5p)\n"
    )
    result = parse_s3_markdown(text)
    by_id = {a.question_number: a.max_score for a in result}
    assert by_id["mc1"] == 10
    assert by_id["mc2"] == 15
    assert by_id["fb1"] == 5


def test_ahmet_yesevi_full_paper() -> None:
    """Gerçek 'Ahmet Yesevi Ortaokulu' sınav kâğıdı formatına benzer simülasyon."""
    text = (
        "Ahmet Yesevi Ortaokulu\n"
        "2. dönem 1. yazılı soruları\n"
        "Ad: Soyad: Sınıf:\n"
        "\n"
        "*Çoktan seçmeli soru\n"
        "1) Türkiye'nin coğrafi konumu düşünüldüğünde, doğu ve batı uçları arasındaki yerel saat farkı kaç dakikadır? (10p)\n"
        "A) Devşirme Sistemi\n"
        "B) İskân Politikası\n"
        "C) Tımar Sistemi\n"
        "D) İltizam Sistemi\n"
        "\n"
        "*Çoktan seçmeli soru\n"
        "2) 'Sinekli Bakkal' ve 'Türk'ün Ateşle İmtihanı' gibi eserlerin yazarı olan ünlü edebiyatçımız kimdir? (10p)\n"
        "A) Halide Edip Adıvar\n"
        "B) Reşat Nuri Güntekin\n"
        "C) Yakup Kadri Karaosmanoğlu\n"
        "D) Sait Faik Abasıyanık\n"
        "\n"
        "*boşluk doldurma\n"
        "1) Dünya'nın en büyük yüzölçümüne sahip ülkesi ............'dir. (5p)\n"
        "\n"
        "*boşluk doldurma\n"
        "2) Güneş sistemimizde 'Kızıl Gezegen' olarak bilinen gezegen ............'tır. (5p)\n"
        "\n"
        "1-) Fotosentez süreci temel olarak nasıl işler ve bu sürecin canlı yaşamı için önemi nedir? (10p)\n"
        "Cevap: bitkiler güneş ışığını kullanarak su ve CO2'den glikoz ve oksijen üretir.\n"
        "\n"
        "2-) Sera etkisi nedir ve dünya sıcaklığı üzerindeki temel işlevi nasıl gerçekleşir? (10p)\n"
        "Cevap: atmosferdeki sera gazları yeryüzünden yansıyan ısıyı tutar.\n"
        "\n"
        "3-) Enflasyon kavramını en temel ekonomik ifadesiyle nasıl tanımlarsınız? (10p)\n"
        "Cevap: para biriminin alım gücünün zamanla azalmasıdır.\n"
    )
    result = parse_s3_markdown(text)

    by_id = {a.question_number: a for a in result}
    expected_ids = {"mc1", "mc2", "fb1", "fb2", "oe1", "oe2", "oe3"}
    assert set(by_id.keys()) == expected_ids, f"Beklenen {expected_ids}, alınan {set(by_id.keys())}"

    # MC tipleri
    assert by_id["mc1"].question_type == QuestionType.MULTIPLE_CHOICE
    assert by_id["mc1"].max_score == 10
    assert by_id["mc2"].question_type == QuestionType.MULTIPLE_CHOICE
    assert by_id["mc2"].max_score == 10

    # FB tipleri
    assert by_id["fb1"].question_type == QuestionType.FILL_BLANK
    assert by_id["fb1"].max_score == 5
    assert by_id["fb2"].question_type == QuestionType.FILL_BLANK
    assert by_id["fb2"].max_score == 5

    # OE tipleri
    assert by_id["oe1"].question_type == QuestionType.OPEN_ENDED
    assert by_id["oe1"].max_score == 10
    assert by_id["oe2"].question_type == QuestionType.OPEN_ENDED
    assert by_id["oe3"].question_type == QuestionType.OPEN_ENDED


# --- Eski test'lerin yeni format'a uyarlamaları ---


def test_parse_empty_input_returns_empty_list() -> None:
    assert parse_s3_markdown("") == []
    assert parse_s3_markdown("Sadece düz metin, soru numarası yok.") == []


def test_parse_filters_hallucination_lines() -> None:
    text = (
        "*çoktan seçmeli soru\n"
        "1) Soru metni (10p)\n"
        "Yanlış cevap: B\n"
        "Not: Doğru C olmalıydı.\n"
    )
    result = parse_s3_markdown(text)
    body = result[0].extracted_answer
    assert "Yanlış cevap" not in body
    assert "Not:" not in body


def test_parse_skips_empty_answer_marker() -> None:
    text = (
        "*çoktan seçmeli soru\n"
        "1) ... (10p)\n"
        "C\n"
        "*çoktan seçmeli soru\n"
        "2) ... (10p)\n"
        "(boş)\n"
    )
    result = parse_s3_markdown(text)
    ids = [a.question_number for a in result]
    # mc2 boş cevap → atlanır
    assert ids == ["mc1"]


def test_parse_markdown_h3_headers() -> None:
    """### 1) gibi Markdown başlıklar hâlâ çalışır."""
    text = (
        "### 1) Açık uçlu soru (10p)\n"
        "Cevap: lorem ipsum\n"
    )
    result = parse_s3_markdown(text)
    assert result[0].question_number == "oe1"


# --- VLM OCR yazım varyantları (canlı testte gözlemlendi 2026-05-23) ---


def test_classify_section_handles_ocr_typo_secimli() -> None:
    """VLM 'çoktan seçmeli' yerine 'çoktan seçimli' yazabilir."""
    assert _classify_section_text("çoktan seçimli soru") == QuestionType.MULTIPLE_CHOICE


def test_classify_section_handles_ocr_typo_bosluktan() -> None:
    """VLM 'boşluk doldurma' yerine 'boşluktan doldurma' yazabilir."""
    assert _classify_section_text("boşluktan doldurma") == QuestionType.FILL_BLANK


def test_classify_section_handles_ocr_typo_uclulu() -> None:
    """VLM 'açık uçlu' yerine 'açık uçlulu' yazabilir."""
    assert _classify_section_text("Açık uçlulu") == QuestionType.OPEN_ENDED


def test_parse_real_ahmet_yesevi_ocr_output() -> None:
    """Gerçek VLM ham çıkışı (Ahmet Yesevi kâğıdı, 2026-05-23 canlı test).

    Yazım hataları içeriyor: 'seçimli', 'boşluktan', 'uçlulu'.
    Parser yine de doğru tip ataması yapmalı.
    """
    text = (
        "### Çözümler:\n"
        "\n"
        "*çoktan seçimli soru\n"
        "**1)** Türkiye'nin coğrafi konumu... (10 p)\n"
        "Cevap: 90 Dakika\n"
        "\n"
        "*boşluktan doldurma\n"
        "**1)** Dinova'nın en büyük... ülke?\n"
        "Cevap: Arjantin\n"
        "\n"
        "*çoktan seçimli soru\n"
        "**2)** Güneş sistemi... 'Kızıl Gezegen'?\n"
        "Cevaptır: Merkür\n"
        "\n"
        "*Açık uçlulu\n"
        "**1**) Fotosentez süreci... (10 p)\n"
        "Cevap: İki ileriye\n"
        "\n"
        "*Açık uçlulu\n"
        "**2**) Sera efekti... (10p)\n"
        "Ceva: Sıcak ileser\n"
        "\n"
        "*Açık uçlulu\n"
        "**3**) Enflasyon ne demek... (10p)?\n"
        "Cevaptir: Birimiyoforu\n"
    )
    result = parse_s3_markdown(text)
    by_id = {a.question_number: a for a in result}

    # 2 MC + 1 FB + 3 OE bekleniyor
    expected_ids = {"mc1", "mc2", "fb1", "oe1", "oe2", "oe3"}
    assert set(by_id.keys()) == expected_ids, f"Beklenen {expected_ids}, alınan {set(by_id.keys())}"

    assert by_id["mc1"].question_type == QuestionType.MULTIPLE_CHOICE
    assert by_id["mc2"].question_type == QuestionType.MULTIPLE_CHOICE
    assert by_id["fb1"].question_type == QuestionType.FILL_BLANK
    assert by_id["oe1"].question_type == QuestionType.OPEN_ENDED
    assert by_id["oe2"].question_type == QuestionType.OPEN_ENDED
    assert by_id["oe3"].question_type == QuestionType.OPEN_ENDED

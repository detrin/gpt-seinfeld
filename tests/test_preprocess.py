from __future__ import annotations

from scripts.preprocess import (
    build_example,
    extract_keywords,
    normalize_scene_tag,
    parse_script,
    strip_parentheticals,
)


def test_strip_parentheticals_removes_parens():
    assert strip_parentheticals("(to George) You know what?") == "You know what?"


def test_strip_parentheticals_no_parens():
    assert strip_parentheticals("I don't think so.") == "I don't think so."


def test_normalize_scene_tag_simple():
    assert normalize_scene_tag("[JERRY'S APARTMENT]") == "[JERRY'S APARTMENT]"


def test_normalize_scene_tag_strips_description():
    assert (
        normalize_scene_tag("[JERRY'S APARTMENT. JERRY AND GEORGE TALKING.]")
        == "[JERRY'S APARTMENT]"
    )


def test_normalize_scene_tag_returns_none_for_non_tag():
    assert normalize_scene_tag("JERRY: Hello.") is None


def test_parse_script_filters_non_main_characters():
    text = (
        "[MONK'S DINER]\nJERRY: You know what I love?\nNEWMAN: Stamps.\nGEORGE: What do you love?\n"
    )
    result = parse_script(text, "TheEpisode")
    assert len(result) == 1
    _, lines = result[0]
    assert any(line.startswith("JERRY:") for line in lines)
    assert any(line.startswith("GEORGE:") for line in lines)
    assert not any(line.startswith("NEWMAN:") for line in lines)


def test_parse_script_strips_parentheticals_from_dialogue():
    text = (
        "[JERRY'S APARTMENT]\nJERRY: (laughing) That's insane."
        "\nELAINE: (entering) What's going on?\nGEORGE: Unbelievable.\n"
    )
    _, lines = parse_script(text, "TheEpisode")[0]
    assert "JERRY: That's insane." in lines
    assert "ELAINE: What's going on?" in lines


def test_build_example_has_correct_structure():
    result = build_example(
        "[MONK'S DINER]",
        ["JERRY: Hello.", "GEORGE: Hi.", "ELAINE: Hey."],
        "TheDiner",
        "jerry george diner coffee parking tickets",
    )
    assert result.startswith("TOPIC:")
    assert "[MONK'S DINER]" in result
    assert "JERRY: Hello." in result
    assert result.strip().endswith("[END]")


def test_extract_keywords_excludes_stopwords():
    text = "parking spot buick circling driving around block"
    kw = extract_keywords(text, n=3)
    assert "parking" in kw
    assert "buick" in kw
    assert "the" not in kw
    assert "around" not in kw


def test_normalize_scene_tag_rejects_stage_direction():
    assert normalize_scene_tag("[pause]") is None
    assert normalize_scene_tag("[Continued]") is None


def test_extract_keywords_excludes_character_names():
    text = "jerry george elaine kramer parking spot buick"
    kw = extract_keywords(text, n=3)
    assert "jerry" not in kw
    assert "parking" in kw

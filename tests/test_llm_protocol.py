import pytest

from transparency_sim.llm_blind_id import (
    AnswerCmd,
    FetchCmd,
    ListCmd,
    ProtocolViolation,
    ResolveCmd,
    parse_command,
    sanitize_answer,
)


def test_parse_fetch_and_resolve_and_list():
    assert parse_command("LIST") == ListCmd()
    assert parse_command("FETCH DOC_ABCD") == FetchCmd("DOC_ABCD")
    assert parse_command("RESOLVE DOC_ABCD DOC_1234") == ResolveCmd("DOC_ABCD", "DOC_1234")


def test_parse_uses_final_nonempty_line():
    text = "I will inspect one id first.\n\nFETCH DOC_ABCD\n"

    assert parse_command(text) == FetchCmd("DOC_ABCD")


def test_parse_rejects_bad_arity_and_bad_ids():
    assert isinstance(parse_command("FETCH"), ProtocolViolation)
    assert isinstance(parse_command("FETCH BAD_1234"), ProtocolViolation)
    assert isinstance(parse_command("RESOLVE DOC_ABCD"), ProtocolViolation)
    assert isinstance(parse_command("RESOLVE DOC_ABCD BAD_1234"), ProtocolViolation)


def test_parse_answer_valid_json_object():
    parsed = parse_command('ANSWER {"component_1": "value"}')

    assert parsed == AnswerCmd({"component_1": "value"})


def test_parse_answer_invalid_json_is_violation():
    assert isinstance(parse_command("ANSWER {"), ProtocolViolation)
    assert isinstance(parse_command('ANSWER ["not", "object"]'), ProtocolViolation)


def test_sanitize_drops_unknown_keys_and_counts():
    answer, dropped = sanitize_answer(
        {"component_1": "a", "component_9": "b"},
        {"component_1", "component_2"},
    )

    assert answer == {"component_1": "a"}
    assert dropped == 1


def test_sanitize_drops_non_string_values():
    answer, dropped = sanitize_answer(
        {"component_1": 3, "component_2": None, "component_3": "ok"},
        {"component_1", "component_2", "component_3"},
    )

    assert answer == {"component_3": "ok"}
    assert dropped == 2

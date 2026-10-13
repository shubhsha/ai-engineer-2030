import json
import logging
import sys

from ai_engineer.foundations.logs import JsonFormatter


def record(**fields):
    return logging.makeLogRecord(
        {"name": "aie.llm", "levelname": "INFO", **fields}
    )


def test_extra_fields_become_keys_you_can_query():
    line = JsonFormatter().format(
        record(msg="model call", model="haiku", input_tokens=200)
    )
    entry = json.loads(line)
    assert entry["message"] == "model call"
    assert (entry["level"], entry["logger"]) == ("INFO", "aie.llm")
    assert (entry["model"], entry["input_tokens"]) == ("haiku", 200)
    assert entry["time"].endswith("+00:00")  # UTC, with its offset


def test_arguments_are_formatted_into_the_message():
    entry = json.loads(
        JsonFormatter().format(record(msg="loaded %d rows", args=(3,)))
    )
    assert entry["message"] == "loaded 3 rows"
    assert "args" not in entry
    assert "msg" not in entry


def test_a_traceback_stays_on_one_line():
    try:
        raise ValueError("bad row 7")
    except ValueError:
        line = JsonFormatter().format(
            record(msg="load failed", exc_info=sys.exc_info())
        )
    assert "\n" not in line
    assert "ValueError: bad row 7" in json.loads(line)["error"]

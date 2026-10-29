"""Tests for the append-only manual decision log (AC11, AC12).

Covers append-only history, required justification/actor/entity, timestamp
recording, entity linking, the technical-state vs exploratory-label
independence guard, and structured export.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prebenchmark.decisions import (
    DECISION_TYPES,
    EXPLORATORY_LABELS,
    TECHNICAL_STATES,
    decision_to_record,
    decisions_to_records,
    is_exploratory_label,
    is_technical_state,
    record_decision,
)
from prebenchmark.store import DECISIONS_JSONL_FILE, ResultsStore

EXPERIMENT_ID = "exp-dec-0001"
RUN_ID = "20260101T000000Z-abcdef"


@pytest.fixture
def store(tmp_path: Path) -> ResultsStore:
    return ResultsStore(tmp_path / "results")


def raw_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record(store: ResultsStore, **overrides):
    payload = {
        "decision_type": "keep",
        "justification": "stable across repetitions",
        "actor": "analyst",
        "run_id": RUN_ID,
    }
    payload.update(overrides)
    return record_decision(store, EXPERIMENT_ID, **payload)


# ---------------------------------------------------------------------------
# Recording basics
# ---------------------------------------------------------------------------


def test_record_decision_appends_and_returns_record(store: ResultsStore) -> None:
    decision = record(store)
    assert decision["decision_id"].startswith("dec-")
    assert decision["entity"] == {"run_id": RUN_ID}
    assert decision["decision_type"] == "keep"
    assert decision["label"] is None
    assert decision["timestamp"]
    assert store.load_decisions(EXPERIMENT_ID) == [decision]


def test_record_decision_requires_justification(store: ResultsStore) -> None:
    with pytest.raises(ValueError, match="justification"):
        record(store, justification="   ")


def test_record_decision_requires_actor(store: ResultsStore) -> None:
    with pytest.raises(ValueError, match="actor"):
        record(store, actor="")


def test_record_decision_requires_entity_reference(store: ResultsStore) -> None:
    with pytest.raises(ValueError, match="run_id, config_id or group_id"):
        record_decision(
            store,
            EXPERIMENT_ID,
            decision_type="keep",
            justification="no entity here",
            actor="analyst",
        )


def test_record_decision_rejects_unknown_type(store: ResultsStore) -> None:
    with pytest.raises(ValueError, match="decision_type"):
        record(store, decision_type="banana")


def test_record_decision_records_custom_timestamp(store: ResultsStore) -> None:
    decision = record(store, timestamp="2026-03-03T09:00:00Z")
    assert decision["timestamp"] == "2026-03-03T09:00:00Z"


def test_record_decision_links_run_and_config(store: ResultsStore) -> None:
    decision = record(store, run_id=RUN_ID, config_id="cfg-v1-abc123abc123")
    assert decision["entity"] == {"run_id": RUN_ID, "config_id": "cfg-v1-abc123abc123"}


# ---------------------------------------------------------------------------
# Append-only history
# ---------------------------------------------------------------------------


def test_decision_history_is_append_only_and_ordered(store: ResultsStore) -> None:
    first = record(store, decision_type="keep", justification="looks stable")
    second = record(store, decision_type="repeat", justification="needs more seeds")

    lines = raw_lines(store.experiment_dir(EXPERIMENT_ID) / DECISIONS_JSONL_FILE)
    assert len(lines) == 2

    history = store.load_decisions(EXPERIMENT_ID, run_id=RUN_ID)
    assert [decision["decision_type"] for decision in history] == ["keep", "repeat"]
    # the earlier entry is untouched
    assert history[0]["decision_id"] == first["decision_id"]
    assert history[0]["justification"] == "looks stable"
    assert history[1]["decision_id"] == second["decision_id"]


# ---------------------------------------------------------------------------
# Technical state vs exploratory label independence
# ---------------------------------------------------------------------------


def test_label_must_be_exploratory_not_technical_state(store: ResultsStore) -> None:
    with pytest.raises(ValueError, match="technical execution state"):
        record(store, decision_type="label", label="COMPLETED")
    with pytest.raises(ValueError, match="technical execution state"):
        record(store, decision_type="label", label="FAILED")


def test_unknown_label_is_rejected(store: ResultsStore) -> None:
    with pytest.raises(ValueError, match="unknown exploratory label"):
        record(store, decision_type="label", label="MAYBE")
    # a real exploratory label is accepted
    decision = record(store, decision_type="label", label="NON_VIABLE")
    assert decision["label"] == "NON_VIABLE"


def test_state_and_label_are_independent(store: ResultsStore) -> None:
    # A run is technically COMPLETED.
    store.persist_run(EXPERIMENT_ID, RUN_ID, {"state": "COMPLETED"})

    # Recording an exploratory UNSTABLE label must not touch the execution state.
    record(
        store,
        decision_type="label",
        justification="loss oscillated across epochs",
        actor="analyst",
        label="UNSTABLE",
    )
    record(
        store,
        decision_type="label",
        justification="trade-off worth keeping",
        actor="analyst",
        label="INTERESTING",
    )

    loaded = store.load_run(RUN_ID)
    assert loaded["metadata"]["state"] == "COMPLETED"  # unchanged

    history = store.load_decisions(EXPERIMENT_ID, run_id=RUN_ID)
    assert [decision["label"] for decision in history] == ["UNSTABLE", "INTERESTING"]
    # no decision record smuggles an execution state into the label field
    for decision in history:
        assert decision["label"] not in TECHNICAL_STATES or decision["label"] == "UNSTABLE"


def test_vocabulary_helpers_and_separation() -> None:
    assert is_technical_state("COMPLETED")
    assert is_technical_state("FAILED")
    assert not is_technical_state("VIABLE")
    assert is_exploratory_label("NON_VIABLE")
    assert is_exploratory_label("PENDING_REVIEW")
    assert not is_exploratory_label("FAILED")
    # only the word UNSTABLE is shared between the two vocabularies
    assert set(TECHNICAL_STATES) & set(EXPLORATORY_LABELS) == {"UNSTABLE"}
    assert {"keep", "exclude", "repeat", "investigate"} <= set(DECISION_TYPES)


# ---------------------------------------------------------------------------
# Manual-only (no automatic classification) and validation decisions
# ---------------------------------------------------------------------------


def test_no_label_is_inferred_when_omitted(store: ResultsStore) -> None:
    # Manual only: a decision without an explicit label carries none -- nothing
    # is classified automatically from the run's metrics.
    decision = record(store, decision_type="investigate", label=None)
    assert decision["label"] is None


def test_validation_decision_records_rationale(store: ResultsStore) -> None:
    decision = record(
        store,
        decision_type="validate",
        justification="Adam with lr=0.1 is WARNING: may be unstable",
        actor="analyst",
        run_id=None,
        config_id="cfg-v1-abc123abc123",
        evidence={"validation_state": "WARNING", "rationale": ["Adam with lr >= 0.1 may be unstable"]},
    )
    assert decision["decision_type"] == "validate"
    assert decision["label"] is None
    assert decision["entity"] == {"config_id": "cfg-v1-abc123abc123"}
    assert decision["evidence"]["validation_state"] == "WARNING"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_decision_to_record_flattens_entity(store: ResultsStore) -> None:
    decision = record(store, run_id=RUN_ID, config_id="cfg-v1-abc123abc123", label="VIABLE")
    row = decision_to_record(decision)
    assert row["run_id"] == RUN_ID
    assert row["config_id"] == "cfg-v1-abc123abc123"
    assert row["group_id"] == ""
    assert row["label"] == "VIABLE"
    assert row["actor"] == "analyst"

    assert len(decisions_to_records([decision])) == 1
    assert len(decisions_to_records(decision)) == 1


def test_decision_history_retrievable_by_entity(store: ResultsStore) -> None:
    record(store, run_id=RUN_ID)
    record(store, run_id="other-run")
    assert len(store.load_decisions(EXPERIMENT_ID, run_id=RUN_ID)) == 1
    assert len(store.load_decisions(EXPERIMENT_ID, run_id="other-run")) == 1

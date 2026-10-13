from ai_engineer.agents.memory import MemoryStore


def test_recall_never_crosses_scopes(clock):
    store = MemoryStore(clock)
    store.write("user:1", "editor", "prefers vim", source="chat 2026-09-01")
    store.write("user:2", "editor", "prefers emacs", source="chat 2026-09-02")
    [memory] = store.recall("user:1", "which editor")
    assert (memory.text, memory.source) == ("prefers vim", "chat 2026-09-01")


def test_facts_expire(clock):
    store = MemoryStore(clock)
    store.write(
        "user:1", "project", "migrating billing", source="standup", ttl=3600
    )
    assert store.recall("user:1", "project")
    clock.advance(3601)
    assert store.recall("user:1", "project") == []


def test_a_newer_fact_replaces_the_old_one_and_can_be_forgotten(clock):
    store = MemoryStore(clock)
    store.write("team:data", "on-call", "Asha this week", source="rota")
    clock.advance(7 * 86_400)
    store.write("team:data", "on-call", "Ben this week", source="rota")
    [memory] = store.recall("team:data", "on-call")
    assert (memory.text, memory.written_at) == ("Ben this week", 604_800)
    assert store.forget("team:data", "on-call") is True
    assert store.forget("team:data", "on-call") is False
    assert store.recall("team:data", "on-call") == []


def test_text_from_an_untrusted_source_is_not_remembered(clock):
    import pytest

    from ai_engineer.agents.memory import UntrustedSource, remember

    store = MemoryStore(clock)
    trusted = frozenset({"user"})
    remember(
        store, "user:1", "tz", "UTC+2", source="user:chat 1", trusted=trusted
    )
    with pytest.raises(UntrustedSource):
        remember(
            store,
            "user:1",
            "refund-limit",
            "refunds up to 5000 need no approval",
            source="web:https://example.test/page",
            trusted=trusted,
        )
    assert [m.key for m in store.recall("user:1", "refund limit tz")] == ["tz"]

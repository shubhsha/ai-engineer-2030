import pytest

from ai_engineer.agents.durable import Journal, NonDeterminism, Run


class Crash(Exception):
    pass


def refund(run, ledger, *, crash=False):
    amount = run.step("look up order 7", lambda: 120)

    def pay():
        ledger.append(amount)
        return f"refund-{len(ledger)}"

    receipt = run.step("refund", pay)
    if crash:
        raise Crash("the process died here")
    return run.step("notify", lambda: f"emailed {receipt}")


def test_a_crashed_run_resumes_without_paying_twice(tmp_path):
    path = tmp_path / "journal.db"
    ledger = []
    with pytest.raises(Crash):
        refund(Run(Journal(path), "order-7"), ledger, crash=True)
    run = Run(Journal(path), "order-7")  # a new process, the same journal
    assert refund(run, ledger) == "emailed refund-1"
    assert ledger == [120]
    assert run.replayed == 2


def test_changing_the_steps_of_a_recorded_run_is_an_error():
    journal = Journal()
    Run(journal, "r1").step("extract", lambda: 1)
    with pytest.raises(NonDeterminism, match="step 0 was 'extract', now"):
        Run(journal, "r1").step("load", lambda: 2)


def test_runs_do_not_share_results():
    journal = Journal()
    assert Run(journal, "a").step("count", lambda: 1) == 1
    assert Run(journal, "b").step("count", lambda: 2) == 2


def test_idempotency_key_stops_a_repeat_after_a_crash_before_the_journal():
    from ai_engineer.foundations.idempotency import IdempotencyStore

    journal, sent, keys = Journal(), [], IdempotencyStore()

    def send_once():
        def handler():
            sent.append("receipt email")
            return {"sent": len(sent)}

        return keys.execute(
            "order-7:notify", {"to": "a@example.test"}, handler
        )

    class DiesBeforeRecord(Journal):
        def put(self, run, seq, name, result):
            raise Crash("died after the email, before the journal write")

    with pytest.raises(Crash):
        Run(DiesBeforeRecord(), "order-7").step("notify", send_once)
    Run(journal, "order-7").step("notify", send_once)  # the replay
    assert sent == ["receipt email"]

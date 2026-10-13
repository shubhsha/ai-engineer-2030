"""Chapter 33: workflow patterns to reach for before an agent.

The five patterns of Anthropic's "Building effective agents" (2024):
prompt chaining, routing, parallelization, orchestrator-workers and
evaluator-optimizer. Each is a few lines of ordinary code around model
calls, which is the point: the control flow stays in your code, where
you can read it, test it and bound it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

from ai_engineer.llm.model import Model, ask
from ai_engineer.llm.prompt import tagged


class GateFailed(RuntimeError):
    def __init__(self, step: int, output: str) -> None:
        super().__init__(f"gate rejected the output of step {step}")
        self.step, self.output = step, output


def text(model: Model, prompt: str) -> str:
    return ask(model, prompt).text.strip()


def chain(
    model: Model,
    steps: Sequence[str],
    data: str,
    *,
    gate: Callable[[int, str], bool] | None = None,
) -> str:
    """Prompt chaining: each step's output is the next step's input.
    A gate checks the work between steps, in code."""
    for i, instruction in enumerate(steps):
        data = text(model, f"{instruction}\n\n{tagged('input', data)}")
        if gate is not None and not gate(i, data):
            raise GateFailed(i, data)
    return data


def route(
    model: Model,
    message: str,
    handlers: Mapping[str, Callable[[str], str]],
    *,
    fallback: str,
) -> tuple[str, str]:
    """Routing: one cheap classification, then a specialized handler.
    An unexpected label goes to ``fallback``, never to a guess."""
    labels = ", ".join(sorted(handlers))
    label = text(
        model,
        f"Classify the message as one of: {labels}. Reply with the label "
        f"only.\n\n{tagged('message', message)}",
    ).lower()
    label = label if label in handlers else fallback
    return label, handlers[label](message)


def parallel(
    model: Model, prompts: Sequence[str], *, limit: int = 4
) -> list[str]:
    """Parallelization: independent prompts at once, results in order."""
    with ThreadPoolExecutor(max_workers=limit) as pool:
        return list(pool.map(lambda p: text(model, p), prompts))


def vote(model: Model, prompt: str, *, n: int = 5) -> tuple[str, float]:
    """Ask ``n`` times; return the most common answer and its share."""
    answers = parallel(model, [prompt] * n, limit=n)
    answer, count = Counter(answers).most_common(1)[0]
    return answer, count / n


def orchestrate(
    planner: Model, worker: Model, task: str, *, max_parts: int = 4
) -> str:
    """Orchestrator-workers: the planner decides the parts at run time,
    workers do them in parallel, and the planner merges the results."""
    plan = text(
        planner,
        f"Split the task into at most {max_parts} independent parts, one "
        f"per line, with no numbering.\n\n{tagged('task', task)}",
    )
    parts = [line.strip() for line in plan.splitlines() if line.strip()]
    briefs = [
        f"{tagged('task', task)}\n{tagged('your_part', part)}"
        for part in parts[:max_parts]
    ]
    results = "\n".join(tagged("part", r) for r in parallel(worker, briefs))
    return text(
        planner,
        f"Merge the parts into one answer.\n{tagged('task', task)}\n{results}",
    )


def refine(
    generator: Model, evaluator: Model, task: str, *, max_rounds: int = 3
) -> tuple[str, int]:
    """Evaluator-optimizer: draft, critique, revise, until the evaluator
    replies PASS or the rounds run out. Returns the draft and rounds."""
    draft = text(generator, task)
    for round_ in range(1, max_rounds + 1):
        verdict = text(
            evaluator,
            "Reply PASS if the draft fully meets the task. Otherwise list "
            f"what to fix.\n{tagged('task', task)}\n{tagged('draft', draft)}",
        )
        if verdict.upper().startswith("PASS"):
            return draft, round_
        draft = text(
            generator,
            f"Revise the draft.\n{tagged('task', task)}\n"
            f"{tagged('draft', draft)}\n{tagged('feedback', verdict)}",
        )
    return draft, max_rounds

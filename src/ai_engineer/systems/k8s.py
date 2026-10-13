"""Kubernetes arithmetic for AI workloads (chapter 56).

A manifest linter over plain dicts, the Horizontal Pod Autoscaler rule,
and a bin-packing scheduler with first-fit and first-fit decreasing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

HOURS_PER_MONTH = 730
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


# region lint
@dataclass(frozen=True, slots=True)
class Finding:
    """One problem found in a manifest."""

    rule: str
    container: str
    message: str


def _containers(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    spec = manifest.get("spec", {})
    if not isinstance(spec, dict):
        return []
    template = spec.get("template")
    if isinstance(template, dict):
        spec = template.get("spec", {})
    found = spec.get("containers", []) if isinstance(spec, dict) else []
    return [c for c in found if isinstance(c, dict)]


def _tag(image: str) -> str:
    last = image.rsplit("/", 1)[-1]
    if "@" in last:
        return "pinned"
    return last.split(":", 1)[1] if ":" in last else ""


def lint_manifest(manifest: Mapping[str, object]) -> list[Finding]:
    """Check a Pod or Deployment dict against five review rules."""
    out: list[Finding] = []
    for c in _containers(manifest):
        name = str(c.get("name", "?"))
        limits = c.get("resources", {})
        if not (isinstance(limits, dict) and limits.get("limits")):
            out.append(Finding("K001", name, "no resource limits"))
        if "livenessProbe" not in c or "readinessProbe" not in c:
            out.append(Finding("K002", name, "missing liveness or readiness"))
        if _tag(str(c.get("image", ""))) in ("", "latest"):
            out.append(Finding("K003", name, "image tag is latest or absent"))
        ctx = c.get("securityContext", {})
        safe = isinstance(ctx, dict) and (
            ctx.get("runAsNonRoot") is True
            or int(ctx.get("runAsUser", 0) or 0) > 0
        )
        if not safe:
            out.append(Finding("K004", name, "may run as root"))
        for env in _env(c):
            if _literal_secret(env):
                label = f"secret literal {env['name']}"
                out.append(Finding("K005", name, label))
    return out


def _env(c: Mapping[str, object]) -> list[dict[str, object]]:
    env = c.get("env", [])
    if not isinstance(env, list):
        return []
    return [e for e in env if isinstance(e, dict)]


def _literal_secret(env: object) -> bool:
    if not isinstance(env, dict) or "value" not in env:
        return False
    return any(w in str(env.get("name", "")).upper() for w in SECRET_WORDS)


# endregion


# region hpa
def hpa_desired(
    current: int, metric: float, target: float, tolerance: float = 0.1
) -> int:
    """The HPA rule: ceil(current * metric / target), unless close enough."""
    ratio = metric / target
    if abs(ratio - 1.0) <= tolerance:
        return current
    return math.ceil(current * ratio)


# endregion


# region packing
@dataclass(frozen=True, slots=True)
class Pod:
    """A resource request: millicores, MiB of memory, whole GPUs."""

    name: str
    cpu: int
    memory: int
    gpu: int = 0


@dataclass(frozen=True, slots=True)
class Node:
    """A machine of some kind with fixed capacity."""

    name: str
    kind: str
    cpu: int
    memory: int
    gpu: int = 0


@dataclass(frozen=True, slots=True)
class Packing:
    """Where pods went, which nodes were opened, what was left over."""

    placements: tuple[tuple[str, str], ...]
    used: tuple[Node, ...]
    unplaced: tuple[Pod, ...]
    stranded_gpu: int
    stranded_cpu: int


def pack(pods: Sequence[Pod], nodes: Sequence[Node], strategy: str) -> Packing:
    """Place pods on the first node that fits.

    strategy is "first_fit" (arrival order) or "ffd" (largest first, by
    gpu, then cpu, then memory). A node counts as used once it holds a pod.
    """
    if strategy == "ffd":
        pods = sorted(
            pods, key=lambda p: (p.gpu, p.cpu, p.memory), reverse=True
        )
    elif strategy != "first_fit":
        raise ValueError(f"unknown strategy: {strategy}")
    free = {n.name: [n.cpu, n.memory, n.gpu] for n in nodes}
    opened: dict[str, Node] = {}
    placed: list[tuple[str, str]] = []
    unplaced: list[Pod] = []
    for pod in pods:
        for node in nodes:
            f = free[node.name]
            if f[0] >= pod.cpu and f[1] >= pod.memory and f[2] >= pod.gpu:
                f[0] -= pod.cpu
                f[1] -= pod.memory
                f[2] -= pod.gpu
                opened[node.name] = node
                placed.append((pod.name, node.name))
                break
        else:
            unplaced.append(pod)
    used = tuple(n for n in nodes if n.name in opened)
    return Packing(
        tuple(placed),
        used,
        tuple(unplaced),
        sum(free[n.name][2] for n in used),
        sum(free[n.name][0] for n in used),
    )


def monthly_cost(
    nodes: Sequence[Node], prices: Mapping[str, Decimal]
) -> Decimal:
    """Hourly price per node kind times 730 hours, summed."""
    return sum((prices[n.kind] for n in nodes), Decimal(0)) * HOURS_PER_MONTH


# endregion

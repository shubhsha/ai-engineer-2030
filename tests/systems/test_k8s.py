from __future__ import annotations

import random
from decimal import Decimal

import pytest

from ai_engineer.systems.k8s import (
    Node,
    Pod,
    hpa_desired,
    lint_manifest,
    monthly_cost,
    pack,
)


def deployment(**overrides: object) -> dict[str, object]:
    container: dict[str, object] = {
        "name": "api",
        "image": "registry.example/api:1.4.2",
        "resources": {"limits": {"memory": "2Gi"}},
        "livenessProbe": {"httpGet": {"path": "/healthz"}},
        "readinessProbe": {"httpGet": {"path": "/ready"}},
        "securityContext": {"runAsNonRoot": True},
    }
    container.update(overrides)
    return {
        "kind": "Deployment",
        "spec": {"template": {"spec": {"containers": [container]}}},
    }


# region test_lint
def test_clean_manifest_has_no_findings() -> None:
    assert lint_manifest(deployment()) == []


def test_bare_container_trips_every_rule() -> None:
    bare = {
        "spec": {
            "containers": [
                {
                    "name": "m",
                    "image": "m:latest",
                    "env": [{"name": "API_KEY", "value": "sk-test"}],
                }
            ]
        }
    }
    rules = [f.rule for f in lint_manifest(bare)]
    assert rules == ["K001", "K002", "K003", "K004", "K005"]


# endregion


def test_untagged_and_digest_images() -> None:
    assert [f.rule for f in lint_manifest(deployment(image="m"))] == ["K003"]
    pinned = deployment(image="r.example/m@sha256:abc")
    assert lint_manifest(pinned) == []


def test_secret_from_reference_is_fine() -> None:
    env = [{"name": "API_KEY", "valueFrom": {"secretKeyRef": {}}}]
    assert lint_manifest(deployment(env=env)) == []


def test_run_as_user_zero_is_root() -> None:
    ctx = {"runAsUser": 0}
    rules = [f.rule for f in lint_manifest(deployment(securityContext=ctx))]
    assert rules == ["K004"]


# region test_hpa
def test_hpa_by_hand_case() -> None:
    assert hpa_desired(4, 90, 60) == 6


def test_hpa_tolerance_and_scale_down() -> None:
    assert hpa_desired(6, 63, 60) == 6
    assert hpa_desired(6, 20, 60) == 2
    assert hpa_desired(4, 12, 60) == 1


# endregion


def workload() -> list[Pod]:
    r = random.Random(2030)
    pods: list[Pod] = []
    for i in range(60):
        if r.random() < 0.5:
            g = r.choice([1, 2, 3, 5, 6])
            pods.append(Pod(f"gpu-{i}", 4000 * g, 32768 * g, g))
        else:
            cpu = r.choice([500, 1000, 2000, 4000, 6000])
            mem = r.choice([1024, 4096, 8192, 16384])
            pods.append(Pod(f"cpu-{i}", cpu, mem))
    return pods


# region test_pack
def test_ffd_beats_first_fit_by_hand_case() -> None:
    sizes = [1, 1, 1, 6, 6]
    pods = [Pod(f"p{i}", 1000, 1024, g) for i, g in enumerate(sizes)]
    nodes = [Node(f"n{i}", "gpu", 64000, 65536, 8) for i in range(5)]
    assert len(pack(pods, nodes, "first_fit").used) == 3
    assert len(pack(pods, nodes, "ffd").used) == 2


def test_seeded_workload_saves_a_gpu_node() -> None:
    nodes = [Node(f"g{i}", "gpu", 64000, 524288, 8) for i in range(30)]
    nodes += [Node(f"c{i}", "cpu", 16000, 65536) for i in range(30)]
    naive = pack(workload(), nodes, "first_fit")
    better = pack(workload(), nodes, "ffd")
    assert (len(naive.used), naive.stranded_gpu) == (15, 8)
    assert (len(better.used), better.stranded_gpu) == (14, 0)
    prices = {"cpu": Decimal("0.80"), "gpu": Decimal("30.00")}
    saved = monthly_cost(naive.used, prices) - monthly_cost(
        better.used, prices
    )
    assert saved == Decimal("21900")


# endregion


def test_unplaced_and_bad_strategy() -> None:
    big = Pod("big", 1000, 1024, 16)
    node = Node("n", "gpu", 64000, 65536, 8)
    result = pack([big], [node], "ffd")
    assert result.unplaced == (big,)
    assert result.used == ()
    with pytest.raises(ValueError, match="unknown strategy"):
        pack([big], [node], "best")

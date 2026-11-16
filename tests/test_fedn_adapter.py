"""The FEDn server functions compute exactly what telltale's detector does.

The class is loaded the way FEDn loads it: the class source alone, run with
only the names FEDn's import allowlist provides. So this also checks that
nothing outside the allowlist is needed. FEDn itself is not required.
"""

import ast
import json
from pathlib import Path

import numpy as np

from telltale.detect import cusum, level_series
from telltale.integrations.wire import decode_scores, encode_scores

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "fedn"
    / "server_functions.py"
)
R, K, P, C = 30, 6, 50, 10


def load_class_source_only():
    src = SOURCE.read_text()
    tree = ast.parse(src)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ServerFunctions"
    )
    logged = []
    scope = {
        "np": np,
        "print": logged.append,
        "Dict": dict,
        "List": list,
        "Tuple": tuple,
        "ServerFunctionsBase": object,
    }
    ns = {}
    code = compile(ast.get_source_segment(src, cls), "sf", "exec")
    exec(code, scope, ns)  # noqa: S102
    return ns["ServerFunctions"], logged


def fleet_with_one_drifting_client():
    rng = np.random.default_rng(7)
    base = rng.dirichlet(np.ones(C) * 0.3, size=P)
    probs = np.empty((R, K, P, C), dtype=np.float32)
    for r in range(R):
        for k in range(K):
            noise = rng.dirichlet(np.ones(C) * 50, size=P)
            w = 0.1 + (0.03 * (r - 17) if k == 2 and r >= 18 else 0.0)
            probs[r, k] = (1 - w) * base + w * noise
    return probs


def run_adapter(probs):
    cls, logged = load_class_source_only()
    sf = cls()
    ids = [f"client-{k}" for k in range(K)]
    params = [np.zeros((2, 2), dtype=np.float32)]
    for r in range(R):
        sf.client_settings(params)
        updates = {
            ids[k]: (
                params,
                {
                    "num_examples": 1,
                    "telltale_probe_scores": encode_scores(probs[r, k]).hex(),
                    "telltale_shape": [P, C],
                    "telltale_fingerprint": "same",
                },
            )
            for k in range(K)
        }
        sf.aggregate(params, updates)
    return cls, sf, ids, logged


def test_matches_telltale_detector():
    probs = fleet_with_one_drifting_client()
    # The server sees float16 on the wire.
    wire = np.stack(
        [
            [decode_scores(encode_scores(probs[r, k]), P, C) for k in range(K)]
            for r in range(R)
        ]
    )
    cls, sf, ids, logged = run_adapter(probs)
    assert not [m for m in logged if m.startswith("TELLTALE_ERROR")]

    ref_level = level_series(wire)["level"]
    ref_cusum = cusum(ref_level, slice(cls.REF_START, cls.REF_STOP))

    got_level = np.array(
        [[sf.levels[ids[k]][r] for k in range(K)] for r in range(R)]
    )
    np.testing.assert_allclose(got_level, ref_level, rtol=0, atol=1e-12)

    records = [
        json.loads(m[len("TELLTALE ") :])
        for m in logged
        if m.startswith("TELLTALE {")
    ]
    assert len(records) == R
    for rec in records:
        r = rec["round"]
        for k in range(K):
            got = rec["cusum"].get(ids[k])
            if np.isnan(ref_cusum[r, k]):
                assert got is None
            else:
                assert abs(got - ref_cusum[r, k]) < 1e-9
    # The drifting client ends far above the others.
    assert np.nanargmax(ref_cusum[-1]) == 2


def test_aggregate_is_weighted_fedavg_and_ignores_missing_scores():
    cls, logged = load_class_source_only()
    sf = cls()
    sf.client_settings([np.zeros(2)])
    out = sf.aggregate(
        [np.zeros(2)],
        {
            "a": ([np.array([1.0, 1.0])], {"num_examples": 1}),
            "b": ([np.array([4.0, 4.0])], {"num_examples": 3}),
        },
    )
    np.testing.assert_allclose(out[0], [3.25, 3.25])
    rec = json.loads(logged[-1][len("TELLTALE ") :])
    assert rec["skipped"] == ["a", "b"] and rec["n_scored"] == 0

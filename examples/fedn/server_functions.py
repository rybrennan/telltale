"""telltale as FEDn server functions.

A per-client drift monitor that runs inside ``aggregate``. Each client scores a
small fixed probe set with its locally trained model and puts the class
probabilities in its training metadata. The monitor judges clients only on
those answers, never on their model updates. It compares each client's answers
with the other clients' and accumulates the excess over that client's own
early rounds.

It flags only. Aggregation is plain weighted FedAvg over every client.

Everything is inside the class because the platform ships the class source
alone. Only names from ``allowed_import`` are used (numpy and the logger), so
there is no ``json`` and no ``from __future__`` import.

The statistics are the ones telltale uses everywhere else
(``telltale.detect.level_series`` and ``telltale.detect.cusum``):

    level   mean Jensen-Shannon divergence between a client's probe answers
            and the leave-one-out mean of the other clients, per round
    CUSUM   S_t = max(0, S_{t-1} + (level_t - mu_ref) / sd_ref - K)
            where mu_ref and sd_ref are the client's own level over the
            reference rounds [REF_START, REF_STOP)

Client metadata this expects (see ``client_train.py``):

    telltale_probe_scores   hex of float16 bytes, (n_probes, n_classes)
    telltale_shape          [n_probes, n_classes]
    telltale_fingerprint    hash of the probe set the client scored
    telltale_site           optional label, copied into the log
"""

from fedn.network.combiner.hooks.allowed_import import (
    Dict,
    List,
    ServerFunctionsBase,
    Tuple,
    np,
    print,
)


class ServerFunctions(ServerFunctionsBase):
    """Weighted FedAvg with telltale's per-client monitor in front.

    Each round writes one ``TELLTALE {...}`` line to the hook log with every
    client's level and CUSUM, and a ``TELLTALE_FLAG`` line when a client is
    above ``THRESHOLD``.
    """

    # First and one-past-last round of the reference epoch.
    REF_START = 5
    REF_STOP = 15
    # CUSUM allowance in SD units. 0.5 is the textbook default, not tuned.
    K = 0.5
    # A client above this in the current round is flagged. None records only.
    THRESHOLD = None
    LEARNING_RATE = 0.01
    # Test rig only: extra settings forwarded to clients, used to tell one
    # client to corrupt its own data. The detector never reads this.
    EXPERIMENT_SETTINGS = {}
    # Test rig only: makes each run's source unique so state starts fresh.
    RUN_TAG = ""

    def __init__(self) -> None:
        self.round = -1
        self.levels = {}  # client_id -> {round: level}
        self.cusum = {}  # client_id -> running sum
        self.ever_flagged = []

    def client_settings(self, global_model: List[np.ndarray]) -> dict:
        """Advance the round counter and send it to clients.

        Args:
            global_model: The current global model (unused).

        Returns:
            Learning rate, the round index, and any test-rig settings.
        """
        self.round += 1
        settings = {
            "learning_rate": self.LEARNING_RATE,
            "telltale_round": self.round,
        }
        settings.update(self.EXPERIMENT_SETTINGS)
        return settings

    def aggregate(
        self,
        previous_global: List[np.ndarray],
        client_updates: Dict[str, Tuple[List[np.ndarray], dict]],
    ) -> List[np.ndarray]:
        """Run the monitor, then return the weighted FedAvg of every update.

        Args:
            previous_global: Last round's global model.
            client_updates: Client id to (parameters, metadata).

        Returns:
            The new global model.
        """
        try:
            self._monitor(client_updates)
        except Exception as e:  # the monitor must never break aggregation
            print("TELLTALE_ERROR " + repr(e))
        return self._fedavg(previous_global, client_updates)

    # -- aggregation: weighted FedAvg, nobody excluded ------------------------
    def _fedavg(
        self,
        previous_global: list,
        client_updates: dict,
    ) -> list:
        """Average parameters weighted by each client's ``num_examples``."""
        if len(client_updates) == 0:
            return previous_global
        weighted_sum = [np.zeros_like(param) for param in previous_global]
        total_weight = 0
        for _, (client_parameters, metadata) in client_updates.items():
            num_examples = metadata.get("num_examples", 1)
            total_weight += num_examples
            for i in range(len(weighted_sum)):
                weighted_sum[i] += client_parameters[i] * num_examples
        return [weighted / total_weight for weighted in weighted_sum]

    # -- the monitor ----------------------------------------------------------
    def _monitor(self, client_updates: dict) -> None:
        """Score this round and write one record to the log."""
        r = self.round
        scores, sites, skipped = self._decode(client_updates)
        level = self._level(scores)
        cusum_now = {}
        flagged = []
        for cid in sorted(level):
            self.levels.setdefault(cid, {})[r] = level[cid]
            s = self._cusum_step(cid, r)
            if s is None:
                continue
            cusum_now[cid] = s
            if self.THRESHOLD is not None and s > self.THRESHOLD:
                flagged.append(cid)
                if cid not in self.ever_flagged:
                    self.ever_flagged.append(cid)
        record = {
            "tag": self.RUN_TAG,
            "round": r,
            "n_scored": len(scores),
            "level": level,
            "cusum": cusum_now,
            "flagged": flagged,
            "skipped": skipped,
            "site": sites,
        }
        print("TELLTALE " + self._json(record))
        if flagged:
            msg = "TELLTALE_FLAG round " + str(r) + " clients "
            print(msg + ", ".join(flagged))

    def _decode(self, client_updates: dict) -> tuple:
        """Return ({client_id: probs}, {client_id: site}, [skipped ids])."""
        raw = {}
        sites = {}
        prints = {}
        skipped = []
        for cid, (_, metadata) in client_updates.items():
            blob = metadata.get("telltale_probe_scores")
            shape = metadata.get("telltale_shape")
            if blob is None or shape is None:
                skipped.append(cid)
                continue
            p = np.frombuffer(bytes.fromhex(blob), dtype=np.float16)
            p = p.reshape(int(shape[0]), int(shape[1]))
            raw[cid] = p.astype(np.float64)
            prints[cid] = metadata.get("telltale_fingerprint")
            if "telltale_site" in metadata:
                sites[cid] = metadata["telltale_site"]
        # Instrument self-check: a client that scored a different probe set is
        # an instrument fault, not drift. Keep only the most common set.
        counts = {}
        for fp in prints.values():
            counts[fp] = counts.get(fp, 0) + 1
        scores = {}
        if counts:
            common = sorted(counts, key=lambda fp: (-counts[fp], str(fp)))[0]
            for cid in raw:
                if prints[cid] == common:
                    scores[cid] = raw[cid]
                else:
                    skipped.append(cid)
        return scores, sites, sorted(skipped)

    def _level(self, scores: dict) -> dict:
        """Mean JS divergence to the leave-one-out mean of the others."""
        ids = sorted(scores)
        n = len(ids)
        if n < 2:
            return {}
        total = np.zeros_like(scores[ids[0]])
        for cid in ids:
            total = total + scores[cid]
        out = {}
        for cid in ids:
            p = scores[cid]
            q = (total - p) / (n - 1)
            out[cid] = float(self._js(p, q).mean())
        return out

    def _js(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Jensen-Shannon divergence in nats, per row."""
        p = np.clip(p, 1e-12, 1.0)
        q = np.clip(q, 1e-12, 1.0)
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=-1)
        kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=-1)
        return 0.5 * kl_pm + 0.5 * kl_qm

    def _cusum_step(self, cid: str, r: int) -> float | None:
        """Advance one client's CUSUM by this round. None until it can run."""
        if r < self.REF_STOP:
            return None
        hist = self.levels[cid]
        ref = [
            hist[t] for t in sorted(hist) if self.REF_START <= t < self.REF_STOP
        ]
        if len(ref) < 2:
            return None
        mu = float(np.mean(ref))
        sd = float(np.std(ref, ddof=1))
        if sd <= 0:
            return None
        z = (hist[r] - mu) / sd
        s = max(0.0, self.cusum.get(cid, 0.0) + z - self.K)
        self.cusum[cid] = s
        return s

    def _json(self, obj: object) -> str:
        """Minimal JSON writer. ``json`` is not on the import allowlist."""
        if obj is None:
            return "null"
        if isinstance(obj, bool):
            return "true" if obj else "false"
        if isinstance(obj, float):
            return repr(float(obj))  # numpy floats print as np.float64(...)
        if isinstance(obj, int):
            return str(obj)
        if isinstance(obj, str):
            return '"' + obj.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(obj, dict):
            items = [
                self._json(str(k)) + ": " + self._json(v)
                for k, v in obj.items()
            ]
            return "{" + ", ".join(items) + "}"
        return "[" + ", ".join(self._json(v) for v in obj) + "]"

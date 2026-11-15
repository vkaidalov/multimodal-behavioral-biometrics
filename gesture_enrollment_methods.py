"""Shared, cached enrollment scoring for centroid and nearest-5 templates.

The score of a k-gesture window is ALWAYS the arithmetic mean of its k
individual gesture scores. All windows stay inside session/repetition.
Enrollment uses sessions 1 and 2 only, verification sessions 3 and 4 only.
"""
from __future__ import annotations
from collections import defaultdict
import time
import numpy as np
from gesture_evaluation_core import get_session_ids, unit, biometric_scores_for_scenario

METHODS = ('centroid', 'top5')


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def nearest_k_score(probes, enrollment, k=5, chunk_size=512):
    """Mean of k largest cosine scores; normalized rows, bounded memory."""
    probes = normalize_rows(probes)
    enrollment = normalize_rows(enrollment)
    if not len(enrollment):
        raise ValueError('Empty enrollment template')
    k = min(k, len(enrollment))
    out = np.empty(len(probes), dtype=np.float64)
    for i in range(0, len(probes), chunk_size):
        sim = probes[i:i+chunk_size] @ enrollment.T
        out[i:i+chunk_size] = np.partition(sim, -k, axis=1)[:, -k:].mean(axis=1)
    return out


class PreparedScenario:
    """One scenario prepared once, reused for all methods and window sizes."""
    def __init__(self, emb, meta, users, splits, scenario, methods=METHODS, verbose=False):
        self.emb_raw = np.asarray(emb)
        self.emb = np.asarray(emb, dtype=np.float64)
        self.users_map = users
        self.meta = meta
        self.users = sorted(users)
        self.methods = tuple(methods)
        self.scenario = scenario
        self._splits = splits
        self.groups = {}
        self.enroll = {}
        self.ref = {}
        self.scores = {}
        self._windows = {}
        begin = time.perf_counter()
        for user in self.users:
            indices = users[user]
            sessions = get_session_ids(splits, scenario, user)
            if len(sessions) != 4:
                raise ValueError(f'Expected four sessions for {scenario}/{user}')
            present = {str(meta[i]['session_id']) for i in indices}
            if present != set(sessions):
                raise ValueError(f'Unexpected/missing sessions: {scenario}/{user}: {present} != {set(sessions)}')
            s1 = np.asarray([i for i in indices if str(meta[i]['session_id']) == sessions[0]], dtype=np.int64)
            s2 = np.asarray([i for i in indices if str(meta[i]['session_id']) == sessions[1]], dtype=np.int64)
            if not len(s1) or not len(s2):
                raise ValueError(f'Missing enrollment sessions: {scenario}/{user}')
            self.enroll[user] = np.concatenate([s1, s2])
            # Same reference as legacy core.user_reference.
            self.ref[user] = unit((unit(self.emb[s1].mean(axis=0)) +
                                   unit(self.emb[s2].mean(axis=0))) / 2)
            g = defaultdict(list)
            for i in indices:
                m = meta[i]
                if str(m['session_id']) in sessions[2:]:
                    g[(str(m['session_id']), str(m.get('repetition','')))].append(i)
            sorted_groups = []
            for (sid, rep), idx in sorted(g.items()):
                idx.sort(key=lambda i: (float(meta[i]['start_ms']), float(meta[i]['end_ms']), i))
                sorted_groups.append((sid, rep, np.asarray(idx, dtype=np.int64)))
            if not sorted_groups:
                raise ValueError(f'Missing verification gestures: {scenario}/{user}')
            self.groups[user] = sorted_groups
        # For each target, score all probe gestures of each source ONCE.
        for j, target in enumerate(self.users, 1):
            self.scores[target] = {}
            template = self.emb[self.enroll[target]]
            for source in self.users:
                probes = np.concatenate([idx for _,_,idx in self.groups[source]])
                vals = {}
                if 'top5' in self.methods:
                    vals['top5'] = nearest_k_score(self.emb[probes], template, k=5)
                self.scores[target][source] = vals
            if verbose and (j == 1 or j % 3 == 0 or j == len(self.users)):
                print(f'[TEMPLATES] {scenario}: {j}/{len(self.users)} targets; '
                      f'elapsed={time.perf_counter()-begin:.1f}s', flush=True)

    def windows(self, k):
        """Metadata for all source windows, indexed by flat probe positions."""
        if k in self._windows:
            return self._windows[k]
        result = {}
        for source in self.users:
            entries = []
            offset = 0
            for session, rep, idx in self.groups[source]:
                n = len(idx)
                if n >= k:
                    starts = np.arange(n-k+1, dtype=np.int64)
                    ends = starts+k-1
                    for start, end in zip(starts, ends):
                        start_ms = float(self.meta[idx[start]]['start_ms'])
                        end_ms = float(self.meta[idx[end]]['end_ms'])
                        if end_ms < start_ms:
                            raise ValueError('Nonchronological gesture window')
                        entries.append((offset+int(start), offset+int(end),
                                        start_ms, end_ms, session, rep,
                                        (end_ms-start_ms)/1000))
                offset += n
            result[source] = entries
        self._windows[k] = result
        return result

    def score_records(self, k, method, impostor_per_source=100, seed=20261112):
        """Exactly the same eligible user set / impostor sampling for both methods.

        Returns legacy tuple rows: target, source, label, score, start, end,
        session, repetition, collection latency.
        """
        if method not in self.methods:
            raise ValueError(f'Method not prepared: {method}')
        if k <= 0:
            raise ValueError('k must be positive')
        if method == 'centroid':
            # Preserve legacy float32 window averaging and exact scores.
            # A float64 prefix-sum implementation differs in the last bits
            # from float32 averaging; exact reproducibility matters here.
            return biometric_scores_for_scenario(self.emb_raw, self.meta,
                self.users_map, self._splits, self.scenario, k,
                impostor_per_source, seed)
        candidates = self.windows(k)
        active = [u for u in self.users if candidates[u]]
        if len(active) < 2:
            return []
        rng = np.random.default_rng(seed+k)
        rows = []
        for target in active:
            for source in active:
                desc = candidates[source]
                score_by_gesture = self.scores[target][source][method]
                cumsum = np.empty(len(score_by_gesture)+1, dtype=np.float64)
                cumsum[0] = 0
                np.cumsum(score_by_gesture, out=cumsum[1:])
                if source != target and impostor_per_source > 0 and len(desc) > impostor_per_source:
                    selected = np.sort(rng.choice(len(desc), size=impostor_per_source, replace=False))
                else:
                    selected = range(len(desc))
                for j in selected:
                    start_i,end_i,start_ms,end_ms,session,rep,latency = desc[int(j)]
                    score = float((cumsum[end_i+1]-cumsum[start_i])/k)
                    rows.append((target,source,int(target == source),score,start_ms,end_ms,
                                 session,rep,latency))
        return rows

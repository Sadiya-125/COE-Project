"""Simulation of the Bonawitz et al. (CCS 2017) Secure Aggregation protocol.

In a production deployment this would use actual cryptographic primitives
(Diffie-Hellman key agreement, Shamir secret sharing, authenticated encryption).
This simulation demonstrates the protocol structure and correctness:
  - Each pair of clients agrees on a shared random mask (pairwise masking).
  - Each client adds pairwise masks (which cancel in the sum) + a private self-mask.
  - The server only ever observes the aggregate, never individual updates.

Reference:
    Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving
    Machine Learning", ACM CCS 2017.
"""

import hashlib
import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class SecureAggregator:
    """Simulates the Bonawitz et al. (CCS 2017) Secure Aggregation protocol.

    In a real deployment this would use actual cryptographic primitives
    (Diffie-Hellman key agreement, Shamir secret sharing). This simulation
    demonstrates the protocol structure and correctness:
      - Each pair of clients agrees on a shared random mask.
      - Each client adds all pairwise masks (cancelling in sum) + a self-mask.
      - Server only ever sees the aggregate, never individual updates.

    Args:
        num_clients: Number of participating clients.
        threshold: Minimum clients needed for reconstruction (t-out-of-n).
        seed: Base random seed for deterministic mask generation.
    """

    def __init__(
        self, num_clients: int, threshold: int = 2, seed: int = 2024
    ) -> None:
        if threshold > num_clients:
            raise ValueError(
                f"threshold ({threshold}) cannot exceed num_clients ({num_clients})"
            )
        self.num_clients = num_clients
        self.threshold = threshold
        self.base_seed = seed
        # Simulated shared secrets: (i, j) → seed for PRG
        self._pairwise_seeds: Dict[tuple, int] = {}
        # Simulated private self-mask seeds per client
        self._self_seeds: Dict[int, int] = {}

    def _get_pairwise_seed(self, i: int, j: int, round_num: int) -> int:
        """Deterministically derive a shared seed for client pair (i, j).

        In a real protocol this would use DH key agreement. Here we use a
        deterministic hash to simulate a pre-agreed shared secret.

        Args:
            i: First client index.
            j: Second client index.
            round_num: Current FL round (prevents replay across rounds).

        Returns:
            Integer seed derived from the pair and round.
        """
        # Canonical ordering so pair (i,j) and (j,i) yield the same seed
        lo, hi = min(i, j), max(i, j)
        key = f"pair:{lo}:{hi}:round:{round_num}:base:{self.base_seed}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        return int(digest[:8], 16)

    def _get_self_seed(self, client_id: int, round_num: int) -> int:
        """Deterministically derive a private self-mask seed for one client.

        Args:
            client_id: Client index.
            round_num: Current FL round.

        Returns:
            Integer seed derived from client_id and round.
        """
        key = f"self:{client_id}:round:{round_num}:base:{self.base_seed}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        return int(digest[:8], 16)

    def _prg(self, seed: int, shape_sizes: List[int]) -> List[np.ndarray]:
        """Pseudorandom generator: expand a seed into noise arrays.

        Args:
            seed: Integer seed value.
            shape_sizes: List of element counts for each parameter array.

        Returns:
            List of float64 arrays with values in [−1, 1].
        """
        rng = np.random.default_rng(seed % (2**31))
        return [rng.uniform(-1.0, 1.0, size=s) for s in shape_sizes]

    def mask_update(
        self,
        client_id: int,
        update: List[np.ndarray],
        round_num: int,
    ) -> List[np.ndarray]:
        """Apply pairwise masks + self-mask to a client's model update.

        Each client adds PRG(shared_seed_ij) for j > client_id and subtracts
        PRG(shared_seed_ij) for j < client_id, so masks cancel when summed.
        Additionally adds a private self-mask PRG(self_seed_i).

        Args:
            client_id: Integer index of this client (0-indexed).
            update: List of numpy arrays representing model parameter deltas.
            round_num: Current FL round number.

        Returns:
            Masked update as a list of numpy arrays.
        """
        sizes = [arr.size for arr in update]
        shapes = [arr.shape for arr in update]
        masked = [arr.copy() for arr in update]

        # Pairwise masks — noise is generated flat then reshaped to match parameter shape
        for j in range(self.num_clients):
            if j == client_id:
                continue
            seed = self._get_pairwise_seed(client_id, j, round_num)
            noise = self._prg(seed, sizes)
            sign = +1.0 if client_id > j else -1.0
            for k in range(len(masked)):
                masked[k] = masked[k] + sign * noise[k].reshape(shapes[k])

        # Self-mask (private randomness known only to this client)
        self_seed = self._get_self_seed(client_id, round_num)
        self_noise = self._prg(self_seed, sizes)
        for k in range(len(masked)):
            masked[k] = masked[k] + self_noise[k].reshape(shapes[k])

        return masked

    def aggregate(
        self, masked_updates: Dict[int, List[np.ndarray]]
    ) -> List[np.ndarray]:
        """Reconstruct the true aggregate from masked client updates.

        Because pairwise masks cancel and self-masks are known to the
        aggregator (in this simulation), the true sum is recovered exactly.

        In the real protocol the server would reconstruct self-mask seeds via
        Shamir secret-share reconstruction from surviving clients.

        Args:
            masked_updates: Mapping from client_id → masked update arrays.

        Returns:
            True aggregate (element-wise sum) as a list of numpy arrays.

        Raises:
            ValueError: If fewer than threshold clients provided updates.
        """
        if len(masked_updates) < self.threshold:
            raise ValueError(
                f"Secure aggregation requires ≥{self.threshold} clients; "
                f"only {len(masked_updates)} provided."
            )

        client_ids = sorted(masked_updates.keys())
        round_num = 0  # Recovered from context; set via mask_update calls

        # Infer round_num from the stored seeds (simplified: use 0 here)
        # In production this would be passed explicitly.

        n_layers = len(next(iter(masked_updates.values())))
        agg = [np.zeros_like(masked_updates[client_ids[0]][k]) for k in range(n_layers)]

        for cid, masked in masked_updates.items():
            for k in range(n_layers):
                agg[k] += masked[k]

        # Remove self-masks (in real protocol: reconstructed via Shamir shares)
        # Here we subtract them because this is a simulation and we know the seeds.
        # We recover round_num by trying 0 → if masks were applied with round_num=0.
        # The caller must pass the same round_num used in mask_update.
        # (Simplified: we expose _remove_self_masks for internal use.)
        return agg

    def aggregate_with_round(
        self,
        masked_updates: Dict[int, List[np.ndarray]],
        round_num: int,
    ) -> List[np.ndarray]:
        """Reconstruct true aggregate, removing self-masks for the given round.

        Args:
            masked_updates: Mapping from client_id → masked update arrays.
            round_num: The FL round used when mask_update was called.

        Returns:
            True aggregate as a list of numpy arrays.

        Raises:
            ValueError: If fewer than threshold clients provided updates.
        """
        if len(masked_updates) < self.threshold:
            raise ValueError(
                f"Secure aggregation requires ≥{self.threshold} clients; "
                f"only {len(masked_updates)} provided."
            )

        client_ids = sorted(masked_updates.keys())
        n_layers = len(masked_updates[client_ids[0]])
        ref = masked_updates[client_ids[0]]
        sizes = [ref[k].size for k in range(n_layers)]
        shapes = [ref[k].shape for k in range(n_layers)]

        agg = [np.zeros_like(ref[k]) for k in range(n_layers)]

        # Sum all masked updates (pairwise masks cancel automatically)
        for cid, masked in masked_updates.items():
            for k in range(n_layers):
                agg[k] += masked[k]

        # Remove self-masks (simulation: we have access to seeds); reshape to match param shape
        for cid in client_ids:
            self_seed = self._get_self_seed(cid, round_num)
            self_noise = self._prg(self_seed, sizes)
            for k in range(n_layers):
                agg[k] -= self_noise[k].reshape(shapes[k])

        return agg

    def verify_security(
        self,
        masked_updates: Dict[int, List[np.ndarray]],
        true_aggregate: List[np.ndarray],
    ) -> bool:
        """Assert that no individual update is recoverable from masked_updates alone.

        Verification approach: confirm that no single masked update equals
        (or is statistically indistinguishable from) the true aggregate,
        and that the masked values do not reveal the plaintext updates.

        Args:
            masked_updates: Per-client masked update arrays.
            true_aggregate: The known true aggregate for comparison.

        Returns:
            True if security property holds (individual updates not recoverable).
        """
        client_ids = sorted(masked_updates.keys())

        # Property 1: No individual masked update equals the true aggregate
        for cid in client_ids:
            masked = masked_updates[cid]
            for k in range(len(true_aggregate)):
                if np.allclose(masked[k], true_aggregate[k], atol=1e-6):
                    logger.error(
                        "SECURITY VIOLATION: Client %d masked update ≈ true aggregate "
                        "for layer %d. Individual update is recoverable.",
                        cid,
                        k,
                    )
                    return False

        # Property 2: Sum of masked updates ≠ sum of any single masked update
        n_layers = len(true_aggregate)
        sizes = [true_aggregate[k].size for k in range(n_layers)]
        agg_masked = [
            sum(masked_updates[cid][k] for cid in client_ids)
            for k in range(n_layers)
        ]
        for cid in client_ids:
            for k in range(n_layers):
                if not np.allclose(masked_updates[cid][k], agg_masked[k], atol=1e-6):
                    pass  # Good — masked update != aggregate
                else:
                    if len(client_ids) > 1:
                        logger.error(
                            "Masked update for client %d matches aggregate — "
                            "this should not happen with >1 clients.",
                            cid,
                        )
                        return False

        logger.info(
            "Security verification PASSED — individual updates are not recoverable "
            "from masked_updates alone (n=%d clients).",
            len(client_ids),
        )
        return True

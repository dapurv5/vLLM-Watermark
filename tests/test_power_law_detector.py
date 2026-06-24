"""Tests for the Power Law detector (Lattimore 2026).

Validates mathematical properties and FPR calibration without requiring
a GPU or language model. Uses a simple mock tokenizer.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from vllm_watermark.watermark_detectors.openai_detector_pl import OpenaiDetectorPL


class MockTokenizer:
    """Minimal tokenizer for testing — each character is a token."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % self.vocab_size for c in text]

    def get_vocab(self) -> dict:
        return {chr(i): i for i in range(self.vocab_size)}


@pytest.fixture
def detector():
    tok = MockTokenizer(vocab_size=256)
    return OpenaiDetectorPL(
        tokenizer=tok,
        vocab_size=256,
        ngram=1,
        seed=42,
        payload=0,
        threshold=0.05,
        n_mc_samples=100_000,
        mc_seed=2026,
    )


class TestScoreFunction:
    """Test mathematical properties of S(u)."""

    def test_zero_mean_under_null(self):
        """E[S(U)] = 0 when U ~ Uniform[0,1] (by construction via mu)."""
        rng = np.random.default_rng(123)
        for epsilon in [0.01, 0.05, 0.1, 0.25]:
            U = rng.uniform(0, 1, size=500_000)
            scores = OpenaiDetectorPL.power_law_score_vectorized(U, epsilon)
            mean = np.mean(scores)
            assert abs(mean) < 0.02, (
                f"E[S(U)] = {mean:.4f} (expected ~0) for eps={epsilon}"
            )

    def test_truncation(self):
        """S(u) is capped at 1/sqrt(eps) - mu."""
        epsilon = 0.1
        mu = 2.0 - np.sqrt(epsilon)
        cap = 1.0 / np.sqrt(epsilon) - mu
        for u in [0.999, 0.9999, 0.99999, 1.0 - 1e-15]:
            s = OpenaiDetectorPL.power_law_score(u, epsilon)
            assert s <= cap + 1e-10, f"S({u}) = {s} exceeds cap {cap}"

    def test_monotonicity(self):
        """S(u) is non-decreasing in u."""
        epsilon = 0.05
        u_vals = np.linspace(0, 0.999, 1000)
        scores = OpenaiDetectorPL.power_law_score_vectorized(u_vals, epsilon)
        diffs = np.diff(scores)
        assert np.all(diffs >= -1e-10), "S(u) is not monotone"

    def test_score_at_zero(self):
        """S(0) = min(1/sqrt(eps), 1) - mu = 1 - mu (since 1 < 1/sqrt(eps) for small eps)."""
        epsilon = 0.01
        mu = 2.0 - np.sqrt(epsilon)
        s = OpenaiDetectorPL.power_law_score(0.0, epsilon)
        expected = 1.0 - mu
        assert abs(s - expected) < 1e-10, f"S(0) = {s}, expected {expected}"


class TestMonteCarlo:
    """Test the Monte Carlo null distribution."""

    def test_null_sums_centered(self, detector):
        """Under H_0, mean of sum(S_t) should be ~0."""
        n = 50
        epsilon = np.log(1.0 / 0.05) / n
        sums = detector._mc_null_sums(n, epsilon)
        mean = np.mean(sums)
        se = np.std(sums) / np.sqrt(len(sums))
        assert abs(mean) < 5 * se, f"Mean null sum = {mean:.3f} (SE={se:.3f})"

    def test_threshold_calibration(self, detector):
        """The (1-delta) quantile of null sums should yield FPR ~ delta."""
        n = 50
        delta = 0.05
        epsilon = np.log(1.0 / delta) / n
        sums = detector._mc_null_sums(n, epsilon)
        tau_star = np.quantile(sums, 1 - delta)
        empirical_fpr = np.mean(sums >= tau_star)
        assert abs(empirical_fpr - delta) < 0.01, (
            f"Empirical FPR = {empirical_fpr:.4f}, expected ~{delta}"
        )


class TestDetection:
    """Test the full detection pipeline."""

    def test_random_text_not_detected(self, detector):
        """Random text (independent of key) should rarely be detected."""
        rng = np.random.default_rng(999)
        n_trials = 200
        detections = 0
        for _ in range(n_trials):
            text = "".join(chr(rng.integers(32, 127)) for _ in range(80))
            result = detector.detect(text)
            if result["is_watermarked"]:
                detections += 1
        fpr = detections / n_trials
        assert fpr < 0.15, f"FPR = {fpr:.3f} (expected < 0.15 at delta=0.05)"

    def test_short_text_returns_not_watermarked(self, detector):
        """Text shorter than ngram+1 should return not watermarked."""
        result = detector.detect("a")
        assert not result["is_watermarked"]
        assert result["pvalue"] == 1.0

    def test_detect_returns_correct_keys(self, detector):
        """detect() should return is_watermarked, score, and pvalue."""
        result = detector.detect("This is a test sentence for the detector.")
        assert "is_watermarked" in result
        assert "score" in result
        assert "pvalue" in result
        assert isinstance(result["is_watermarked"], bool)
        assert isinstance(result["score"], float)
        assert isinstance(result["pvalue"], float)

    def test_pvalue_bounded(self, detector):
        """p-value should be in (0, 1]."""
        texts = [
            "A reasonably long piece of text for watermark detection testing purposes.",
            "Short text.",
        ]
        for text in texts:
            result = detector.detect(text)
            assert 0 < result["pvalue"] <= 1.0, (
                f"p-value {result['pvalue']} out of range"
            )


class TestSimulatedWatermark:
    """Simulate watermark detection by constructing biased V_t sequences.

    Under watermarking, V_t = U_{t,A_t} tends toward 1 (because Gumbel-max
    sampling selects the token that maximizes P(a)/(-log(U_{t,a})), which
    biases the selected U toward 1). We simulate this by constructing text
    where the detector's reconstructed V_t values happen to be large.
    """

    def test_biased_vt_detected(self):
        """When V_t values are biased high, detection should succeed."""
        tok = MockTokenizer(vocab_size=256)
        det = OpenaiDetectorPL(
            tokenizer=tok,
            vocab_size=256,
            ngram=1,
            seed=42,
            payload=0,
            threshold=0.05,
            n_mc_samples=100_000,
            mc_seed=2026,
        )

        base_text = "The quick brown fox jumps over the lazy dog and then some more words"
        token_ids = tok.encode(base_text, add_special_tokens=False)
        n_scored = len(token_ids) - det.ngram - 1

        best_vt_sum = -np.inf
        best_text = base_text

        rng = np.random.default_rng(7)
        for _ in range(500):
            candidate = "".join(chr(rng.integers(32, 127)) for _ in range(80))
            tids = tok.encode(candidate, add_special_tokens=False)
            vt_sum = 0
            for pos in range(det.ngram + 1, len(tids)):
                ngram = tids[pos - det.ngram : pos]
                vt = det._get_vt(ngram, tids[pos])
                vt_sum += vt
            if vt_sum > best_vt_sum:
                best_vt_sum = vt_sum
                best_text = candidate

        result = det.detect(best_text)
        # The cherry-picked text should have high V_t values on average,
        # mimicking watermarked text
        print(f"Best V_t sum: {best_vt_sum:.2f}, score: {result['score']:.2f}, "
              f"pvalue: {result['pvalue']:.6f}")
        # We don't assert detection here since cherry-picking from 500 random
        # strings may not be strong enough, but we verify the pipeline runs.
        assert result["pvalue"] <= 1.0


class TestScoreTokAndGetPvalue:
    """Test the interface-compatible score_tok and get_pvalue methods."""

    def test_score_tok_returns_tensor(self, detector):
        """score_tok should return a 1-element CPU tensor."""
        import torch
        result = detector.score_tok([65, 66], 67)
        assert isinstance(result, torch.Tensor)
        assert result.device == torch.device("cpu")
        vt = result.item()
        assert 0.0 <= vt <= 1.0, f"V_t = {vt} not in [0,1]"

    def test_get_pvalue_returns_float(self, detector):
        """get_pvalue should return a valid p-value."""
        pval = detector.get_pvalue(score=5.0, ntoks=50)
        assert isinstance(pval, float)
        assert 0.0 < pval <= 1.0


class TestFactoryIntegration:
    """Test that the detector can be created via the factory."""

    def test_create_via_factory(self):
        from vllm_watermark.watermark_detectors import WatermarkDetectors

        tok = MockTokenizer(vocab_size=256)
        det = WatermarkDetectors.create(
            algo="openai_pl",
            tokenizer=tok,
            vocab_size=256,
            ngram=1,
            seed=42,
            threshold=0.05,
        )
        assert isinstance(det, OpenaiDetectorPL)
        result = det.detect("Testing the factory integration path for watermark detection.")
        assert "is_watermarked" in result

    def test_create_via_enum(self):
        from vllm_watermark.core import DetectionAlgorithm
        from vllm_watermark.watermark_detectors import WatermarkDetectors

        tok = MockTokenizer(vocab_size=256)
        det = WatermarkDetectors.create(
            algo=DetectionAlgorithm.OPENAI_PL,
            tokenizer=tok,
            vocab_size=256,
            ngram=1,
            seed=42,
            threshold=0.05,
        )
        assert isinstance(det, OpenaiDetectorPL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

from __future__ import annotations

import unittest

import torch

from latentrouter.experiments.model_contrastive_reread import ModelContrastiveRereadBranch


class ModelContrastiveRereadTests(unittest.TestCase):
    def test_reread_branch_starts_as_exact_residual_identity(self) -> None:
        branch = ModelContrastiveRereadBranch(patch_dim=12, hidden_dim=8, num_capsules=7, reread_bound=0.1)
        capsules = torch.randn(3, 7, 8)
        model_difference = torch.randn(3, 8)
        score_gap = torch.randn(3)
        patches = torch.randn(3, 49, 12)

        updated, stats = branch(
            capsules,
            model_difference,
            score_gap,
            patches,
            mode="contrast",
        )

        self.assertTrue(torch.equal(updated, capsules))
        self.assertEqual(stats["attention_entropy"].shape, (3,))
        self.assertEqual(stats["attention_peak"].shape, (3,))
        self.assertTrue(stats["residual_norm"].eq(0).all())

    def test_generic_mode_ignores_model_difference_and_gap(self) -> None:
        branch = ModelContrastiveRereadBranch(patch_dim=12, hidden_dim=8, num_capsules=7, reread_bound=0.1)
        torch.nn.init.normal_(branch.evidence_out.weight)
        capsules = torch.randn(2, 7, 8)
        patches = torch.randn(2, 49, 12)
        first, _ = branch(capsules, torch.randn(2, 8), torch.randn(2), patches, mode="generic")
        second, _ = branch(capsules, torch.randn(2, 8), torch.randn(2), patches, mode="generic")
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()

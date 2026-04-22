from __future__ import annotations

import unittest

from research.phase4_decision_policy import infer_branch_status


class Phase4HypothesisUpdateTests(unittest.TestCase):
    def test_infer_branch_status_marks_killed_after_replication_kill(self) -> None:
        status = infer_branch_status(
            [
                {"candidate_name": "branch-a", "stage": "scout", "decision": "promote"},
                {"candidate_name": "branch-a", "stage": "replication", "decision": "kill"},
            ]
        )
        self.assertEqual(status["branch-a"], "killed")

    def test_infer_branch_status_marks_promoted_after_scout_promote(self) -> None:
        status = infer_branch_status([{"candidate_name": "branch-b", "stage": "scout", "decision": "promote"}])
        self.assertEqual(status["branch-b"], "promoted")


if __name__ == "__main__":
    unittest.main()

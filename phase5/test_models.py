"""Semantic checks for the changed reset rule and Bellman implementation."""
import unittest
import numpy as np
from model import CASES, Config, Model, regions, solve


class ModelTests(unittest.TestCase):
    def test_default_size_and_plus_z0_pattern(self):
        m = Model(Config('replenishing_plus'))
        self.assertEqual(m.config.states, 70)
        policy = solve(m)[2]
        self.assertEqual(regions(policy[:, 0].tolist()), [[35, 42], [67, 70]])

    def test_replacement_preserves_every_budget(self):
        for case in CASES:
            m = Model(Config(case))
            np.testing.assert_allclose(m.p.sum(axis=1), 1.0)
            for z in range(m.layers):
                for s in range(m.config.states):
                    ns, nz, cost = m.transition(s,z,0,0.25)
                    self.assertEqual(nz,z)
                    self.assertEqual(cost,9990)
                    self.assertEqual(ns,m.transition(0,z,0,0.25)[0])

    def test_budget_boundaries_and_cost_signs(self):
        for case in CASES:
            m = Model(Config(case))
            for s in range(m.config.states):
                for z in range(m.layers):
                    raw = z-int(m.safety[s])
                    if raw < 0:
                        with self.assertRaises(ValueError):
                            m.transition(s,z,1,0.9)
                    else:
                        _,nz,_ = m.transition(s,z,1,0.9)
                        self.assertEqual(nz,min(m.layers-1,raw))
        m = Model(Config('replenishing_plus'))
        self.assertEqual(m.transition(0,11,1,0.1)[1],12)
        self.assertEqual(m.transition(42,0,1,0.1)[1],1)
        with self.assertRaises(ValueError):
            m.transition(34,0,1,0.1)

    def test_vectorized_backup_against_scalar_definition(self):
        rng = np.random.default_rng(51)
        for case in CASES:
            m = Model(Config(case))
            n = m.config.states
            v = rng.uniform(100,50000,(n,m.layers))
            actual = m.q_from_value(v)
            for s in range(n):
                for z in range(m.layers):
                    expected = 9990+0.7*sum(m.p[0,j]*v[j,z] for j in range(n))
                    self.assertAlmostEqual(actual[s,z,0],expected,places=8)
                    raw = z-int(m.safety[s])
                    if raw < 0:
                        self.assertTrue(np.isinf(actual[s,z,1]))
                    else:
                        nz = min(m.layers-1,raw)
                        expected = m.cost[s]+0.7*sum(m.p[s,j]*v[j,nz] for j in range(n))
                        self.assertAlmostEqual(actual[s,z,1],expected,places=8)

    def test_baseline_lower_bound_and_greedy_policy_evaluation(self):
        baseline = solve(Model(Config('baseline')))[0]
        for case in CASES:
            v,q,p,r = solve(Model(Config(case)))
            self.assertTrue(np.all(v >= baseline-1e-7))
            self.assertLess(r['bellman_residual'],1e-8)
            self.assertLess(r['policy_evaluation_error'],1e-7)
            self.assertEqual(r['unsafe_actions'],0)


if __name__ == '__main__':
    unittest.main()

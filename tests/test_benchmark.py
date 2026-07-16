import torch

import benchmark


def test_problem_composition_is_exact():
    X = torch.rand(8, 6, dtype=torch.double)
    for name in ("zdt1", "zdt2", "zdt3", "dtlz2"):
        problem = benchmark.get_problem(name, 6)
        assert torch.allclose(
            problem.compose(problem.components(X), X),
            problem.evaluate(X),
        )


def test_composite_samples_respect_g_domain():
    X = torch.zeros(2, 6, dtype=torch.double)
    C = torch.tensor([[-2.0], [2.0]], dtype=torch.double)
    zdt = benchmark.get_problem("zdt1", 6).compose(C, X)
    dtlz = benchmark.get_problem("dtlz2", 6).compose(C, X)
    assert torch.all(zdt[:, 1] >= 1.0)
    assert torch.all(dtlz[:, 0] >= 1.0)


def test_zdt3_uses_true_ideal():
    ideal = benchmark.get_problem("zdt3", 6).ideal
    assert torch.allclose(
        ideal,
        torch.tensor([0.0, -0.7733690123], dtype=torch.double),
    )

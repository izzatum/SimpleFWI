import inspect
import pytest
import torch
from simplefwi.fwi2d.parameters import VelocityParameters
from simplefwi.fwi2d.optimizers import OPTIMIZER_RECIPES, make_optimizer
from simplefwi.fwi2d.inversion import InversionRunner


class Quadratic:
    regularizer = None
    def iter_data_losses(self, m, source_batch_size):
        yield ((m*1e6-0.3)**2).sum()/2
    def loss(self, m):
        return ((m*1e6-0.3)**2).sum()/2


def native_inventory():
    return {name for name, cls in vars(torch.optim).items()
            if inspect.isclass(cls) and issubclass(cls, torch.optim.Optimizer)
            and cls is not torch.optim.Optimizer}


def test_complete_native_inventory():
    assert torch.__version__.split('+')[0] == '2.12.0'
    assert native_inventory() == set(OPTIMIZER_RECIPES)


@pytest.mark.parametrize('name', sorted(native_inventory()))
def test_native_recipe_executes(name):
    recipe = OPTIMIZER_RECIPES[name]
    initial = torch.full((3, 4), 1600., dtype=torch.float64)
    mask = torch.ones_like(initial, dtype=torch.bool)
    mask[0, 0] = False
    model = VelocityParameters(initial, (1400., 2200.), mask,
                               representation=recipe.representation)
    optimizer = make_optimizer(name, model)
    result = InversionRunner(Quadratic(), model, optimizer, source_batch_size=1).step()
    assert result.accepted and torch.isfinite(torch.tensor(result.loss))
    assert model.velocity()[0, 0] == initial[0, 0]
    assert bool(((model.velocity() >= 1400) & (model.velocity() <= 2200)).all())
    for p in model.parameters():
        assert torch.isfinite(p).all()
        grad = p.grad.coalesce().values() if p.grad.is_sparse else p.grad
        assert torch.isfinite(grad).all()
    def finite_tree(value):
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
        elif isinstance(value, dict):
            for item in value.values(): finite_tree(item)
        elif isinstance(value, (list, tuple)):
            for item in value: finite_tree(item)
    finite_tree(optimizer.state)


def test_incompatible_native_instance_rejected_early():
    model = VelocityParameters([[1600., 1700.]], (1400., 2200.), None)
    with pytest.raises(ValueError, match='SparseAdam'):
        InversionRunner(Quadratic(), model, torch.optim.SparseAdam(model.parameters()), source_batch_size=1)
    sparse = VelocityParameters([[1600., 1700.]], (1400., 2200.), None,
                                representation='sparse_embedding')
    with pytest.raises(ValueError, match='sparse_embedding'):
        InversionRunner(Quadratic(), sparse, torch.optim.Adam(sparse.parameters()), source_batch_size=1)

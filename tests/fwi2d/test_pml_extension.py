import torch
from simplefwi.fwi2d.types import Grid2D, BoundaryConfig
from simplefwi.fwi2d.pml import extend_model


def test_nearest_extension_and_boundary_gradient():
    grid = Grid2D((3,4), (2.,7.), (-4.,8.))
    boundary = BoundaryConfig(pml_cells=2)
    m = torch.arange(12, dtype=torch.float64).reshape(grid.shape).requires_grad_()
    result = extend_model(m, grid, boundary)
    expected = m.detach()[torch.tensor([0,0,0,1,2,2,2])[:,None], torch.tensor([0,0,0,1,2,3,3,3])[None,:]]
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    torch.testing.assert_close(m.grad, torch.tensor([3.,1.,3.])[:,None]*torch.tensor([3.,1.,1.,3.])[None,:], check_dtype=False)


def test_free_surface_has_no_top_padding():
    grid = Grid2D((3,4), (2.,7.))
    m = torch.arange(12, dtype=torch.float64).reshape(grid.shape)
    extended = extend_model(m, grid, BoundaryConfig(top='free_surface', pml_cells=2))
    assert extended.shape == (5,8)
    torch.testing.assert_close(extended[0,2:6], m[0])

"""Pinned geometry, physically fixed refinement and independent observations."""
import json

import numpy as np
import pytest
import torch

from examples.fwi2d.camembert import (DEFAULT_CONFIG, make_camembert,
                                     make_refined_camembert, simulate_data)
from simplefwi.fwi2d.dispersion import assess_dispersion


def test_exact_geometry_acquisition_and_active_windows():
    grid,true,initial,acquisition,boundary=make_camembert(DEFAULT_CONFIG)
    assert grid.shape == (48,48)
    assert grid.spacing_m == (10.,10.) and grid.origin_m == (0.,0.)
    assert true.dtype == initial.dtype == torch.float64
    assert torch.all(initial == 1500.)
    z,x=torch.meshgrid(torch.arange(48)*10.,torch.arange(48)*10.,indexing='ij')
    circle=(z-235.)**2+(x-235.)**2 <= 60.**2
    assert torch.equal(true == 2000.,circle)
    assert torch.all(true[~circle] == 1500.)
    assert len(acquisition.sources_m) == 8
    assert len(acquisition.receivers_m) == len(set(acquisition.receivers_m)) == 32
    assert not set(acquisition.sources_m) & set(acquisition.receivers_m)
    for z,x in acquisition.sources_m+acquisition.receivers_m:
        assert 40 <= z <= 430 and 40 <= x <= 430
        assert z in (40.,430.) or x in (40.,430.)
    config=json.loads(DEFAULT_CONFIG.read_text())
    assert config['candidate_frequencies_hz'] == [5.,10.,15.]
    assert config['active_windows'] == {'routine':[16,32,16,32],'extended':[12,36,12,36]}
    assert boundary.pml_cells == 80 and boundary.sigma_max == 60.


def test_separate_surface_acquisition_and_frequency_classification():
    grid,_,_,acquisition,_=make_camembert(acquisition='surface')
    assert all(z == 40 for z,x in acquisition.sources_m)
    assert all(z == 60 for z,x in acquisition.receivers_m)
    assert len(set(acquisition.sources_m)) == 8
    assert len(set(acquisition.receivers_m)) == 32
    assert [assess_dispersion(grid,(1500.,2000.),[f]).passed for f in (5.,10.,15.)] == [True,True,False]
    config=json.loads(DEFAULT_CONFIG.read_text())
    assert config['frequency_classification']['15'] == 'algebraic-only-on-10m-grid'


def test_refinement_retains_circle_in_metres_and_boundary_thickness():
    grid,true,_,acquisition,boundary=make_camembert()
    fine,fine_true,_,fine_acquisition,fine_boundary=make_refined_camembert()
    assert fine.shape == (135,135) and fine.spacing_m == (5.,5.)
    assert fine.origin_m == (-100.,-100.)
    assert fine_acquisition == acquisition
    assert torch.equal(fine_true[20:115:2,20:115:2],true)
    assert fine_boundary.pml_cells*fine.spacing_m[0] == boundary.pml_cells*grid.spacing_m[0]
    assert fine_boundary.sigma_max == boundary.sigma_max
    assert assess_dispersion(fine,(1500.,2000.),(15.,)).passed


@pytest.mark.parametrize('origin',[(0.,0.),(-70.,120.)])
def test_free_surface_refinement_preserves_top_plane_and_source_depths(tmp_path,origin):
    config=json.loads(DEFAULT_CONFIG.read_text())
    config['boundary']['top']='free_surface'
    config['grid']['origin_m']=list(origin)
    config['circle']['centre_m']=[235.+offset for offset in origin]
    for key in ('acquisition','surface_acquisition'):
        for points in config[key].values():
            for point in points:
                point[:]=[coordinate+offset for coordinate,offset in zip(point,origin)]
    path=tmp_path/'free_surface.json'
    path.write_text(json.dumps(config))
    grid,true,_,acquisition,boundary=make_camembert(path,acquisition='surface')
    fine,fine_true,_,fine_acquisition,fine_boundary=make_refined_camembert(
        path,acquisition='surface')
    assert fine.shape == (115,135)
    assert fine.origin_m == (origin[0],origin[1]-100.)
    assert fine_acquisition == acquisition
    assert fine_boundary.top == boundary.top == 'free_surface'
    assert [z-fine.origin_m[0] for z,x in fine_acquisition.sources_m] == [40.]*8
    assert [z-fine.origin_m[0] for z,x in fine_acquisition.receivers_m] == [60.]*32
    assert torch.equal(fine_true[0:95:2,20:115:2],true)
    assert fine_boundary.pml_cells*fine.spacing_m[0] == boundary.pml_cells*grid.spacing_m[0]


def test_larger_finer_observations_are_distinct_from_same_grid_derivative_data():
    grid,true,initial,acquisition,boundary=make_camembert()
    fine,fine_true,_,fine_acquisition,fine_boundary=make_refined_camembert()
    coarse_data=simulate_data(grid,true,acquisition,boundary,[5.])
    independent=simulate_data(fine,fine_true,fine_acquisition,fine_boundary,[5.])
    initial_data=simulate_data(grid,initial,acquisition,boundary,[5.])
    assert coarse_data.shape == independent.shape == (1,8,32)
    assert independent.dtype == torch.complex128 and torch.isfinite(independent).all()
    difference=float(torch.linalg.vector_norm(independent-coarse_data)/torch.linalg.vector_norm(independent))
    contrast=float(torch.linalg.vector_norm(initial_data-independent)/torch.linalg.vector_norm(independent))
    # These broad nondegeneracy checks are not a continuum-accuracy threshold.
    assert 1e-5 < difference < .1
    assert difference < contrast
    print('Camembert coarse/fine data discrepancy / initial contrast:',difference,contrast)


@pytest.mark.parametrize('kwargs',[{'refinement':1},{'refinement':2.5},{'margin_m':0.},{'margin_m':103.}])
def test_invalid_reference_geometry_rejected(kwargs):
    with pytest.raises(ValueError):
        make_refined_camembert(**kwargs)

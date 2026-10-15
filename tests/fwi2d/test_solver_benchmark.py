"""A solved benchmark is accepted only after its accuracy criteria pass."""
import copy
import math

import pytest
import torch

from benchmarks.fwi2d import solver as benchmark


def classify(record, dtype):
    assert hasattr(benchmark, 'classify_accuracy'), 'benchmark must enforce accuracy before passing'
    return benchmark.classify_accuracy(record, dtype)


def record():
    return {'status': 'solved', 'waveform_relative_error': 1e-12,
            'gradient_relative_error': 1e-12, 'hvp_relative_error': 1e-12,
            'setup_plus_all_rhs_seconds': .125}


@pytest.mark.parametrize('dtype', [torch.complex128, torch.complex64])
@pytest.mark.parametrize('quantity', ['waveform', 'gradient', 'hvp'])
def test_out_of_tolerance_quantity_cannot_pass(dtype, quantity):
    result = record()
    result[quantity+'_relative_error'] = .1
    original = copy.deepcopy(result)
    classified = classify(result, dtype)
    assert classified['status'] == 'failed_accuracy'
    assert quantity+'_relative_error' in classified['accuracy_failures']
    assert classified['accuracy_criteria']['maximum_relative_errors'][quantity+'_relative_error'] < .1
    assert result == original
    assert classified['setup_plus_all_rhs_seconds'] == .125


@pytest.mark.parametrize('invalid', [math.nan, math.inf, -1.])
def test_invalid_measurement_cannot_pass(invalid):
    result = record()
    result['gradient_relative_error'] = invalid
    assert classify(result, torch.complex128)['status'] == 'failed_accuracy'


def test_missing_measurement_cannot_pass():
    result = record()
    del result['hvp_relative_error']
    assert classify(result, torch.complex128)['status'] == 'failed_accuracy'


@pytest.mark.parametrize('dtype', [torch.complex128, torch.complex64])
def test_exact_thresholds_are_accepted_and_reported(dtype):
    classified = classify(record(), dtype)
    limits = classified['accuracy_criteria']['maximum_relative_errors']
    boundary = record() | limits
    assert classify(boundary, dtype)['status'] == 'passed'
    assert not classified['accuracy_failures']
    assert classified['accuracy_evaluation'] == 'passed'


def test_nonconvergence_stays_failed_with_accuracy_unrun():
    result = {'status': 'failed', 'report': {'iterations': 120, 'relative_residual': .0005}}
    classified = classify(result, torch.complex128)
    assert classified['status'] == 'failed'
    assert classified['report'] == result['report']
    assert classified['accuracy_evaluation'] == 'unrun_after_failed_solve'


@pytest.mark.parametrize(('dtype', 'limits'), [
    (torch.complex128, (1e-7, 1e-7, 1e-5)),
    (torch.complex64, (1e-3, 1e-2, 1e-2)),
])
def test_declared_precision_limits(dtype, limits):
    actual = classify(record(), dtype)['accuracy_criteria']['maximum_relative_errors']
    assert tuple(actual[name+'_relative_error'] for name in ('waveform', 'gradient', 'hvp')) == limits


def test_saved_cpu_measurements_have_current_criteria_and_preserve_failures():
    import json
    from pathlib import Path
    saved = json.loads((Path(__file__).resolve().parents[2]/'benchmarks/fwi2d/solver-cpu.json').read_text())
    assert saved['accuracy_annotation']['measurements_rerun'] is False
    assert [item['fgmres']['status'] for item in saved['records']] == ['passed', 'passed', 'failed', 'failed']
    for item in saved['records']:
        result = item['fgmres']
        assert classify(result, saved['dtype']) == result

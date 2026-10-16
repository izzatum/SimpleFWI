# Running the SI PyTorch workflow

Use Python 3.12 and PyTorch 2.12.0. Follow the CPU or CUDA installation sequence
in the [README](../README.md); the exact dependency versions are in
[the lock file](../requirements/fwi2d-py312.lock). The lock fixes release versions;
it does not make CPU and CUDA wheels interchangeable. Select the appropriate
Torch wheel before installing it. Other Python versions have not been certified
for this new implementation, even though legacy package metadata is broader.

Copy the complete Python block from the README into a file in a fresh output
directory. It creates `quickstart.pt` with detached velocity and complex sampled
data and prints the accepted-step count and finite-value checks. This convenience
file is **not** a restart checkpoint; native optimizer/RNG/configuration state is
saved by the checkpoint API described in [outputs](outputs.md).

The example uses the direct CPU reference solver. Adam's accepted step means
finite parameters, gradients, state and final objective; it does not guarantee a
monotone objective or a reconstructed inclusion after one step. The small domain
and five PML cells are illustrative. The test uses same-grid observations, so it
is an optimizer/PDE integration check rather than an independent accuracy result.

The installed command line interface provides a similar small Camembert run:

```bash
python -m simplefwi.fwi2d --output /tmp/simplefwi-new-run
```

Choose an output directory that does not already contain an archive. Inspect the
manifest's `accuracy` classification before interpreting output. For explicit
configurations, run from the checkout:

```bash
python examples/fwi2d/marmousi.py --config examples/fwi2d/configs/camembert.json --output /tmp/camembert-new-run
python examples/fwi2d/marmousi.py --config examples/fwi2d/configs/marmousi.json --output /tmp/marmousi-new-run
```

The latter is a cropped, low-frequency Marmousi smoke configuration, not a
full-model reconstruction or performance result. Explicitly supplied datasets
must have the units and provenance required by the loader. Geometry cannot be
silently clipped and zero frequencies are rejected.

To regenerate the domain figure from the README's literal geometry assignments:

```bash
python examples/fwi2d/render_domain.py --output asset/fwi-workflow-domain.svg
python -m pytest tests/fwi2d/test_readme_example.py -q
```

Use [optimizer recipes](optimizers.md) for alternative native algorithms,
[accuracy guidance](accuracy.md) before changing frequency/grid/PML, and
[validation evidence](validation.md) for current limitations. Matrix-free
[Hessian actions](uncertainty.md) and accepted-state [recording](outputs.md) are
separate optional operations, not retained differentiable wavefields by default.

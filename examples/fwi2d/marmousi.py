"""Run a supplied bounded SI Marmousi JSON config with an installed package.

From any working directory:
    python /path/to/examples/fwi2d/marmousi.py --config /path/to/config.json --output run
The JSON's explicit model.path is relative to that JSON, not the current cwd.
Use configs/marmousi.json for the bundled 25 m cropped smoke. Full-domain,
12.5/6.25 m, 5–30 Hz and denser source sets remain gated research experiments.
"""
from simplefwi.fwi2d.__main__ import main

if __name__ == '__main__':
    main()

"""Deterministic geometry illustration generated from the README quickstart."""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from simplefwi.fwi2d import Acquisition, BoundaryConfig, Grid2D


def read_geometry(readme):
    """Read only literal geometry constructor arguments; never execute Markdown."""
    code, = re.findall(r'```python\n(.*?)\n```', readme.read_text(), re.S)
    classes = {'grid': Grid2D, 'boundary': BoundaryConfig, 'acquisition': Acquisition}
    found = {}
    for node in ast.parse(code).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in classes:
            continue
        call = node.value
        expected = classes[target.id]
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != expected.__name__:
            raise ValueError('README geometry must use explicit literal public constructors')
        found[target.id] = expected(*(ast.literal_eval(a) for a in call.args),
                                    **{k.arg: ast.literal_eval(k.value) for k in call.keywords})
    if found.keys() != classes.keys():
        raise ValueError('README must define grid, boundary and acquisition')
    return found['grid'], found['boundary'], found['acquisition']


def render(output, *, readme, preview=None):
    grid, boundary, acquisition = read_geometry(readme)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.hashsalt': 'simplefwi-fwi2d-domain-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.9))
    z0, x0 = grid.origin_m
    dz, dx = grid.spacing_m
    nz, nx = grid.shape
    width, height = (nx-1)*dx, (nz-1)*dz
    p = boundary.pml_cells
    for axis, top in zip(axes, ('pml', 'free_surface')):
        b = replace(boundary, top=top)
        top_padding = p*dz if b.top == 'pml' else 0.
        left, upper = x0-p*dx, z0-top_padding
        total_width, total_height = width+2*p*dx, height+top_padding+p*dz
        axis.add_patch(Rectangle((left, upper), total_width, total_height,
                                facecolor='#c8deee', edgecolor='#345976', linewidth=1.2))
        axis.add_patch(Rectangle((x0, z0), width, height,
                                facecolor='#fff7df', edgecolor='#665f48', linewidth=1.2))
        for x in np.arange(left, left+total_width+dx/2, dx):
            axis.plot([x, x], [upper, upper+total_height], color='#56758a', alpha=.16, linewidth=.5)
        for z in np.arange(upper, upper+total_height+dz/2, dz):
            axis.plot([left, left+total_width], [z, z], color='#56758a', alpha=.16, linewidth=.5)
        axis.add_patch(Circle((250., 200.), 75., facecolor='#c67b42',
                              edgecolor='#9c542c', alpha=.7))
        axis.text(x0+width/2, z0+height-45, 'Physical model [z, x]', ha='center', color='#403b31')
        axis.text(x0+width/2, z0+height+p*dz/2, f'PML: {p} cells = {p*dz:g} m', ha='center')
        if top == 'pml':
            axis.text(x0+width/2, z0-top_padding/2, 'Top PML', ha='center')
        else:
            axis.plot([x0, x0+width], [z0, z0], color='#733c8a', linewidth=3)
            axis.text(x0+width/2, z0-32, 'Free surface: p = 0', color='#733c8a', ha='center')
        sources = np.asarray(acquisition.sources_m)
        receivers = np.asarray(acquisition.receivers_m)
        axis.scatter(sources[:,1], sources[:,0], marker='*', s=110,
                     color='#a92d2b', edgecolor='white', linewidth=.5, label='Source', zorder=5)
        axis.scatter(receivers[:,1], receivers[:,0], marker='v', s=36,
                     color='#126c69', label='Receiver', zorder=5)
        axis.set_xlim(x0-p*dx-15, x0+width+p*dx+15)
        axis.set_ylim(z0+height+p*dz+15, z0-p*dz-15)
        axis.set_aspect('equal')
        axis.set_xlabel('x (m)')
        axis.set_ylabel('z (m), increasing downward')
        axis.set_title('Example: four-sided PML' if top == 'pml' else 'Option: pressure-release top', pad=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center', bbox_to_anchor=(.5,.06), frameon=False)
    fig.suptitle(f'{nz} × {nx} physical nodes · {dz:g} m spacing · origin (z, x) = {grid.origin_m} m', y=.97)
    fig.text(.5,.035, 'Lines mark node locations. Shading extends to outer padded nodes; the top row is constrained in the free-surface case.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, top=.87, bottom=.20, wspace=.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    if preview is not None:
        fig.savefig(preview, dpi=130)
    fig.savefig(output, format='svg', metadata={'Date': None, 'Creator': 'SimpleFWI render_domain.py',
                                              'Description': 'README physical grid, PML padding and optional free surface; source and receiver coordinates in metres.'})
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--output', type=Path, default=root/'asset/fwi-workflow-domain.svg')
    parser.add_argument('--readme', type=Path, default=root/'README.md')
    parser.add_argument('--preview', type=Path, help='Optional PNG for visual inspection')
    args = parser.parse_args()
    render(args.output, readme=args.readme, preview=args.preview)

"""Execute the README's public quickstart from outside the source tree."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_readme_quickstart(tmp_path):
    readme = (ROOT / 'README.md').read_text()
    blocks = re.findall(r'```python\n(.*?)\n```', readme, re.S)
    assert len(blocks) == 1, 'Keep one complete executable Python quickstart'
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['MPLCONFIGDIR'] = str(tmp_path / 'mpl')
    process = subprocess.run([sys.executable, '-c', blocks[0]], cwd=tmp_path,
                             env=env, text=True, capture_output=True, check=True)
    result = json.loads(process.stdout.strip().splitlines()[-1])
    assert result['accepted_steps'] == 1
    assert result['data_shape'] == [1, 2, 3]
    assert result['finite_data'] and result['finite_loss']
    assert result['closure_evaluations'] >= 1
    assert result['model_changed']
    assert result['loss'] >= 0
    assert (tmp_path / 'quickstart.pt').exists()


def test_readme_local_links_and_domain_illustration():
    readme = (ROOT / 'README.md').read_text()
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', readme):
        if '://' not in target and not target.startswith('#'):
            assert (ROOT / target.split('#')[0]).exists(), target
    assert '```mermaid' in readme
    svg = ROOT / 'asset/fwi-workflow-domain.svg'
    assert svg.exists()
    text = svg.read_text()
    assert '<svg' in text
    assert 'Free surface' in text and 'PML' in text


def test_domain_regenerates_from_readme_geometry(tmp_path):
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / 'mpl'))
    env.pop('PYTHONPATH', None)
    generated = tmp_path / 'domain.svg'
    subprocess.run([sys.executable, str(ROOT / 'examples/fwi2d/render_domain.py'),
                    '--output', str(generated)], cwd=tmp_path, env=env,
                   text=True, capture_output=True, check=True)
    assert generated.read_bytes() == (ROOT / 'asset/fwi-workflow-domain.svg').read_bytes()

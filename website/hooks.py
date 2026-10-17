"""Build-time metadata. Verification values are supplied by site owners only."""

import os
from pathlib import Path
import re
import shutil

from mkdocs.structure.files import File


def on_config(config):
    config.extra["verification"] = {}
    for variable, meta_name in (
        ("GOOGLE_SITE_VERIFICATION", "google-site-verification"),
        ("BING_SITE_VERIFICATION", "msvalidate.01"),
    ):
        value = os.environ.get(variable, "").strip()
        if value:
            if not re.fullmatch(r"[A-Za-z0-9_-]{6,256}", value):
                raise ValueError(f"{variable} must contain the actual verification token only")
            config.extra["verification"][meta_name] = value
    config.extra["source_schema"] = {
        "@context": "https://schema.org",
        "@type": "SoftwareSourceCode",
        "@id": config.repo_url + "#software",
        "name": "SimpleFWI",
        "description": config.site_description,
        "url": config.site_url,
        "codeRepository": config.repo_url,
        "programmingLanguage": "Python",
        "runtimePlatform": "Python 3.12; PyTorch 2.12.0",
        "license": "https://opensource.org/license/mit/",
        "isAccessibleForFree": True,
        "keywords": [
            "2D full-waveform inversion", "PyTorch", "frequency-domain FWI",
            "acoustic Helmholtz equation", "perfectly matched layer",
            "Gauss-Newton Hessian", "Marmousi", "Camembert",
        ],
    }
    return config


def on_files(files, config):
    """Expose the existing diagram to link validation without a duplicate source."""
    root = Path(config.config_file_path).resolve().parent
    files.append(File.generated(
        config,
        "assets/fwi-workflow-domain.svg",
        content=(root / "asset/fwi-workflow-domain.svg").read_bytes(),
    ))
    return files


def on_post_build(config):
    """Publish the reviewed root metadata without maintaining duplicate sources."""
    root = Path(config.config_file_path).resolve().parent
    for filename in ("llms.txt", "CITATION.cff"):
        shutil.copyfile(root / filename, Path(config.site_dir) / filename)

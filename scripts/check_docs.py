"""Check the rendered site's links and discovery metadata without network calls."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
import xml.etree.ElementTree as ET


CANONICAL_BASE = "https://izzatum.github.io/SimpleFWI/"
REPOSITORY = "https://github.com/izzatum/SimpleFWI"
PAGES = (
    "index.html",
    "helmholtz-pml/index.html",
    "multiscale-fwi/index.html",
    "hessian-uncertainty/index.html",
    "validation/index.html",
)


class Document(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.h1_count = 0
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.metas: dict[str, list[str]] = {}
        self.canonicals: list[str] = []
        self.schemas: list[dict] = []
        self._in_title = False
        self._in_schema = False
        self._schema = ""
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"] or "")
        if tag == "title":
            self._in_title = True
        if tag == "h1":
            self.h1_count += 1
        if tag == "meta":
            key = attributes.get("name") or attributes.get("property")
            if key:
                self.metas.setdefault(key, []).append(attributes.get("content") or "")
        if tag == "link" and attributes.get("rel") == "canonical":
            self.canonicals.append(attributes.get("href") or "")
        for attr in ("href", "src"):
            if attributes.get(attr):
                self.links.append(attributes[attr] or "")
        if tag == "script" and attributes.get("type") == "application/ld+json":
            self._in_schema = True
            self._schema = ""

    def handle_endtag(self, tag: str):
        if tag == "title":
            self._in_title = False
        if tag == "script" and self._in_schema:
            self.schemas.append(json.loads(self._schema))
            self._in_schema = False

    def handle_data(self, data: str):
        if self._in_title:
            self.title += data
        if self._in_schema:
            self._schema += data


def check(site_dir: Path, root: Path) -> None:
    documents = {
        path.relative_to(site_dir).as_posix(): Document(path.read_text(encoding="utf-8"))
        for path in site_dir.rglob("*.html")
    }
    assert set(PAGES) <= documents.keys(), "A required documentation page is missing"
    titles = set()
    descriptions = set()
    expected_urls = set()
    for path in PAGES:
        page = documents[path]
        expected = CANONICAL_BASE + path.removesuffix("index.html")
        expected_urls.add(expected)
        assert page.h1_count == 1, f"{path}: expected exactly one H1"
        assert page.title and page.title.endswith(" | SimpleFWI"), f"{path}: title"
        assert page.title not in titles, f"{path}: duplicate title"
        titles.add(page.title)
        assert page.canonicals == [expected], f"{path}: canonical URL mismatch"
        description = page.metas.get("description", [])
        assert len(description) == 1 and 60 <= len(description[0]) <= 220, f"{path}: description"
        assert description[0] not in descriptions, f"{path}: duplicate description"
        descriptions.add(description[0])
        assert page.metas.get("og:url") == [expected], f"{path}: OpenGraph URL"
        assert page.metas.get("og:title") == [page.title], f"{path}: OpenGraph title"
        assert page.metas.get("og:description") == description, f"{path}: OpenGraph description"
        assert page.metas.get("twitter:card") == ["summary"], f"{path}: social card"
        assert not any("noindex" in value.lower() for value in page.metas.get("robots", [])), f"{path}: noindex"
        assert len(page.schemas) == 1, f"{path}: expected one source-code schema"
        schema = page.schemas[0]
        assert schema.get("@type") == "SoftwareSourceCode", f"{path}: schema type"
        assert schema.get("codeRepository") == REPOSITORY, f"{path}: schema repository"
        assert schema.get("url") == CANONICAL_BASE, f"{path}: schema site URL"
        assert schema.get("license") == "https://opensource.org/license/mit/", f"{path}: schema license"

    link_count = 0
    for path, page in documents.items():
        if path == "404.html":
            continue
        page_url = CANONICAL_BASE + path.removesuffix("index.html")
        for link in page.links:
            if link.startswith(("mailto:", "data:", "javascript:")):
                continue
            url = urlsplit(urljoin(page_url, link))
            if url.scheme not in ("http", "https"):
                continue
            absolute = f"{url.scheme}://{url.netloc}{url.path}"
            if absolute.startswith(CANONICAL_BASE):
                relative = unquote(absolute[len(CANONICAL_BASE):])
                target = (site_dir / relative).resolve()
                assert target.is_relative_to(site_dir), f"{path}: escaping local link {link}"
                if target.is_dir():
                    target /= "index.html"
                assert target.is_file(), f"{path}: missing local target {link}"
                if url.fragment and target.suffix == ".html":
                    target_document = documents[target.relative_to(site_dir).as_posix()]
                    assert unquote(url.fragment) in target_document.ids, f"{path}: missing anchor {link}"
                link_count += 1
            elif absolute.startswith(REPOSITORY + "/blob/main/"):
                relative = unquote(absolute[len(REPOSITORY + "/blob/main/"):])
                assert (root / relative).is_file(), f"{path}: repository target missing {link}"
                link_count += 1

    sitemap = ET.parse(site_dir / "sitemap.xml")
    urls = {node.text for node in sitemap.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")}
    assert urls == expected_urls, "Sitemap must include exactly the five canonical page URLs"
    for filename in ("llms.txt", "CITATION.cff"):
        assert (site_dir / filename).read_bytes() == (root / filename).read_bytes(), f"Copied {filename} differs from reviewed root source"
    assert (site_dir / "assets/fwi-workflow-domain.svg").read_bytes() == (root / "asset/fwi-workflow-domain.svg").read_bytes(), "Domain illustration differs from README source"
    assert not (site_dir / "robots.txt").exists(), "Project-path robots.txt cannot control host-root crawling"
    print(f"PASS: {len(PAGES)} pages; {link_count} local/repository links; unique titles/descriptions; canonical/OpenGraph/source schema; sitemap; copied metadata and figure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    args = parser.parse_args()
    check(args.site_dir.resolve(), Path(__file__).resolve().parents[1])

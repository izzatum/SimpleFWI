# Maintaining the documentation site

The canonical project site is configured as
`https://izzatum.github.io/SimpleFWI/`. The five pages are static HTML built
with the standard MkDocs theme; navigation and content work without JavaScript.
No numerical dependencies, external fonts, analytics or GPU are needed.

```bash
python3.12 -m venv /tmp/simplefwi-docs
/tmp/simplefwi-docs/bin/python -m pip install -r requirements/docs.txt
/tmp/simplefwi-docs/bin/python -m mkdocs build --strict
/tmp/simplefwi-docs/bin/python scripts/check_docs.py --site-dir site
```

Preview with `python -m mkdocs serve` in that environment. The build writes to
the ignored `site/` directory. `website/docs/` owns page content;
`website/overrides/` adds metadata; `website/hooks.py` reads optional owner-supplied
verification values. Its `on_files` hook exposes the existing
`asset/fwi-workflow-domain.svg` as a generated MkDocs asset, so link validation
sees it and the built illustration stays byte-identical without a duplicate
tracked source. The checker rejects divergence.

The GitHub Actions workflow builds and checks pull requests without deployment
permissions. On `main`, a separate read-only job checks Pages configuration.
If Pages is disabled (HTTP 404) or still uses branch-based builds, it explicitly
skips deployment and explains the owner setup step; unexpected API errors fail
visibly. After the owner selects **Settings → Pages → Build and deployment →
Source → GitHub Actions**, a manual workflow run or the next `main` push deploys
the successful build. Only the deployment job has Pages write and OIDC
permissions. A local build or workflow file does not mean the URL is live or
indexed.

The build copies root `llms.txt` and `CITATION.cff` into the output without
maintaining duplicate tracked copies. `llms.txt` is an optional curated document
index, not a crawler requirement or ranking signal. See the repository's
[ownership and indexing guide](../guides/search-visibility.md) for the complete
owner handoff.

## Ownership verification and discovery

For a Search Console URL-prefix property at the canonical URL, add the actual
Google HTML-tag verification **content token** as the repository Actions
variable `GOOGLE_SITE_VERIFICATION`. For Bing Webmaster Tools, use the actual
Bing HTML-tag content token in `BING_SITE_VERIFICATION`. The build includes
these tags only when their values are supplied. The same environment variables
work locally. Verification tags are public ownership tokens, not API keys; do
not provide account credentials or invented placeholders.

After a live deployment and owner verification, submit
`https://izzatum.github.io/SimpleFWI/sitemap.xml` in Search Console and Bing
Webmaster Tools. URL inspection and indexing requests require access to the
relevant owner accounts. Monitor impressions, queries and crawl status there;
publication does not guarantee indexing or a search position.

Do not add a project-path `robots.txt` and assume it controls the shared
`izzatum.github.io` host root. The site has no crawler exclusions or robots
directives of its own. If a custom domain is adopted, update `site_url`, verify
the live canonical URL and regenerate the sitemap before changing backlinks.

Metadata identifies real source code with `SoftwareSourceCode` JSON-LD. Visible
FAQ answers cite repository evidence; no FAQ rich-result eligibility, ratings,
paper, DOI or ranking guarantee is claimed. Keep evidence claims tied to their
recorded numerical revision when documentation changes.

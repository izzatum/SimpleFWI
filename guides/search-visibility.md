# Search visibility and indexing

SimpleFWI targets researchers searching for **2D full-waveform inversion in
PyTorch**, with supporting pages about Python Helmholtz solvers, PML,
multi-frequency inversion and Hessian uncertainty. Search engines choose which
pages to index and rank; repository metadata or structured data cannot guarantee
a top position. Rankings also vary by query, location and search history.

## Repository About settings

Use the gear beside **About** on the GitHub repository. The current integration
can change code but returned `403 Resource not accessible by integration` when
editing the description and topics; the repository owner can enter these values.

Description:

```text
2D full-waveform inversion (FWI) in PyTorch: frequency-domain acoustic Helmholtz solvers, PML/free surface, multiscale inversion, Hessian uncertainty, and Marmousi tutorials.
```

Suggested topics:

```text
full-waveform-inversion, fwi, pytorch, seismic-inversion, seismic-imaging,
geophysics, helmholtz-equation, frequency-domain, inverse-problems,
perfectly-matched-layer, marmousi, gauss-newton, hessian,
uncertainty-quantification, scientific-computing, python, gpu
```

Add `https://izzatum.github.io/SimpleFWI/` as the Website only after a successful
Pages deployment and a verified HTTP 200 response. Keep the project name,
description, citation and website consistent across GitHub and any genuine
research or package listings.

## Publish the documentation

The site is prepared for `https://izzatum.github.io/SimpleFWI/`. Its HTML has
descriptive titles, canonical URLs, visible answers, links to numerical evidence
and SoftwareSourceCode structured data. The source and validation reports remain
available on GitHub. The sitemap is at
`https://izzatum.github.io/SimpleFWI/sitemap.xml` after deployment.

The repository's documentation workflow builds and checks the site. GitHub Pages
must first be enabled in **Settings → Pages**, with **GitHub Actions** as the
build source. After configuration, run the documentation workflow from the
default branch and verify its deployment. A successful local build or skipped
deployment does not mean the website is live.

The current integration also returned `403 Resource not accessible by integration`
when enabling Pages. The repository owner must select the build source above.
Until deployment is verified, the README and package metadata link to the
documentation on GitHub. Switch those links and the citation's `url` to the
published site after verifying it.

To preview and validate the site locally without installing numerical dependencies:

```bash
python -m venv .venv-docs
.venv-docs/bin/python -m pip install -r requirements/docs.txt
.venv-docs/bin/python -m mkdocs build --strict
.venv-docs/bin/python scripts/check_docs.py --site-dir site
.venv-docs/bin/python -m mkdocs serve
```

No project-level `robots.txt` is supplied. A file at
`/SimpleFWI/robots.txt` cannot control crawlers for `izzatum.github.io`; robots
rules are read from the host root, `/robots.txt`. Never block a page that needs
to be crawled to read its indexing directives.

## Google Search Console and Bing Webmaster Tools

These dashboards require the owner's Google/Microsoft account; GitHub access
does not grant access to them. Do not share account passwords or API keys in
repository files.

1. After the site is live, add a **URL-prefix property** for
   `https://izzatum.github.io/SimpleFWI/` in
   [Google Search Console](https://search.google.com/search-console/).
   This verifies the documentation site, not `github.com/izzatum/SimpleFWI`.
2. Use Google's HTML meta-tag verification method. In GitHub **Settings → Secrets
   and variables → Actions → Variables**, set `GOOGLE_SITE_VERIFICATION` to the
   exact `content` token Google provides (not the entire HTML tag), then rerun the
   documentation workflow. This public verification token becomes a meta tag in
   the generated HTML. Verify ownership only after the rendered homepage contains
   that tag. DNS verification of `github.io` is not available to a project owner.
3. Submit `https://izzatum.github.io/SimpleFWI/sitemap.xml`. Inspect the homepage
   and the most useful guide URLs, then request indexing if eligible. Repeated
   requests do not speed crawling or guarantee inclusion.
4. Add the site in [Bing Webmaster Tools](https://www.bing.com/webmasters/).
   Import the verified Search Console property when offered, or use Bing's
   supported verification method. For Bing's HTML meta-tag method, set the
   repository variable `BING_SITE_VERIFICATION` to its exact `content` token and
   redeploy. Submit the same sitemap and inspect crawl and indexing reports.
5. Record a baseline for branded searches (`SimpleFWI`) and relevant unbranded
   searches. Track impressions, clicks, indexed URLs and query-specific position
   over time. Bing's AI Performance report can also show citations when available;
   citation counts are distinct from search rank or model preference.

The public documentation contains answers and citations in ordinary HTML, so
core content is available without running JavaScript. `llms.txt` is an optional
curated reading index. It is not a Google/Bing ranking requirement or a guarantee
that an AI service will read or cite the project. SoftwareSourceCode metadata
describes the software; it does not promise a search rich result.

## Earn relevant references

Keep examples reproducible and benchmark limitations visible. Link the repository
and documentation from genuine author/lab pages, related publications and
relevant community contributions when appropriate. Use `CITATION.cff` for
software attribution and record exact experiment revisions. Register a DOI only
through a real archival release; do not invent a paper, DOI, review score or
performance claim. Avoid paid backlink schemes, fake stars and repetitive
keyword pages.

## Authoritative guidance

Reviewed 17 October 2026. These sources explain supported mechanisms; none
establishes SimpleFWI's current ranking. The attempted public branded searches
did not yield trustworthy ranking or indexing measurements.

- [Google SEO Starter Guide](https://developers.google.com/search/docs/fundamentals/seo-starter-guide)
- [Google AI features and website guidance](https://developers.google.com/search/docs/appearance/ai-features)
- [Google recrawl requests](https://developers.google.com/search/docs/crawling-indexing/ask-google-to-recrawl)
- [Search Console ownership verification](https://support.google.com/webmasters/answer/9008080?hl=en)
- [Google sitemap guidance](https://developers.google.com/search/docs/crawling-indexing/sitemaps/overview)
- [Google robots.txt scope](https://developers.google.com/crawling/docs/robots-txt/create-robots-txt)
- [Bing AI Performance reporting](https://blogs.bing.com/webmaster/2026/2/Introducing-AI-Performance-in-Bing-Webmaster-Tools-Public-Preview/)
- [Bing verification and Search Console import](https://blogs.bing.com/webmaster/2025/6/Start-Using-Bing-Webmaster-Tools-to-Improve-Your-Site-Visibility/)
- [Bing sitemap guidance](https://blogs.bing.com/webmaster/2025/7/Keeping-Content-Discoverable-with-Sitemaps-in-AI-Powered-Search/)
- [GitHub Pages overview](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

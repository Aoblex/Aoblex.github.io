# Copilot Coding Agent Instructions

## Repository Overview

**al-folio** Jekyll theme for academics. Personal site at https://aoblex.github.io.

- **Type:** Jekyll static site generator
- **Key Features:** CV, publications, blog, projects, news

## Tech Stack

- **Jekyll:** v4.x, **Ruby:** 3.3.5, **Python:** 3.13 (nbconvert), **Node.js** (purgecss, prettier)
- **Build:** `bundle exec jekyll build`, **Purgecss** for production CSS
- **Prettier:** v3.8.0+ with `@shopify/prettier-plugin-liquid`

## Local Development

```bash
bundle install
pip install jupyter   # if using Jupyter notebooks
bundle exec jekyll serve --port 4000
# Site at http://localhost:4000
```

**Requirements:** ImageMagick, nbconvert. Production: `JEKYLL_ENV=production`.

## Key Files

- `_config.yml` – Site config (url, baseurl, feature flags)
- `_data/` – socials.yml, cv.yml, citations.yml, venues.yml, coauthors.yml
- `_pages/`, `_posts/`, `_projects/`, `_news/`, `_teachings/` – Content
- `_bibliography/papers.bib` – Publications
- `_layouts/`, `_includes/`, `_sass/` – Theme

## Workflows

- **deploy.yml** – Build and deploy to GitHub Pages on push to main
- **prettier.yml** – Code formatting (run `npx prettier . --write` before committing)

## Common Pitfalls

- **YAML:** Quote values with `:`, `&`, `#` → `title: "My: Site"`
- **url/baseurl:** Personal site: `baseurl:` empty. Project site: `baseurl: /repo-name/`
- **Port in use:** `lsof -i :4000` then kill the process

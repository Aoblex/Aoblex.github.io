# Makefile for Aoblex.github.io (Jekyll al-folio)

.PHONY: install serve serve-drafts build clean help

# Default target: show help
help:
	@echo "Usage:"
	@echo "  make install       - Install dependencies (bundle + pip)"
	@echo "  make serve         - Start local dev server (port 4000)"
	@echo "  make serve-drafts  - Start server with draft posts preview"
	@echo "  make build         - Build static site to _site/"
	@echo "  make clean         - Remove _site/ build output"
	@echo ""
	@echo "Example: make serve"

# Install dependencies
install:
	bundle install
	pip install -r requirements.txt

# Start local dev server with livereload
serve:
	bundle exec jekyll serve --livereload --port 4000

# Start dev server including draft posts
serve-drafts:
	bundle exec jekyll serve --drafts --livereload --port 4000

# Build site (for local production build verification)
build:
	JEKYLL_ENV=production bundle exec jekyll build

# Remove build output
clean:
	rm -rf _site/

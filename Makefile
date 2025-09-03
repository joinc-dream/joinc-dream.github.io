.PHONY: install build deploy deploy-gh-pages deploy-docs clean

install:
	@echo "Installing Ruby gems..."
	bundle install

build:
	@echo "Building Jekyll site..."
	bundle exec jekyll build

deploy: build
	@echo "Jekyll site built to _site/."
	@echo "Please choose a deployment target:"
	@echo "  make deploy-gh-pages  (for gh-pages branch deployment)"
	@echo "  make deploy-docs      (for docs/ folder deployment in main branch)"

deploy-gh-pages: build
	@echo "Deploying to gh-pages branch..."
	@# Save current branch and stash changes if any
	@# CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
	@# git stash save --keep-index --include-untracked

	git checkout gh-pages
	git rm -rf .
	cp -r _site/. .
	git add .
	git commit -m "Deploy latest build to gh-pages"
	git push origin gh-pages
	@echo "Deployment to gh-pages branch complete."
	@# Restore original branch and unstash changes
	@# git checkout $(CURRENT_BRANCH)
	@# git stash pop

deploy-docs: build
	@echo "Deploying to docs/ folder in main branch..."
	@# Ensure you are on the main branch or your primary branch
	@# git checkout main

	cp -r _site/* docs/
	git add docs/
	git commit -m "Deploy latest build to docs/"
	git push origin main
	@echo "Deployment to docs/ folder complete."

clean:
	@echo "Cleaning _site/ directory..."
	rm -rf _site

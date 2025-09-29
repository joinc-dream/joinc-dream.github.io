.PHONY: install build deploy clean

install:
	@echo "Installing Ruby gems..."
	bundle install

build:
	@echo "Building Jekyll site..."
	bundle exec jekyll build

deploy: build
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
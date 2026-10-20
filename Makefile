.PHONY: verify reproduce test test-reference paper
verify:
	python -m eats verify
reproduce:
	python -m eats reproduce --bootstrap
test:
	python -m pytest -m 'not reference'
test-reference:
	python -c "import platform; assert (platform.system(), platform.machine()) == ('Darwin', 'arm64'), 'Reference tests require macOS ARM64 with the pinned runtime'"
	python -m pytest -m reference
paper:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
	cd paper && bibtex main
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex

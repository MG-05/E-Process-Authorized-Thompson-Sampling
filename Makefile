.PHONY: verify reproduce test paper
verify:
	python -m eats verify
reproduce:
	python -m eats reproduce --bootstrap
test:
	python -m pytest
paper:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
	cd paper && bibtex main
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex

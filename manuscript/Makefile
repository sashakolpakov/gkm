.PHONY: all

all: free_energy_abstraction.pdf

free_energy_abstraction.pdf: free_energy_abstraction.tex references.bib
	pdflatex -interaction=nonstopmode free_energy_abstraction.tex
	bibtex free_energy_abstraction
	pdflatex -interaction=nonstopmode free_energy_abstraction.tex
	pdflatex -interaction=nonstopmode free_energy_abstraction.tex

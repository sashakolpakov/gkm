LATEXMK ?= latexmk
LATEXFLAGS ?= -pdf -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all canonical compatibility clean

all: canonical

canonical: free_energy_abstraction.pdf

compatibility: semantic_cones.pdf

free_energy_abstraction.pdf: free_energy_abstraction.tex references.bib
	$(LATEXMK) $(LATEXFLAGS) free_energy_abstraction.tex

semantic_cones.pdf: semantic_cones.tex free_energy_abstraction.tex references.bib
	$(LATEXMK) $(LATEXFLAGS) semantic_cones.tex

clean:
	$(LATEXMK) -C free_energy_abstraction.tex
	$(LATEXMK) -C semantic_cones.tex

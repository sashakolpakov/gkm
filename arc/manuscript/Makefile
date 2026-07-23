PYTHON ?= python
MAIN := arc_agi3
MPLCONFIGDIR ?= /tmp/gkm-matplotlib
XDG_CACHE_HOME ?= /tmp/gkm-cache

.PHONY: all figures diagram bibliography paper one-page clean

all: diagram paper

figures:
	mkdir -p $(MPLCONFIGDIR) $(XDG_CACHE_HOME) figures
	MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) \
		$(PYTHON) scripts/generate_figures.py --output-dir figures

diagram:
	cd figure_sources && pdflatex -interaction=nonstopmode -halt-on-error \
		inverse_colimit_attachment_standalone.tex

bibliography:
	@if command -v bibtex >/dev/null 2>&1 && [ -x "$$(command -v bibtex)" ]; then \
		bibtex $(MAIN); \
	elif [ -x /usr/bin/bibtex.original ]; then \
		/usr/bin/bibtex.original $(MAIN); \
	else \
		echo "No working BibTeX executable found" >&2; exit 1; \
	fi

paper: figures
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(MAKE) bibliography
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

one-page:
	pdflatex -interaction=nonstopmode -halt-on-error gkm_one_page_summary.tex
	pdflatex -interaction=nonstopmode -halt-on-error gkm_one_page_summary.tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f gkm_one_page_summary.aux gkm_one_page_summary.log gkm_one_page_summary.out
	rm -f figure_sources/inverse_colimit_attachment_standalone.aux
	rm -f figure_sources/inverse_colimit_attachment_standalone.log

MAIN = arc_agi3
PYTHON ?= python
MPLCONFIGDIR ?= /tmp/gkm-matplotlib
XDG_CACHE_HOME ?= /tmp/gkm-cache

all: figures
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

figures:
	mkdir -p $(MPLCONFIGDIR) $(XDG_CACHE_HOME)
	MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) ../build_ls20_sawtooth_figure.py
	MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) ../build_marginal_complexity_figure.py
	MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) $(PYTHON) ../build_campaign_profiles_figure.py

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out

.PHONY: all figures clean

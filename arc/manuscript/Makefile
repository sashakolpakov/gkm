PYTHON ?= python
MAIN := arc_agi3
MPLCONFIGDIR ?= /tmp/gkm-matplotlib
XDG_CACHE_HOME ?= /tmp/gkm-cache
ENDPOINT_ROOT ?= ../crack_lab/releases/arc_agi3_gkm_v2_181/artifacts
ACQUISITION_ROOT ?= ../crack_lab/releases/arc_agi3_gkm_v2_181/acquisition_source
HISTORY_ROOT ?=
SOURCE_HISTORY_REVISION ?= 4d0e42f34d7b1db8305f03d725528dfdefe22511
RELEASE_RECEIPT ?=
RELEASE_VERIFIER_ROOT ?=
export RELEASE_VERIFIER_ROOT

.PHONY: all figures tables diagram bibliography paper one-page arxiv-bundle arxiv-check reproduce reproduce-full manifest verify-manifest test clean

all: diagram paper

figures:
	mkdir -p $(MPLCONFIGDIR) $(XDG_CACHE_HOME) figures
	MPLCONFIGDIR=$(MPLCONFIGDIR) XDG_CACHE_HOME=$(XDG_CACHE_HOME) \
		$(PYTHON) scripts/generate_figures.py --output-dir figures \
			--solutions-dir "$(ACQUISITION_ROOT)"

tables:
	$(PYTHON) scripts/generate_empirical_tables.py \
		--solutions-dir "$(ACQUISITION_ROOT)" --output-dir generated \
		--rst-output ../../docs/generated/marginal_complexity_by_level.rst

test:
	cd ../.. && $(PYTHON) -m pytest -q \
		arc/test_audit_action_boundaries.py \
		arc/test_audit_submission_taint.py \
		arc/test_audit_marginal_literal_reuse.py \
		arc/test_build_artifact_docs.py \
		arc/crack_lab/test_gkm_arena.py \
		arc/crack_lab/test_gkm_legs.py \
		arc/crack_lab/test_replay_scorecard.py \
		arc/crack_lab/test_arc_agi3_leaderboard_v3_gate.py \
		arc/crack_lab/test_verify_frozen_release.py \
		arc/manuscript/scripts/test_generate_figures.py \
		arc/manuscript/scripts/test_generate_empirical_tables.py \
		arc/manuscript/scripts/test_build_arxiv_bundle.py \
		arc/manuscript/scripts/test_reproduce_manuscript.py \
		arc/manuscript/scripts/test_update_sha256_manifest.py
	cd ../.. && $(PYTHON) arc/manuscript/scripts/update_sha256_manifest.py

reproduce:
	@test -n "$(RELEASE_RECEIPT)" || \
		{ echo "RELEASE_RECEIPT is required for the frozen release" >&2; exit 1; }
	$(PYTHON) scripts/reproduce_manuscript.py --write --build-paper \
		--release-root "$(ENDPOINT_ROOT)" \
		--acquisition-root "$(ACQUISITION_ROOT)" \
		--release-receipt "$(RELEASE_RECEIPT)" \
		--history-revision "$(SOURCE_HISTORY_REVISION)" $(if $(HISTORY_ROOT),--history-root "$(HISTORY_ROOT)") \
		--require-complete-lineage
	$(PYTHON) scripts/update_sha256_manifest.py --write

reproduce-full:
	@test -n "$(RELEASE_RECEIPT)" || \
		{ echo "RELEASE_RECEIPT is required for the frozen release" >&2; exit 1; }
	@test -n "$(OPINE_ARTIFACTS)" || \
		{ echo "OPINE_ARTIFACTS is required" >&2; exit 1; }
	@test -n "$(BASELINE_RELEASE)" || \
		{ echo "BASELINE_RELEASE is required" >&2; exit 1; }
	@test -n "$(BASELINE_REPO)" || \
		{ echo "BASELINE_REPO is required" >&2; exit 1; }
	@test -n "$(RETRODICT_RUNS)" || \
		{ echo "RETRODICT_RUNS is required" >&2; exit 1; }
	$(PYTHON) scripts/reproduce_manuscript.py --write --build-paper \
		--release-root "$(ENDPOINT_ROOT)" \
		--acquisition-root "$(ACQUISITION_ROOT)" \
		--release-receipt "$(RELEASE_RECEIPT)" \
		--history-revision "$(SOURCE_HISTORY_REVISION)" $(if $(HISTORY_ROOT),--history-root "$(HISTORY_ROOT)") \
		--require-complete-lineage \
		--opine-artifacts "$(OPINE_ARTIFACTS)" \
		--baseline-release "$(BASELINE_RELEASE)" \
		--baseline-repo "$(BASELINE_REPO)" \
		--retrodict-runs "$(RETRODICT_RUNS)"
	$(PYTHON) scripts/update_sha256_manifest.py --write

manifest:
	$(PYTHON) scripts/update_sha256_manifest.py --write

verify-manifest:
	$(PYTHON) scripts/update_sha256_manifest.py

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

paper: figures tables
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(MAKE) bibliography
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

one-page:
	pdflatex -interaction=nonstopmode -halt-on-error gkm_one_page_summary.tex
	pdflatex -interaction=nonstopmode -halt-on-error gkm_one_page_summary.tex

arxiv-bundle:
	$(PYTHON) scripts/build_arxiv_bundle.py

arxiv-check:
	$(PYTHON) scripts/build_arxiv_bundle.py --compile-check

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f gkm_one_page_summary.aux gkm_one_page_summary.log gkm_one_page_summary.out
	rm -f figure_sources/inverse_colimit_attachment_standalone.aux
	rm -f figure_sources/inverse_colimit_attachment_standalone.log

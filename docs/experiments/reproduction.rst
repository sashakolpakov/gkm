Reproduction Commands
=====================

Core Tests
----------

.. code-block:: bash

   python3 -m py_compile agent.py evo_game.py pattern_fsa.py \
     foraging/run_foraging_ecology.py \
     transduction/run_register_transducer_benchmark.py \
     bongard/run_bongard_symbolic_baseline.py \
     bongard/run_bongard_sparse_classifier.py \
     bongard/run_bongard_overcapacity_ablation.py \
     bongard/run_bongard_logo_adapter.py \
     bongard/run_abstraction_emergence.py \
     foraging/test_evo_game.py transduction/test_pattern_fsa.py \
     bongard/test_bongard_sparse_classifier.py bongard/test_abstraction_emergence.py

   python3 -m unittest

Abstraction
-----------

.. code-block:: bash

   python3 bongard/run_abstraction_emergence.py
   python3 bongard/run_abstraction_emergence.py --scenario or_factor --show-rules

Foraging
--------

.. code-block:: bash

   python3 foraging/run_foraging_ecology.py --width 10 --height 10 --food-count 6 --max-steps 80

Bongard-LOGO
------------

.. code-block:: bash

   git clone https://github.com/NVlabs/Bongard-LOGO.git downloads/Bongard-LOGO
   .venv/bin/python -m pip install pillow pandas
   .venv/bin/python bongard/run_bongard_logo_adapter.py \
     --dataset-dir downloads/Bongard-LOGO \
     --source abstract --feature-set all --limit 26 \
     --support-count 10 --validation-count 3 --hidden-count 3 \
     --max-rule-atoms 2 --max-candidate-atoms 20 --summary-only

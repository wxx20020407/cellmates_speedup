# Cellmates – a proper distance method for single-cell cancer data

## Project status

This project is currently under development. The functionalities that have been implemented are described below.

- [x] Data simulation (main code in function `rand_dataset()` at [datagen.py](simulation/datagen.py))
- [x] EM algorithm for root-to-centroid distance (main code in function `jcb_em()` at [em.py](inference/em.py))
- [x] Tree reconstruction algorithm from distance matrix
(main code in function `build_tree()` at [em.py](inference/em.py))
- [x] Testing for the execution of all three steps described above in [test_tree_inference_synth()](./tests/test_inference/test_em.py)

The pseudo-algorithm is explained in the [general writeup](https://www.overleaf.com/project/62e11c46a9cd5d7659fc29b4)
at Cellmates chapter, while an in-depth explanation of the EM
algorithm is in the [Cellmates article](https://www.overleaf.com/project/63a43d8ae9c4f58d4f48653c).

The next steps are:

- [ ] Implement NJ for tree reconstruction comparisons
- [ ] Add observation model parameters inference

## Requirements

- Python 3.10
- NumPy
- DendroPy
- NetworkX
- anndata


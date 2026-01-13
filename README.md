# Filament Progenitor Finder
This code tracks the progenitors of DisPerSE (Sousbie 2011) filaments across simulation snapshots. It starts from a list of filament structures at z=0, and trace back by calculating the spatial similarity between a descendant and progenitor candidates as:
$$\mathcal{S}=(\mathcal{S}_x\mathcal{S}_y\mathcal{S}_z)^{1/3}$$
where $\mathcal{S}_x$ is defined by multiplying the two normalized histograms of their positions along $x$-axis (See Jhee et al., in prep).

## Usage
Based on filament structures extracted using `DisPerSE`, one can simply run below:
```
$ python run_tree_smt_pandas_node2node.py -clsnum 0 -filnum 7
```

You may need to modify f-string for the DisPerSE file name in `disperse_reader.py`.

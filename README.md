
## `A new group of low-spin 50-70 Solar Mass Black Holes and the high pair-instability mass cutoff`


These files include the codes and data to re-produce the results of the work  _A new group of low-spin 50-70 Solar Mass Black Holes and the high pair-instability mass cutoff_, arXiv: [2510.22698](http://arxiv.org/abs/2510.22698)
, Yuan-Zhu Wang, Yin-Jie Li, Shi-Jie Gao, Shao-Peng Tang, and Yi-Zhong Fan

#### Main requirements
- [BILBY](https://git.ligo.org/lscsoft/bilby)
- [PyMultiNest](https://johannesbuchner.github.io/PyMultiNest/install.html)

#### Data
The events posterior samples are adopted from the [Gravitational Wave Open Science Center](https://www.gw-openscience.org/eventapi/html/GWTC/), 
here `C01:Mixed` samples are used for analysis and stored in `./GWTC3_BBH_Mixed_5000.pickle` and `./O4a_BBH_Mixed5000_Nobs_84.pickle`
We adopt 5000 samples for per event. For events with initial sample sizes less than 5000, the posterior samples are reused.

The injection campaign file `./mixture-semi_o1_o2-real_o3_o4a-polar_spins_20250503134659UTC.hdf`
was released by [Abbot et al.](https://zenodo.org/records/16740128), 
  
#### Hierarchical Bayesian inference
- Inference with our main model: run the python script `LowspinHighmass_infer.py`.

The inferred results `*.json` will be saved to `results`
  
#### Acknowledgements
The  publicly available code [GWPopulation](https://github.com/ColmTalbot/gwpopulation) is referenced to calculate the variance of log-likelihood in the Monte Carlo integrals, and the [FigureScript](https://dcc.ligo.org/public/0171/P2000434/003/Produce-Figures.ipynb) from [LIGO Document P2000434](https://dcc.ligo.org/LIGO-P2000434/public) is referenced to produced figures in this work.




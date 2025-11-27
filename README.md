
## `A new group of low-spin 50-70 Solar Mass Black Holes and the high pair-instability mass cutoff`


These files include the codes and data to re-produce the results of the work  _A new group of low-spin 50-70 Solar Mass Black Holes and the high pair-instability mass cutoff_, arXiv: [2510.22698](http://arxiv.org/abs/2510.22698)
, Yuan-Zhu Wang, Yin-Jie Li, Shi-Jie Gao, Shao-Peng Tang, and Yi-Zhong Fan

#### Main requirements
- [BILBY](https://git.ligo.org/lscsoft/bilby)
- [PyMultiNest](https://johannesbuchner.github.io/PyMultiNest/install.html)

#### Data
The events posterior samples are adopted from the [Gravitational Wave Open Science Center](https://www.gw-openscience.org/eventapi/html/GWTC/), 
here `C01:Mixed` samples are used for analysis and stored in `./GWTC3_BBH_Mixed_5000.pickle` and `./GWTC3_BBH_Mixed_5000.pickle`
We adopt 5000 samples for per event. For events with initial sample sizes less than 5000, the posterior samples are reused.

The injection campaigns `data/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5`
Note, one should first download the injection campaign
`o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5` from [Abbot et al.](https://doi.org/10.5281/zenodo.5546676), 
and set it to `data/`
  
#### Hierarchical Bayesian inference
- Inference with our main model: run the python script `inference.py` , and specify the One-component model or Two-component model by setting `label='Single_spin'` or `label='Double_spin'` in the script.

- Inference with the comparing models: run the python script `compared_inference.py`, and specify the population model *PS&LinearCorrelation*, *PS&DoubleSpin*, *PS&DefaultSpin*, or *PP&DefaultSpin* by setting `label='PS_linear'`, `label='PS_bimodal'`,`label='PS_default'`, or `label='PP_default'` in the script.

The inferred results `*.json` will be saved to `results`

#### Results
- `Double_spin_post.pickle` is the posterior samples inferred by the Two-component model.
- `Double_spin_informed.pickle` is the events' samples reweighed by the Two-component model.

#### Generate figures
Run the python script `figure_script.py`

The figures will be saved to `figures`
  
#### Acknowledgements
The  publicly available code [GWPopulation](https://github.com/ColmTalbot/gwpopulation) is referenced to calculate the variance of log-likelihood in the Monte Carlo integrals, and the [FigureScript](https://dcc.ligo.org/public/0171/P2000434/003/Produce-Figures.ipynb) from [LIGO Document P2000434](https://dcc.ligo.org/LIGO-P2000434/public) is referenced to produced figures in this work.




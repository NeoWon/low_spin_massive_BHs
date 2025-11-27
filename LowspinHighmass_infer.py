from models_PubV1 import *
import sys
#################################################################
#
#read data
#
#################################################################


data_dir='./'


#read data
with open(data_dir+'GWTC3_BBH_Mixed_5000.pickle', 'rb') as fp:
    samples3, evidences3 = pickle.load(fp)
Nobs3=len(samples3)
ln_evidences3=[np.log(ev) for ev in evidences3]
print('number of events in GWTC-3:',Nobs3)

with open(data_dir+'O4a_BBH_Mixed_5000_Nobs_50.pickle', 'rb') as fp:
    samples4_1, ln_evidences4_1 = pickle.load(fp)

with open(data_dir+'O4a_BBH_Mixed_5000_Nobs_34.pickle', 'rb') as fp:
    samples4_2, ln_evidences4_2 = pickle.load(fp)
samples4=samples4_1+samples4_2
ln_evidences4=ln_evidences4_1+ln_evidences4_2
Nobs4=len(samples4)
print('number of events in O4a:',Nobs4)

samples=samples3+samples4
ln_evidences=ln_evidences3+ln_evidences4
Nobs=len(samples)
print('number of events in GWTC-4:',Nobs)

#################################################################
#
#inference
#
#################################################################

outdir='results'
label='Double_nonparametric'
add_label=''
sampler='pymultinest'
sampler='dynesty'

#################################################################
#model
#################################################################
selection_function=Rate_selection_function_with_uncertainty
hyper_prior=hyper_Double_nonpmact
mass_model=Double_nonpmact_pair_nospin
priors=Double_nonpmact_priors()


class Hyper_selection_with_var(HyperparameterLikelihood):

    def convert_params(self):
        #self.parameters=convert_params(self.parameters)
        #self.parameters['constraints']=0
        #self.parameters.pop('constraints')
        pass

    def likelihood_ratio_obs_var(self):
        weights = np.nan_to_num(self.hyper_prior.prob(self.data) / self.data['prior'])
        expectations = np.nan_to_num(np.mean(weights, axis=-1))
        if np.any(expectations==0.):
            nan_count = np.count_nonzero(expectations==0.)
            #print(3.1, nan_count)
            return 100+np.square(nan_count), -1e4 - np.square(nan_count), -2.e10
        else:
            square_expectations = np.mean(weights**2, axis=-1)
            variances = (square_expectations - expectations**2) / (
                self.samples_per_posterior * expectations**2
            )
            variance = np.sum(variances)
            Neffs = expectations**2/square_expectations*self.samples_per_posterior
            Neffmin = np.min(Neffs)
            #if np.isnan(variance) or np.isnan(Neffmin) or np.any(expectations==0.):
            if np.isnan(variance):
                return 100, -1e4 , -2.e10
            else:
                return variance, Neffmin, np.sum(np.log(expectations))
        
    def log_likelihood(self):
        self.hyper_prior.parameters.update(self.parameters)
        self.convert_params()
        obs_vars, obs_Neff, llhr= self.likelihood_ratio_obs_var()
        if (obs_vars<1):
            selection, sel_vars, sel_Neff = selection_function(self.n_posteriors, mass_model, **self.parameters)
            #print(selection, sel_vars, sel_Neff)
            if (sel_vars+obs_vars<1):
                #print(self.parameters)
                #print(1, self.noise_log_likelihood() + llhr + selection)
                return self.noise_log_likelihood() + llhr + selection
            else:
                #print(2, - 2.e10 - np.square(100*(sel_vars+obs_vars-1)))
                return - 2.e10 - np.square(100*(sel_vars+obs_vars-1))
        else:
            #print(3, - 4.e10 - np.square(100*obs_vars - 100))
            return - 4.e10 - np.square(100*obs_vars - 100)
            

bilby.core.utils.setup_logger(outdir=outdir, label=label+add_label)


hp_likelihood = Hyper_selection_with_var(posteriors=samples, hyper_prior=hyper_prior, log_evidences=ln_evidences, max_samples=1e+100)
result = run_sampler(likelihood=hp_likelihood, priors=priors, sampler=sampler, nlive=1000,\
            use_ratio=False, outdir=outdir, label=label+add_label)


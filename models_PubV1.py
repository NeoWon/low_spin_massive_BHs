import numpy as np
from scipy.interpolate import CubicSpline
import bilby
from bilby.core.prior import TruncatedGaussian as TG
from bilby.core.prior import Uniform, LogUniform, PowerLaw
from bilby.core.sampler import run_sampler
from bilby.hyper.likelihood import HyperparameterLikelihood
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15
import astropy.units as u
import h5py
from scipy.special import erf
import os
import pickle
import sys


inject_dir='./'


#############################################################################################################################
#
#redshift
#
#############################################################################################################################
fdVcdz=interp1d(np.linspace(0,5,10000),4*np.pi*Planck15.differential_comoving_volume(np.linspace(0,5,10000)).to(u.Gpc**3/u.sr).value)
zs=np.linspace(0,2.9,2000)
dVdzs=fdVcdz(zs)
logdVdzs=np.log(dVdzs)

#The log likelihood for redshift distribution, which is (log_hyper_prior - log_default_prior), note than each prior is normalized 
def llh_z(z,gamma):
    norm=np.sum((1+zs)**(gamma-1)*dVdzs)*2.9/2000.
    norm0=np.sum((1+zs)**(-1)*dVdzs)*2.9/2000.
    return np.where((z>0) & (z<2.9), (1+z)**gamma/norm*norm0 , 1e-100)

# The normalized redshift distribution: log_hyper_prior
def p_z(z,gamma):
    norm=np.sum((1+zs)**(gamma-1)*dVdzs)*2.9/2000.
    p = (1+z)**(gamma-1)*fdVcdz(z)/norm
    return np.where((z>0) & (z<2.9), p , 1e-100)

# The expected number of mergers in the surveyed VT 
def log_N(T,lgR0,gamma):
    return np.log(T) + lgR0/np.log10(np.e) + np.logaddexp.reduce((gamma-1)*np.log(zs+1) + logdVdzs) + np.log(2.9/2000)

###########################################################################################################
#
#mass and spin
#
###########################################################################################################

############
#mass
############
def smooth(m,mmin,delta):
    A = (m-mmin == 0.)*1e-10 + (m-mmin)
    B = (m-mmin-delta == 0.)*1e-10 + abs(m-mmin-delta)
    f_m_delta = delta/A - delta/B
    return (np.exp(f_m_delta) + 1.)**(-1.)*(m<=(mmin+delta))+1.*(m>(mmin+delta))

def PL(m1,mmin,mmax,alpha,delta):
    norm=(mmax**(1-alpha)-mmin**(1-alpha))/(1-alpha)
    pdf = m1**(-alpha)/norm*smooth(m1,mmin,delta)
    return np.where((mmin<m1) & (m1<mmax), pdf , 1e-10000)

def PS_lowmass(m1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15):
    xi=np.exp(np.linspace(np.log(6),np.log(100),15))
    yi=np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15])
    cs = CubicSpline(xi,yi,bc_type='natural')
    xx=np.linspace(2,200,1000)
    yy=np.exp(cs(xx)*(xx>6)*(xx<100))*PL(xx,mmin,mmax,alpha,delta)
    norm=np.sum(yy)*198./1000.
    pm1 = np.exp(cs(m1)*(m1>6)*(m1<100))*PL(m1,mmin,mmax,alpha,delta)/norm
    return pm1

def PS_highmass(m1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10):
    xi=np.exp(np.linspace(np.log(10),np.log(150),10))
    yi=np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10])
    cs = CubicSpline(xi,yi,bc_type='natural')
    xx=np.linspace(2,200,1000)
    yy=np.exp(cs(xx)*(xx>10)*(xx<150))*PL(xx,mmin,mmax,alpha,delta)
    norm=np.sum(yy)*198./1000.
    pm1 = np.exp(cs(m1)*(m1>10)*(m1<150))*PL(m1,mmin,mmax,alpha,delta)/norm
    return pm1
####################################
#spin
####################################

#magnitude

def spline_a(a,nx1,nx2,nx3,nx4,nx5,amin,amax):
    xi=np.linspace(0,1,5)
    yi=np.array([nx1,nx2,nx3,nx4,nx5])
    cs = CubicSpline(xi,yi,bc_type='natural')
    xx=np.linspace(0,1,1000)
    yy=np.exp(cs(xx))*(xx>amin)*(xx<amax)
    norm=np.sum(yy)*1./1000.
    px = np.exp(cs(a))*(a>amin)*(a<amax)/norm
    return px

#cosine tilt angle

def spline_ct(ct,nx1,nx2,nx3,nx4):
    xi=np.linspace(-1,1,4)
    yi=np.array([nx1,nx2,nx3,nx4])
    cs = CubicSpline(xi,yi,bc_type='natural')
    xx=np.linspace(-1,1,1000)
    yy=np.exp(cs(xx)*(xx>-1)*(xx<1))
    norm=np.sum(yy)*2./1000.
    px = np.exp(cs(ct)*(ct>-1)*(ct<1))/norm
    return px

####################################
#only mass model
####################################
#Double
def Double_diff_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,r2):
    p1=PS_lowmass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15)*(1-r2)
    p2=PS_highmass(m1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10)*r2
    return p1+p2

def Double_diff_mass_pair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,r2,beta):
    pm1=Double_diff_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,r2)
    pm2=Double_diff_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,r2)
    pdf = pm1*pm2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-10000)

########################
#mass vs spin model
########################

###########
# m a ct nonparametric
def Double_nonpmact(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,na1,na2,na3,na4,na5,amin1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,oa1,oa2,oa3,oa4,oa5,amin2,amax2,r2,nx1,nx2,nx3,nx4,t1,t2,t3,t4):
    p1=PS_lowmass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15)*spline_a(a1,na1,na2,na3,na4,na5,amin1,amax1)*(1-r2)*spline_ct(ct1,nx1,nx2,nx3,nx4)
    p2=PS_highmass(m1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10)*spline_a(a1,oa1,oa2,oa3,oa4,oa5,amin2,amax2)*r2*spline_ct(ct1,t1,t2,t3,t4)
    return p1+p2

def Double_nonpmact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,na1,na2,na3,na4,na5,amin1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,oa1,oa2,oa3,oa4,oa5,amin2,amax2,r2,beta,nx1,nx2,nx3,nx4,t1,t2,t3,t4):
    pmact1=Double_nonpmact(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,na1,na2,na3,na4,na5,amin1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,oa1,oa2,oa3,oa4,oa5,amin2,amax2,r2,nx1,nx2,nx3,nx4,t1,t2,t3,t4)
    pmact2=Double_nonpmact(m2,a2,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,na1,na2,na3,na4,na5,amin1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,oa1,oa2,oa3,oa4,oa5,amin2,amax2,r2,nx1,nx2,nx3,nx4,t1,t2,t3,t4)
    pdf = pmact1*pmact2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-100)

def Double_nonpmact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,na1,na2,na3,na4,na5,amin1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,oa1,oa2,oa3,oa4,oa5,amin2,amax2,r2,beta,nx1,nx2,nx3,nx4,t1,t2,t3,t4):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_diff_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_nonpmact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,na1,na2,na3,na4,na5,amin1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,oa1,oa2,oa3,oa4,oa5,amin2,amax2,r2,beta,nx1,nx2,nx3,nx4,t1,t2,t3,t4)/AMP1
    return pdf

def Double_nonpmact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,na1,na2,na3,na4,na5,amin1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,oa1,oa2,oa3,oa4,oa5,amin2,amax2,r2,beta,nx1,nx2,nx3,nx4,t1,t2,t3,t4):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_diff_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_diff_mass_pair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,r2,beta)/AMP1
    pdf_spin = 1/4.
    return pdf*pdf_spin

#########################################################################################################
#
#priors
#
#########################################################################################################
#conversion function:
#Let the second subpopulations spin faster than the first, if exist.
#Each subpopulation has a mass range wider than 20 Msun.

def Double_constraint_GWTC4(params):
    try:
        params['mmax1']=fm(params['S300']) 
    except:
        pass
    params['pop1_scale']=np.sign(params['mmax1']-params['mmin1']-20)-1 
    params['pop2_scale']=np.sign(params['mmax2']-params['mmin2']-20)-1  
    try:
        params['mua2-mua1']=np.sign(params['mu_a2']-params['mu_a1'])-1  
    except:
        pass
    try:
        params['pop3_scale']=np.sign(params['mmax3']-params['mmin3']-20)-1   
    except:
        pass
    return params

#priors

def Double_nonpmact_priors(Double_constraint=Double_constraint_GWTC4):
    priors=bilby.prior.PriorDict(conversion_function=Double_constraint)
    priors.update(dict(
                    delta1=Uniform(1,10),
                    mmin1 = Uniform(2., 50., 'mmin1', '$m_{\\rm min,1}$'),
                    mmax1 = Uniform(20., 100, 'mmax1', '$m_{\\rm max,1}$'),
                    alpha1 = Uniform(-4, 8., 'alpha1', '$\\alpha,1$'),
                    amin1 = 0,
                    amax1 = Uniform(0.2,1),
                    
                    delta2=Uniform(1,20),
                    mmin2 = Uniform(2., 50., 'mmin2', '$m_{\\rm min,2}$'),
                    mmax2 = Uniform(20., 200, 'mmax2', '$m_{\\rm max,2}$'),
                    alpha2 = Uniform(-4., 8., 'alpha2', '$\\alpha_2$'),
                    amin2 = Uniform(0,0.8),
                    amax2 = 1,

                    r2 = Uniform(0,1, 'r2', '$r_2$'),

                    beta = Uniform(0,6,'beta','$\\beta$'),
                    lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
                    gamma= Uniform(-2,7,'gamma',r'$\gamma$')
                 ))
                     
    priors.update({key:bilby.prior.Constraint(minimum=-0.1, maximum=0.1) for key in ['pop1_scale','pop2_scale','mua2-mua1']})
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(15)})
    priors.update({'n1':0,'n'+str(15): 0})
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(10)})
    priors.update({'o1':0,'o'+str(10): 0})

    priors.update({'na'+str(i+1): TG(0,2,-100,100,name='na'+str(i+1))  for i in np.arange(5)})
    priors.update({'oa'+str(i+1): TG(0,2,-100,100,name='oa'+str(i+1))  for i in np.arange(5)})  
    priors.update({'oa1':-10,'na5': -10})

    priors.update({'nx'+str(i+1): TG(0,1,-100,100,name='nx'+str(i+1))  for i in np.arange(4)})
    priors.update({'t'+str(i+1): TG(0,1,-100,100,name='t'+str(i+1))  for i in np.arange(4)})
    return priors

#############################################################################################################################
#
#hyper prior
#
#############################################################################################################################

#Double
def hyper_Double_nonpmact(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,na1,na2,na3,na4,na5,amin1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,oa1,oa2,oa3,oa4,oa5,amin2,amax2,r2,beta,nx1,nx2,nx3,nx4,t1,t2,t3,t4,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_nonpmact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,na1,na2,na3,na4,na5,amin1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,oa1,oa2,oa3,oa4,oa5,amin2,amax2,r2,beta,nx1,nx2,nx3,nx4,t1,t2,t3,t4)*llh_z(z,gamma)
    return hp

#################################################################
#
#selection function
#
#################################################################

def p_a(a):
    xx=np.linspace(0,1,500)
    yy=np.exp(-2*xx**2)
    norm=np.sum(yy)*1./500.
    return np.exp(-2*a**2)/norm

def p_ct(ct):
    return 0.35+0.3*(1+ct)**3/4

def p_t(t):
    return np.sin(t)*p_ct(np.cos(t))

def p_phi(phi):
    return 1/2./np.pi

def lnp_spin(a1,a2,t1,t2,phi1,phi2):
    return np.log(p_a(a1))+np.log(p_a(a2))+np.log(p_t(t1))+np.log(p_t(t2))+np.log(p_phi(phi1))+np.log(p_phi(phi2))

path = inject_dir+"mixture-semi_o1_o2-real_o3_o4a-polar_spins_20250503134659UTC.hdf"
with h5py.File(path, "r") as f:
    Tobs=f.attrs['total_analysis_time']/(365.25*24*3600)
    Ndraw = f.attrs['total_generated']
    events = f["events"][:]
    meta = dict(f.attrs.items())

m1_inj = np.array(events['mass1_source'])
m2_inj = np.array(events['mass2_source'])
z_inj = np.array(events['redshift'])
a1_inj = np.array(events['spin1_magnitude'])
a2_inj = np.array(events['spin2_magnitude'])
t1_inj = np.array(events['spin1_polar_angle'])
t2_inj = np.array(events['spin2_polar_angle'])
phi1_inj = np.array(events['spin1_azimuthal_angle'])
phi2_inj = np.array(events['spin2_azimuthal_angle'])
ct1_inj = np.cos(t1_inj)
ct2_inj = np.cos(t2_inj)
weights = np.array(events["weights"])
ln_ws = np.log(weights)
min_far = np.min([events["%s_far"%search] for search in meta["searches"]], axis=0)
detected = min_far < 1.0 # /year
lnp_draw = np.array(events['lnpdraw_mass1_source_mass2_source_redshift_spin1_magnitude_spin1_polar_angle_spin1_azimuthal_angle_spin2_magnitude_spin2_polar_angle_spin2_azimuthal_angle'])
#lnp_draw = lnp_draw + ln_ws
lnp_draw = lnp_draw - ln_ws

sel_indx=np.where((m2_inj>2) & (m1_inj<300))

m1_inj=m1_inj[sel_indx]
m2_inj=m2_inj[sel_indx]
z_inj=z_inj[sel_indx]
a1_inj=a1_inj[sel_indx]
a2_inj=a2_inj[sel_indx]
ct1_inj=ct1_inj[sel_indx]
ct2_inj=ct2_inj[sel_indx]
t1_inj=t1_inj[sel_indx]
t2_inj=t2_inj[sel_indx]
phi1_inj=phi1_inj[sel_indx]
phi2_inj=phi2_inj[sel_indx]

detected=detected[sel_indx]
lnp_draw=lnp_draw[sel_indx]


log_pspin = lnp_spin(a1_inj,a2_inj,t1_inj,t2_inj,phi1_inj,phi2_inj)
logpdraw=lnp_draw-log_pspin+np.log(1./4.)

detection_selector = detected

log1pz_inj = np.log1p(z_inj)
logdVdz_inj = np.log(4*np.pi) + np.log(Planck15.differential_comoving_volume(z_inj).to(u.Gpc**3/u.sr).value)

#This selection effect accounts for spin distribution
def Rate_selection_function_with_uncertainty(Nobs,mass_spin_model,lgR0,gamma,**kwargs):
    log_dNdz = lgR0/np.log10(np.e) + (gamma-1)*log1pz_inj + logdVdz_inj
    log_dNdmds = np.log(mass_spin_model(m1_inj,m2_inj,a1_inj,a2_inj,ct1_inj,ct2_inj,**kwargs))
    log_dNdzdmds = np.where(detection_selector, log_dNdz+log_dNdmds, np.NINF)
    log_Nexp = np.log(Tobs) + np.logaddexp.reduce(log_dNdzdmds - logpdraw) - np.log(Ndraw)
    log_N_tot = log_N(Tobs,lgR0,gamma)
    term1 = Nobs*log_N_tot
    term2 = -np.exp(log_Nexp)
    selection=term1 + term2
    logmu=log_Nexp-log_N_tot 
    varsel= np.sum(np.exp(2*(np.log(Tobs)+log_dNdzdmds - logpdraw-log_N_tot- np.log(Ndraw))))-np.exp(2*logmu)/Ndraw
    total_vars=Nobs**2 * varsel / np.exp(2*logmu)
    Neff=np.exp(2*logmu)/varsel
    return selection, total_vars, Neff


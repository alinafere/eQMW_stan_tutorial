## Alina Ferecatu
## eQMW - November 2022
### Hierarchical Bayesian Binary Logit in STAN
### LOAD PACKAGES AND COMPILE STAN FUNCIONS #####
rm(list=ls())
library(stargazer)
library(tidyverse)
library(gridExtra)
library(loo)
library(ggpubr)
library(bayesplot)
library(rstan)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

PATH_FUNCTION="/set/your/path/here"
PATH_PLOTS="/set/your/path/here"
PATH_RESULTS="/set/your/path/here"

source(paste(PATH_FUNCTION,"/eQMW_functions.R", sep="") ) 
m = stan_model(model_code = hierarchical_binlogit_nocov)
m = stan_model(model_code = hierarchical_binlogit_fullcov)
m = stan_model(model_code = hierarchical_binlogit_fullcov_noncentered)

#### SIMULATE DATA #######
set.seed(66)
nvar=3                           ## number of coefficients
nlgt=500                        ## number of cross-sectional units
nobs=10                          ## number of observations per unit
nz=1                             ## number of regressors in mixing distribution

##* set hyper-parameters -----
## B=ZDelta + U
Z=as.matrix(rep(1,nlgt))
Delta=matrix(c(-2,-1, 0.5),nrow=nz,ncol=nvar)
iota=matrix(1,nrow=nvar,ncol=1)
Vbeta=.5*diag(nvar)+0.5*iota%*%t(iota)
cov2cor(Vbeta)

##* simulate individual-specific choices ----
longdata=NULL

for (i in 1:nlgt)
{
  betai=t(Delta)%*%Z[i,]+as.vector(t(chol(Vbeta))%*%rnorm(nvar))

  X=matrix(runif(nobs*(nvar-1)),nrow=nobs,ncol=(nvar-1))
  int=rep(1, nobs)
  X=cbind(int, X)

  Pb=exp(X %*% betai)/(1+exp(X %*% betai) )
  unif=runif(nobs,0,1)
  y=ifelse(unif<Pb,1,0)

  longdata[[i]]=list(betai = betai, y=y, X=X)
}

true_betas=NULL
for (i in 1:nlgt)
  true_betas=cbind(true_betas, longdata[[i]]$betai)
rowMeans(true_betas)

#####* Get the stan data set in the right format ###################
data_long=NULL
for (i in 1:nlgt)
  data_long=rbind(data_long,cbind(rep(i, nobs), 1:nobs, longdata[[i]]$y, longdata[[i]]$X))

colnames(data_long)=c("User_ID", "Observation", "Y", "Intercept", "Price", "Promotion")

dat=list(nvar=ncol(X),
              N=nrow(data_long),
              nind=nlgt,
              y=data_long[,3],
              x=data_long[,4:6],
              ind=data_long[,1])

#########  RUN THE STAN MODEL ####################
hbin_logit_Stan_fullcov1 <- stan(model_code = hierarchical_binlogit_fullcov, seed=9000,
                                            data = dat, chains = 3, iter = 100, warmup = 80,
                                            control=list(adapt_delta=0.9, max_treedepth=10))

setwd(PATH_RESULTS)
save(hbin_logit_Stan_nocov, file="hbin_logit_Stan_nocov.RData")

summary(hbin_logit_Stan_fullcov, pars=c("delta"), probs = c(0.025, 0.975))

posterior_nocov=as.array(hbin_logit_Stan_nocov)
posterior_fullcov=as.array(hbin_logit_Stan_fullcov)
posterior_fullcov_ncp=as.array(hbin_logit_Stan_fullcov_noncentered)

##########PLOTS#####################
pdens=mcmc_dens_overlay(posterior_fullcov, pars = c("delta[1]", "delta[2]", "delta[3]"))
pdens
mcmc_pairs(posterior_fullcov_ncp, pars = c("delta[1]", "delta[2]", "delta[3]"),
           off_diag_args = list(size = 1.5))

###* traceplots-----
color_scheme_set("mix-blue-red")
p1=mcmc_trace(posterior_nocov, pars = c("delta[1]", "delta[2]", "delta[3]"),
              facet_args = list(ncol = 1, strip.position = "left"))
p2=mcmc_trace(posterior_fullcov, pars = c("delta[1]", "delta[2]", "delta[3]"),
              facet_args = list(ncol = 1, strip.position = "left"))
p3=mcmc_trace(posterior_fullcov_ncp, pars = c("delta[1]", "delta[2]", "delta[3]"),
              facet_args = list(ncol = 1, strip.position = "left"))
a=ggarrange(p2, p3, ncol=2)


###### Individual parameters plots with bayesplot ######
dimnames(posterior_fullcov_ncp[,,1516:2015])
color_scheme_set("mix-blue-red")
mcmc_intervals(posterior_fullcov_ncp[,,1516:1615], point_est = "none", prob = 0.8, prob_outer = 0.95)+
  ggplot2::geom_point(aes(x=true_betas[1,1:100], y=1:100), alpha=1, size=0.5)+
  theme(axis.ticks.y = element_blank(), axis.text.y = element_blank())+
  labs(x = "True parameter values (black dots) and HDI",
       y = "Consumer 1 (top) to 100 (bottom)")

### posterior predictive checks based on y_rep ######
hbin_ncp_draws=rstan::extract(hbin_logit_Stan_fullcov_noncentered)
names(hbin_ncp_draws)
y=data_long[,3]
y_rep=hbin_ncp_draws$z

## plot of observed # of successes (T(y) - red vertical line), vs. posterior predictive replications of # of successes T(yrep)
## compute 95% confidence interval and p_value for obs T(y)
successes_plot=data.frame(sum_yrep=rowSums(y_rep)) %>%
  ggplot(aes(x=rowSums(y_rep))) +
  geom_histogram(binwidth=10, alpha=0.5) +
  geom_vline(xintercept = sum(y),colour="red")+
  theme_minimal()+labs(x="Number of successes", y="count")

## number of switches, to check whether there is correlation between the trials of the bernoulli
N=5000
ind=data_long[,1]
switch_y=rep(0, length(y))
i=2
while(i<N)
{
  if(ind[i]!=ind[i-1]) i=i+1
  if(y[i]!=y[i-1]) switch_y[i]=1
  i=i+1
}
b=cbind(switch_y, y, ind)

## number of switches in the posterior replications
## s is the number of iterations after burnin
hbin_ncp_draws=rstan::extract(hbin_logit_Stan_fullcov_noncentered)
y_rep=hbin_ncp_draws$z
S=3000
switch_yrep=matrix(rep(0, length(y)*S), nrow=S)
for (j in 1:S)
{
  i=2
  while(i<N)
  {
    if(ind[i]!=ind[i-1]) i=i+1
    if(y_rep[j,i]!=y_rep[j, i-1]) switch_yrep[j,i]=1
    i=i+1
  }
}
rowSums(switch_yrep)


switch_plot=data.frame(sum_switch_yrep=rowSums(switch_yrep)) %>%
  ggplot(aes(x=sum_switch_yrep)) +
  geom_histogram(binwidth=10, alpha=0.5) +
  geom_vline(xintercept = sum(switch_y),colour="red")+
  theme_minimal()+labs(x="Number of switches", y="count")

############# MODEL COMPARISON ######################
logl_fullcov=extract_log_lik(hbin_logit_Stan_fullcov)
logl_nocov=extract_log_lik(hbin_logit_Stan_nocov)
logl_ncp=extract_log_lik(hbin_logit_Stan_fullcov_noncentered)
loo1 <- loo(logl_nocov, save_psis = TRUE)
loo2 <- loo(logl_fullcov, save_psis = TRUE)
loo3 <- loo(logl_ncp, save_psis = TRUE)
loo_compare(loo3, loo2)

df=cbind(loo1$estimates, loo2$estimates, loo3$estimates)
stargazer(round(df,2), summary=F, align=F)

# S3 method for psis_loo
plot(loo3$psis_object,
  diagnostic = c("k", "n_eff"),
  label_points = FALSE,
  main = "PSIS diagnostic plot"
)
###* Examine choice of prior distributions #################
set.seed(999999)
nobs=100
prior_gamma = data.frame(obs=1:100, gamma_11=rgamma(nobs/2, 1, 1),
                         gamma_21=rgamma(nobs/2,2,1),
                         gamma_1half=rgamma(nobs/2, 2, 0.5))
colnames(prior_gamma)<-c("Obs", "Gamma(1,1)", "Gamma(2,1)", "Gamma(2,1/2)")
prior_gamma=prior_gamma %>% pivot_longer(-c("Obs")) %>% mutate(distr = "Gamma prior")
pg = prior_gamma %>%
  ggplot(aes(x=Obs, y=value))+
  geom_point()+
  scale_y_continuous(name="Gamma prior")+
  labs(x="")+theme_classic()+
  facet_wrap(.~ name, ncol=3)


prior_cauchy = data.frame(obs=1:100,  cauchy_01=rcauchy(nobs, 0,1),
                          cauchy_02=rcauchy(nobs, 0,2),
                          cauchy_05=rcauchy(nobs, 0,5))
colnames(prior_cauchy)<-c("Obs", "Half-Cauchy(0,1)", "Half-Cauchy(0,2)", "Half-Cauchy(0,5)")
prior_cauchy=prior_cauchy %>% pivot_longer(-c("Obs")) %>%
  mutate(distr = "Half-Cauchy prior") %>%
  filter(value>0)

pc = prior_cauchy %>%
  ggplot(aes(x=Obs, y=value))+
  geom_point()+
  scale_y_continuous(name="Half-Cauchy prior")+
  labs(x="")+theme_classic()+
  facet_wrap(.~ name, ncol=3)

a=ggarrange(pg, pc, nrow=2)
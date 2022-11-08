## Alina Ferecatu
## eQMW - November 2022
### Hierarchical Bayesian Binary Logit in STAN

################ Hierarchical model - full covariance matrix ###############
hierarchical_binlogit_fullcov="data {
int<lower=1> nvar; // number of parameters in the logit regression
int<lower=0> N; // number of observations
int<lower=1> nind; // number of individuals
int<lower=0,upper=1> y[N];
int<lower=1,upper=nind> ind[N]; // indicator for individuals
row_vector[nvar] x[N];
}

parameters {
vector[nvar] delta;
vector<lower=0>[nvar] tau;
vector[nvar] beta[nind];
corr_matrix[nvar] Omega; // Vbeta - prior correlation
}

model {
to_vector(delta) ~ normal(0, 5);
to_vector(tau) ~ gamma(2, 0.5);
Omega ~ lkj_corr(2);

for (h in 1:nind)
beta[h]~multi_normal(delta, quad_form_diag(Omega, tau));

for (n in 1:N)
y[n] ~ bernoulli_logit(x[n] * beta[ind[n]]);
}

generated quantities {
corr_matrix[nvar] Omega_corr;
int z[N];
real log_lik[N];

Omega_corr=Omega;
for (n in 1:N){
z[n] = bernoulli_logit_rng(x[n] * beta[ind[n]]);
log_lik[n]= bernoulli_logit_lpmf(y[n]|x[n] * beta[ind[n]]);
}

}
"

################ HB Full covariance noncentered reparametrization #############
hierarchical_binlogit_fullcov_noncentered="data {
int<lower=1> nvar; // number of parameters in the logit regression
int<lower=0> N; // number of observations
int<lower=1> nind; // number of individuals
int<lower=0,upper=1> y[N];
int<lower=1,upper=nind> ind[N]; // indicator for individuals
vector[nvar] x[N];
}

parameters {
matrix[nvar, nind] alpha; // nvar*H parameter matrix
row_vector[nvar] delta;
vector<lower=0>[nvar] tau;
cholesky_factor_corr[nvar] L_Omega;
}


transformed parameters{
row_vector[nvar] beta[nind];
matrix[nind,nvar] Vbeta_reparametrized;
Vbeta_reparametrized = (diag_pre_multiply(tau, L_Omega)*alpha)';

for (h in 1:nind)
beta[h]=delta+Vbeta_reparametrized[h];
}

model {
L_Omega~lkj_corr_cholesky(2);
to_vector(delta) ~ normal(0, 5);
to_vector(tau) ~ gamma(2, 0.5);
to_vector(alpha)~ normal(0,1);

for (n in 1:N)
y[n] ~ bernoulli_logit(beta[ind[n]]*x[n]);
}

generated quantities {
corr_matrix[nvar] Omega;
int z[N];
real log_lik[N];

Omega=L_Omega*L_Omega';

for (n in 1:N){
z[n] = bernoulli_logit_rng(beta[ind[n]]*x[n]);
log_lik[n]= bernoulli_logit_lpmf(y[n]|beta[ind[n]]*x[n]);
}

}
"

################ Hierarchical model - variance components only ###############
hierarchical_binlogit_nocov=
  "data {
int<lower=1> nvar; // number of parameters in the logit regression
int<lower=0> N; // number of observations
int<lower=1> nind; // number of individuals
int<lower=0,upper=1> y[N];
int<lower=1,upper=nind> ind[N]; // indicator for individuals
row_vector[nvar] x[N];
}

parameters {
real delta[nvar];
real<lower=0> tau[nvar];
vector[nvar] beta[nind];
}

model {
to_vector(delta) ~ normal(0,5);
to_vector(tau) ~ gamma(2, 0.5);
for (h in 1:nind)
beta[h] ~ normal(delta, tau);

for (n in 1:N)
y[n] ~ bernoulli_logit(x[n] * beta[ind[n]]);
}

generated quantities {
int z[N];
real log_lik[N];

for (n in 1:N){
z[n] = bernoulli_logit_rng(x[n] * beta[ind[n]]);
log_lik[n]= bernoulli_logit_lpmf(y[n]|x[n] * beta[ind[n]]);
}

}
"
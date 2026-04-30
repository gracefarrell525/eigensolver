#!/bin/bash

# parameters
h0=0.03 #0.03 munoz, 0.1 lee
qb=0.9 #0.9 munoz, 0.0 lee
beta=1.0 #1.0 munoz, 0.5 lee (1.25?)
#pindex=0.5 #will override beta and therefore p = 1.5-beta
use_boundary_factor=true #for 1-l0/sqrt(x)
l0=0.7 #0.7 munoz, 0.0 lee

xin=1.875 #1.875 munoz, 1.0e-4 lee
xout=100.0 #100.0 munoz, 50.0 lee

Rcav=2.5
use_cavity=true

Rout=100000.0 #100000 munoz, 1.0 lee
taper_power=0.0 #0 munoz, 1 lee
use_outer_taper=false 

# e0, eprime0, combo
inner_bc_kind=combo #combo munoz, eprime0 lee
outer_bc_kind=combo #combo munoz, eprime0 lee

#isothermal, adiabatic
thermo=isothermal
gamma=1.0 #1.0 munoz, 1.5 lee

nmodes=2
ngrid=1000

savefig=true
outfile="munoz_fig_11_trial.png"

# running pipeline file (which calls disk, eigensolver, and plotting files)
python3 NEW_pipeline.py \
  --thermo "$thermo" \
  --gamma "$gamma" \
  --use_boundary_factor "$use_boundary_factor" \
  --h0 "$h0" \
  --qb "$qb" \
  --beta "$beta" \
  --l0 "$l0" \
  --xin "$xin" \
  --xout "$xout" \
  --Rcav "$Rcav" \
  --use_cavity "$use_cavity" \
  --Rout "$Rout" \
  --taper_power "$taper_power" \
  --use_outer_taper "$use_outer_taper" \
  --inner_bc_kind "$inner_bc_kind" \
  --outer_bc_kind "$outer_bc_kind" \
  --nmodes "$nmodes" \
  --ngrid "$ngrid" \
  --savefig "$savefig" \
  --outfile "$outfile"

#--pindex "$pindex"

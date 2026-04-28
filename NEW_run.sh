#!/bin/bash

# parameters
h0=0.1 #0.03 munoz, 0.1 lee
qb=0.0 #0.9 munoz, 0.0 lee
beta=0.5 #1.0 munoz, 0.5 lee (1.25?)
l0=0.0 #0.7 munoz, 0.0 lee

xin=1.0e-4 #1.875 munoz, 1.0e-4 lee
xout=50.0 #100.0 munoz, 50.0 lee

Rcav=2.5
use_cavity=false

Rout=1.0 #100000 munoz, 1.0 lee
taper_power=1.0 #0 munoz, 1 lee
use_outer_taper=true 

# e0, eprime0, combo
inner_bc_kind=eprime0 #combo munoz, eprime0 lee
outer_bc_kind=eprime0 #combo munoz, eprime0 lee

nmodes=3
ngrid=1000

savefig=true
outfile="lee_fig_1_trial.png"

# running pipeline file (which calls disk, eigensolver, and plotting files)
python3 NEW_pipeline.py \
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

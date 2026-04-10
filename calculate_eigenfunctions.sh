#!/bin/bash
# .sh, shell script, dump it onto the terminal all at once
l0=0.001
h0=0.03
Rcav=1e-3
qb=0.9
Rout=1
xout=50
n=2 # not guaranteed first 20 modes... try mode modes? 200? 
taper=1

python3 eigenvalue_shooting-E.py $h0 $qb $Rcav $l0 $Rout $xout $n $taper
python double_panel_plot.py $h0 $qb $Rcav $l0 $Rout $xout $n $taper
#!/bin/bash

for seed in {1..300}
do
	for omega in 0.1 0.2 0.3
	do
		echo $seed 
		Rscript generate.R  $seed 5 10 $omega
		done
done

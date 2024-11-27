# Byzantine-tolerant distributed learning of finite mixture models

The folder contains the code for both the simulated and real data analysis in the paper entitled **Byzantine-tolerant distributed learning of finite mixture models**. 
There are two subdirectories named real_data and simulation are the code for reproducing the simulation and the real data experiment in the paper respectively. 

## Requirements
The code is written in Python 3.8.10. Package dependencies are listed in requirements.txt. To install the packages, run


```
pip install -r requirements.txt
```

## Simulation

To run the simulation, you first need to install the ``mixtools`` package in R and run the following code.

```
cd simulation/generated_pop
run pop_generate.sh
```

This will produce the parameter values for the 300 repetitions in our experiment. The parameter values are stored in ``txt`` files under the ``generated_pop/true_param`` folder.


Then you can run the simulations by first fit local by running `global_local.py` file and then `simulation.py' file.
The following is the demo code when $n=5000$, $m=20$, MaxOmega=0.1, and under weight attack.

```
cd simulation
python global_local.py --ss 100000 --seed 1 --overlap 0.1 --n_split 20 
python simulation.py --ss 100000 --seed 1 --attack_mode 3 --overlap 0.1 --n_split 20
```
If you want to run the experiment for component-wise failure, you can run the following
```
python simulation.py --ss 100000 --seed 1 --attack_mode 1 --overlap 0.1 --n_split 100 --failure_type component
```


The output will be stored in a pickle file under output/save_data directory.
Then you can load the pickle file to post-process the simulation results.



## Real Data
The ``real_data`` folder contains the check point of the pretrained NN, you can also train it from scratch using `nn_feature_extractor.py` by following the instructions of the README file under real_data.


The NIST folder contains the code for our experiment.
To run an experiment, you can simply run 
```
python nist.py --local_ss 5000 --seed 1
```
Similarly, you can run the code for component-wise failure as follows:
```
python nist.py --local_ss 5000 --seed 1 --n_split 20
```

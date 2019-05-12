SETUP:
Install the following packages using these commands:

apt-get install fceux
pip3 install ppaquette-gym-super-mario
pip3 install neat-python
pip3 install numpy

Run the model using:
python3 script.py run
to run the trained genome

Train using:
python3 script.py train [genCount]
where genCount is the number of generations to train. After training, best_genome file will be stored.

Specify the file to run, the number of parallel computing level and the game mario level to train and run on by the first few lines of the script file.


References:
1. Open AI Gym Environment: https://github.com/vivek3141/super-mario-neat
2. Neat-Python library: https://neat-python.readthedocs.io/en/latest/
3. Mario Neat Implementation: https://github.com/vivek3141/super-mario-neat


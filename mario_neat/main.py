import argparse
import neat
import pickle
import gym, ppaquette_gym_super_mario
import gzip
import neat.genome
import os
import sys
import multiprocessing as mp


gym.logger.set_level(40)
configFile = 'config'
runFile= 'trained'
level="1-1"
mpLevel=4



actions = [
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1],
]

actions_alt= [
[0, 0, 0, 1, 0, 1],
[0, 0, 0, 1, 1, 1],
[0, 0, 0, 0, 1, 1],
[0, 1, 0, 0, 0, 1],
[0, 0, 0, 1, 0, 0],
]

def marioPos(state):
	for i in range(len(state)):
		for j in range(len(state[i])):
			if (state[i][j]==3):
				if (j<=14):
					if (state[i][j+1]==3):
						return [i,j+1]
				return [i,j]
	return [0,0]


def toStr(arr):
	key= ""
	for i in range(len(arr)):
		for j in range(len(arr[i])):
			key=key+str(arr[i][j])
	return key


def predEnemMove(enemPos1, enemyPos2,numSteps):
	x1=enemPos1[0]
	y1=enemPos1[1]
	x2= enemPos2[0]
	y2= enemPos2[1]
	enemVelX=(x1-x2)/numSteps
	enemVelY=(y1-y2)/numSteps
	return (int(x2+enemVelX),int(y2+enemVelY))


# def evaluate(state, info):
# 	state= state.reshape((13,16))
# 	marioPos= marioPos(state)
# 	onGround= onGround(state,marioPos)
# 	if (onGround==0):
# 		return 0
# 	if (info['player_status']!=0)



def runModel(config_file, file, level="1-1"):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    genome = pickle.load(open(file, 'rb'))
    env = gym.make('ppaquette/SuperMarioBros-'+level+'-Tiles-v0')
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    info = {'distance': 0}
    try:
        state = env.reset()
        done = False
        i = 0
        prev_dist=0
        while not done:
            state = state.reshape(208)
            output = net.activate(state)
            ind = output.index(max(output))
            s, reward, done, info = env.step(actions[ind])
            state = s
            i += 1
            if i % 100 == 0:
                    if prev_dist >= info['distance']:
                        break
                    else:
                        prev_dist = info['distance']
        print("Distance: {}".format(info['distance']))
        env.close()
    except KeyboardInterrupt:
        env.close()
        exit()



class Train:
    def __init__(self, generations, parallel=2, level="1-1"):
        self.generations = generations
        self.lock = mp.Lock()
        self.par = parallel
        self.level = level

    def _get_actions(self, a):
        return actions[a.index(max(a))]

    def _fitness_func(self, genome, config, o):
        env = gym.make('ppaquette/SuperMarioBros-'+self.level+'-Tiles-v0')
        try:
            state = env.reset()
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            done = False
            i = 0
            prev_dist = 0
            while not done:
                state = state.flatten()
                output = net.activate(state)
                output = self._get_actions(output)
                s, reward, done, info = env.step(output)
                state = s
                i += 1
                if i % 100 == 0:
                    if prev_dist >= info['distance']:
                        break
                    else:
                        prev_dist = info['distance']

            fitness = -1 if info['distance'] <= 40 else info['distance']
            if fitness >= 3250:
                pickle.dump(genome, open("best_genome", "wb"))
                env.close()
                print("Done")
                exit()
            o.put(fitness)
            env.close()
        except KeyboardInterrupt:
            env.close()
            exit()

    def _eval_genomes(self, genomes, config):
        index, genomes = zip(*genomes)

        for i in range(0, len(genomes), self.par):
            mpOutput = mp.Queue()

            processes = [mp.Process(target=self._fitness_func, args=(genome, config, mpOutput)) for genome in
                         genomes[i:i + self.par]]

            [proc.start() for proc in processes]
            [proc.join() for proc in processes]

            results = [mpOutput.get() for proc in processes]
		
            for n, r in enumerate(results):
                print ("Fitness Value of genome: ", end="")
                print (r)
                genomes[i + n].fitness = r

    def _run(self, config_file, n):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(True))
        pop.add_reporter(neat.Checkpointer(5))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        winner = pop.run(self._eval_genomes, n)
        pickle.dump(winner, open('best_genome', 'wb'))
        

    def main(self, config_file='config'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self._run(config_path, self.generations)



if (len(sys.argv)<2):
	print ("Invalid number of input arguments. Specify train [generationCount] or run")
else:
	arg1= sys.argv[1]
	if (arg1=="run"):
		runModel(configFile,runFile,level)
	elif (arg1=="train"):
		if (len(sys.argv)<3):
			print ("Specify [generationCount]")
		else:
			genCount= int(sys.argv[2])
			t=Train(genCount, mpLevel,level)
			t.main("config")





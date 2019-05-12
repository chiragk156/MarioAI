import gym, ppaquette_gym_super_mario, os
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy as np

class marioModel:

	def __init__(self):
		self.modelFile = "model.h5"
		self.actions = [
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 1],
		[0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 1, 1],
		[0, 0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0, 1],
		[0, 0, 0, 1, 1, 0],
		[0, 0, 0, 1, 1, 1],
		[0, 0, 1, 0, 0, 0],
		[0, 0, 1, 0, 0, 1],
		[0, 0, 1, 0, 1, 0],
		[0, 0, 1, 0, 1, 1],
		[0, 0, 1, 1, 0, 0],
		[0, 0, 1, 1, 0, 1],
		[0, 0, 1, 1, 1, 0],
		[0, 0, 1, 1, 1, 1],
		[0, 1, 0, 0, 0, 0],
		[0, 1, 0, 0, 0, 1],
		[0, 1, 0, 0, 1, 0],
		[0, 1, 0, 0, 1, 1],
		[0, 1, 0, 1, 0, 0],
		[0, 1, 0, 1, 0, 1],
		[0, 1, 0, 1, 1, 0],
		[0, 1, 0, 1, 1, 1],
		[0, 1, 1, 0, 0, 0],
		[0, 1, 1, 0, 0, 1],
		[0, 1, 1, 0, 1, 0],
		[0, 1, 1, 0, 1, 1],
		[0, 1, 1, 1, 0, 0],
		[0, 1, 1, 1, 0, 1],
		[0, 1, 1, 1, 1, 0],
		[0, 1, 1, 1, 1, 1]]

		self.lastScore = 0

		if os.path.isfile('model.h5'):
			self.model = load_model('model.h5')
			self.checkpoint = ModelCheckpoint('model.h5', monitor = 'loss')
			self.callbacks_list = [self.checkpoint]

		else:
			self.model = Sequential()
			self.model.add(Dense(212, input_dim=210, activation='relu'))
			self.model.add(Dense(100, activation='relu'))
			self.model.add(Dense(50, activation='relu'))
			self.model.add(Dense(1, activation='elu'))
			self.model.compile(loss='mse', optimizer='adam')
			self.checkpoint = ModelCheckpoint('model.h5', monitor = 'loss')
			self.callbacks_list = [self.checkpoint]

	def getReward(self, currentScore, lastScore):
		return currentScore - lastScore

	def getScore(self, info):
		score = 0
		score += info['distance']
		score += info['score']
		score += info['coins']*5
		if info['time'] == 0:
			return -1000
		score += 1000.0/info['time']
		score += info['player_status']*50
		if info['life'] == 0:
			score -= 1000

		return score

	def makeInputArray(self, s, a, player_status):
		temp = np.zeros((1,210))
		temp[0,0:208] = s.flatten()
		temp[0,208]= a
		temp[0, 209] = player_status
		return temp

	def train(self):
		level="1-1"
		env = gym.make('ppaquette/SuperMarioBros-'+level+'-Tiles-v0')
		env.reset()
		lastDis = 0
		try:
			first = True
			while True:
				if first:
					s, reward, done, info = env.step(self.actions[4])
					score = self.getScore(info)
					reward = self.getReward(score,self.lastScore)
					self.lastScore = score
					lastDis = info['distance']
					first = False
					self.model.fit(self.makeInputArray(s,6,info['player_status']),[reward],epochs=1, callbacks=self.callbacks_list)
					continue
				
				qValues = []
				for i in range(32):
					qValues.append(self.model.predict(self.makeInputArray(s,i,info['player_status'])))
				action = qValues.index(max(qValues))

				s, reward, done, info = env.step(self.actions[action])
				if lastDis == info['distance']:
					s1, reward1, done, info1 = env.step(self.actions[6])
					s1, reward1, done, info1 = env.step(self.actions[6])
					s1, reward1, done, info1 = env.step(self.actions[6])
					s1, reward1, done, info1 = env.step(self.actions[6])

				score = self.getScore(info)
				reward = self.getReward(score,self.lastScore)
				self.lastScore = score
				lastDis = info['distance']
				self.model.fit(self.makeInputArray(s,action,info['player_status']),[reward],epochs=1,callbacks=self.callbacks_list)
				self.model.save('model.h5')
				if done:
					env.close()
					env = gym.make('ppaquette/SuperMarioBros-'+level+'-Tiles-v0')
					env.reset()
					first = True
					print('d')
				
		except KeyboardInterrupt:
			env.close()
			self.model.save('model.h5')
			exit()
		
		env.close()


if __name__ == "__main__":
	m = marioModel()
	m.train()
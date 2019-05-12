import gym, ppaquette_gym_super_mario
import itertools 
import matplotlib 
import matplotlib.style 
import numpy as np 
import pandas as pd 
import sys 
import plotting
from collections import defaultdict

actions= [
[0, 0, 0, 1, 0, 1],
[0, 0, 0, 1, 1, 1],
[0, 0, 0, 0, 1, 1],
[0, 1, 0, 0, 0, 1],
[0, 0, 0, 1, 0, 0],
]


def _get_actions(a):
	return actions[a.index(max(a))]

def toStr(arr):
	key= ""
	for i in range(len(arr)):
		for j in range(len(arr[i])):
			key=key+str(arr[i][j])
	return key


def marioPos(state):
	for i in range(len(state)):
		for j in range(len(state[i])):
			if (state[i][j]==3):
				if (j<=14):
					if (state[i][j+1]==3):
						return [i,j+1]
				return [i,j]
	return [0,0]


def nearEnem(state, marioPos):
	if (marioPos[1]-2<0 or marioPos[1]+2>15):
		return 0
	for i in range(marioPos[1],marioPos[1]+4):
		if (state[marioPos[0]][i]==2):
			return 1

def predEnemMove(enemPos1, enemyPos2,numSteps):
	x1=enemPos1[0]
	y1=enemPos1[1]
	x2= enemPos2[0]
	y2= enemPos2[1]
	enemVelX=(x1-x2)/numSteps
	enemVelY=(y1-y2)/numSteps
	return (int(x2+enemVelX),int(y2+enemVelY))


def farEnem(state, marioPos):
	if (marioPos[1]-4<0 or marioPos[1]+4>15):
		return 0
	for i in range(marioPos[1]-4,marioPos[1]-2):
		if (state[marioPos[0]][i]==2):
			return 1
	for i in range(marioPos[1]+3,marioPos[1]+5):
		if (state[marioPos[0]][i]==2):
			return 1
def onGround(state, marioPos):
	if (marioPos[0]==12):
		return 0
	if (state[marioPos[0]+1][marioPos[1]]==0):
		return 0
	else:
		return 1
def platJump(state, marioPos, ground):
	if (marioPos[0]>=12 or ground==0 or marioPos[1]>=13):
		return [0,0,0]
	else:
		li= [0,0,0]
		for i in range(0,3):
			if (state[marioPos[0]+1][marioPos[1]+1+i]==0):
				li[i]=1
	return li

def nearObs(state, marioPos):
	if (marioPos[1]==15):
		return [0,0,0,0]
	else:
		li= [0,0,0,0]
		for i in range(0,4):
			if (state[marioPos[0]-i][marioPos[1]+1]==1):
				li[i]=1
		return li


def nextAction(state):
	moveCounter=1
	moveList=[0]
	
	# print (state)
	mario= marioPos(state)
	# print (mario)
	ground = onGround(state,mario)
	obs= nearObs(state,mario)
	plat= platJump(state, mario, ground)
	upPos=obs
	if (upPos[0]==1):
		moveCounter=3
		moveList=[2,2]
		for i in range(1,4):
			if (upPos[i]!=1):
				break
			else:
				moveList.append(2)
				moveList.append(2)
				moveCounter = moveCounter+2
		moveList.append(1)
	rightPos= plat
	if (rightPos[0]==1):
		moveCounter=1
		moveList=[1]
		for i in range(1,3):
			if (rightPos[i]!=1):
				break
			else:
				moveList.append(1)
				moveCounter = moveCounter+1

	enem= nearEnem(state,mario)
	if (enem==1):
		# print (state)
		# print ("HI")
		moveCounter=2
		moveList=[2,0]
	if (state[mario[0]][mario[1]+2]==2) or (state[mario[0]][mario[1]-1]==2):
		moveCounter=2
		moveList=[1,0]
	return moveCounter, moveList

# def evaluate(state, info):
# 	state= state.reshape((13,16))
# 	marioPos= marioPos(state)
# 	onGround= onGround(state,marioPos)
# 	if (onGround==0):
# 		return 0
# 	if (info['player_status']!=0)




level="1-1"

env = gym.make('ppaquette/SuperMarioBros-'+level+'-Tiles-v0')
state= env.reset()
state= state.flatten()
done = False
k=0
moveCounter=0
prevStateArr=[[0],[0]]
stateArr= state.reshape((13,16))
nextCounter=0
try:
	while not done:
		if (moveCounter==0):
			moveCounter, moveList= nextAction(stateArr)
		nextCounter=nextCounter+1
		nextMove=moveList[moveCounter-1]
		moveCounter=moveCounter-1

		action= actions[nextMove]
		s, reward, done, info = env.step(action)
		if (nextCounter==10):
			if (toStr(stateArr)==toStr(prevStateArr)):
				moveCounter=1
				moveList=[1]
			else:
				prevStateArr=stateArr.copy()
			nextCounter=0
		stateArr=s.reshape((13,16))
		
except KeyboardInterrupt:
	env.close()
	exit()
env.close()
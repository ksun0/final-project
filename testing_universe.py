import gym
import universe  # register the universe environments
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output
import random
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import load_model

epochs = 0
gamma = 0.9 #since it may take several moves to goal, making gamma high
epsilon = 1
games = 0
done_n = [False]
DEATH_COST = -10
LOAD = True
PLAY_AFTER = True

def makeMove(state, action):
    mousePositions = []
    for i in range(8):
        mousePositions.append((100 * np.cos(2 * np.pi / 8 * i), 100 * np.sin(2 * np.pi / 8 * i)))
    if action < 8:
        action_n = [[('PointerEvent', mousePositions[int(action)//2][0] + 265, mousePositions[int(action)//2][1] + 235, False)]]
    else:
        action_n = [[('PointerEvent', mousePositions[int(action)//2][0] + 265, mousePositions[int(action)//2][1] + 235, True)]]
    return env.step(action_n)

def simplify(data):
    data = np.array(data)[0:530,0:470,0:3]
    data = rgb2gray(data)
    data = resize(data, (265, 235))
    return np.array(data)

if not LOAD:

    model = Sequential()
    model.add(Dense(164, init='lecun_uniform', input_shape=(265*235,)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

    model.add(Dense(150, init='lecun_uniform'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))

    model.add(Dense(16, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)\

else:
    model = load_model('model250.h5')

env = gym.make('internet.SlitherIO-v0')
env.configure(remotes=1)  # automatically creates a local docker container
state = env.reset()

while games < epochs:

    while True: #we need to call an action to get the state to update
        action_n = [[('PointerEvent', 200, 200, False)]]
        state, reward_n, done_n, info = env.step(action_n)
        env.render()

        try:
            state[0]['vision']
            break
        except:
            pass

    state = simplify(state[0]['vision'])

    #while game still in progress
    while(not done_n[0]):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,530*470/4), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,16)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state, reward, done_n, info = makeMove(state, action)
        if done_n[0]:
            new_state = state
        else:
            new_state = simplify(new_state[0]['vision'])
        env.render()
        #Observe reward

        #Get max_Q(S',a)
        newQ = model.predict(new_state.reshape(1,530*470/4), batch_size=1)
        maxQ = np.max(newQ)
        y = np.zeros((1,16))
        y[:] = qval[:]
        if not done_n[0]: #non-terminal state
            update = ((reward[0] - 0.1) + (gamma * maxQ))
        else: #terminal state
            update = (DEATH_COST + (gamma * maxQ))
        y[0][action] = update #target output
        print("Game #: %s" % (games,))
        model.fit(state.reshape(1,530*470/4), y, batch_size=1, nb_epoch=1, verbose=1)
        state = new_state
        clear_output(wait=True)



    if epsilon > 0.1:
        epsilon -= (1/(epochs)) #we may want to change this later

    games += 1

    if games % 10 == 0:
        model.save('model250.h5')

while PLAY_AFTER:

    while True: #we need to call an action to get the state to update
        action_n = [[('PointerEvent', 200, 200, False)]]
        state, reward_n, done_n, info = env.step(action_n)
        env.render()

        try:
            state[0]['vision']
            break
        except:
            pass

    state = simplify(state[0]['vision'])

    #while game still in progress
    while(not done_n[0]):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,530*470/4), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,16)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        print("action: ", action, "eps: ", epsilon)
        new_state, reward, done_n, info = makeMove(state, action)
        if done_n[0]:
            new_state = state
        else:
            new_state = simplify(new_state[0]['vision'])
        env.render()
        state = new_state
        clear_output(wait=True)

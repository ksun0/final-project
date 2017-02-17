import gym
import universe  # register the universe environments
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output
import random
import numpy as np

def makeMove(state, action):
    mousePositions = []
    for i in range(8):
        mousePositions.append((100 * np.cos(2 * np.pi / 8 * i), 100 * np.sin(2 * np.pi / 8 * i)))
    if action < 8:
        action_n = [[('PointerEvent', mousePositions[int(action)//2][0] + 265, mousePositions[int(action)//2][1] + 235, False)]]
    else:
        action_n = [[('PointerEvent', mousePositions[int(action)//2][0] + 265, mousePositions[int(action)//2][1] + 235, True)]]
    return env.step(action_n)

try:
    model = load_model('my_first_model.h5')
    print("model loaded")
except:
    model = Sequential()
    model.add(Dense(164, init='lecun_uniform', input_shape=(2359296,)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

    model.add(Dense(150, init='lecun_uniform'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))

    model.add(Dense(16, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

env = gym.make('internet.SlitherIO-v0')
env.configure(remotes=1)  # automatically creates a local docker container
state = env.reset()

DEATH_COST = -10

epochs = 50
gamma = 0.9 #since it may take several moves to goal, making gamma high
epsilon = 1
games = 0
done_n = [False]

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

    state = np.array(state[0]['vision'])

    #while game still in progress
    while(not done_n[0]):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,2359296), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,16)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state, reward, done_n, info = makeMove(state, action)
        if done_n[0] or not new_state[0]['vision']:
            new_state = state
        else:
            new_state = np.array(new_state[0]['vision'])
        env.render()
        #Observe reward

        #Get max_Q(S',a)
        newQ = model.predict(new_state.reshape(1,2359296), batch_size=1)
        maxQ = np.max(newQ)
        y = np.zeros((1,16))
        y[:] = qval[:]
        if not done_n[0]: #non-terminal state
            update = ((reward[0] - 0.1) + (gamma * maxQ))
        else: #terminal state
            update = (DEATH_COST + (gamma * maxQ))
        y[0][action] = update #target output
        print("Game #: %s" % (games,))
        model.fit(state.reshape(1,2359296), y, batch_size=1, nb_epoch=1, verbose=1)
        state = new_state
        clear_output(wait=True)



    if epsilon > 0.1:
        epsilon -= (1/(epochs)) #we may want to change this later

    games += 1

model.save('my_first_model.h5')

while True:

    while True: #we need to call an action to get the state to update
        action_n = [[('PointerEvent', 200, 200, False)]]
        state, reward_n, done_n, info = env.step(action_n)
        env.render()

        try:
            state[0]['vision']
            break
        except:
            pass

    state = np.array(state[0]['vision'])

    #while game still in progress
    while(not done_n[0]):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,2359296), batch_size=1)
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
            new_state = np.array(new_state[0]['vision'])
        env.render()
        state = new_state
        clear_output(wait=True)

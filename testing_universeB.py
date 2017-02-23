# CS630: Machine Learning Final Project
# Lior Hirschfeld, Jihoun Im, Kevin Sun, and Henry Desai
# IMPORTS
import gym
import universe  # register the universe environments
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
from IPython.display import clear_output
import random
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import load_model

# HYPERPARAMATERS
epochs = 11             # Sets the number of games the model trains on.
gamma = 0.9             # Sets the amount the model consider future reward.
epsilon = 1             # Sets exploration vs exploitation value.
DEATH_COST = -1/600     # Sets how much should the bot be harmed for dying.
LOAD = False            # Sets whether a previous model should be loaded.
PLAY_AFTER = True       # Sets whether the model plays after it finishes training.

#HELPER METHODS
def makeMove(state, action):
    # This method takes an action and moves the mouse accordingly.
    mousePositions = []
    for i in range(8):
        mousePositions.append((100 * np.cos(2 * np.pi / 8 * i), 100 * np.sin(2 * np.pi / 8 * i)))
    if action < 8:
        action_n = [[('PointerEvent', mousePositions[int(action)//2][0] + 265, mousePositions[int(action)//2][1] + 235, False)]]
    else:
        action_n = [[('PointerEvent', mousePositions[int(action)//2][0] + 265, mousePositions[int(action)//2][1] + 235, True)]]
    return env.step(action_n)

def simplify(data):
    # This method simplifies the data received from Universe to something managable.
    data = np.array(data)[0:530,0:470,0:3] # Ignore all but the game screen.
    data = rgb2gray(data) # Collapse RGB
    return np.array(data)

if not LOAD:
    # Construct a new model.
    # The size an # of hidden layers can be messed with here.
    # We found that these numbers worked moderately well on the GPU.
    model = Sequential()

    model.add(Convolution2D(16, 10, 10, border_mode='same', input_shape=(530, 470, 1)))
    model.add(Convolution2D(8, 5, 5, border_mode='same'))

    model.add(Flatten())
    model.add(Dense(150, init='lecun_uniform'))
    model.add(Activation('relu'))

    model.add(Dense(75, init='lecun_uniform'))
    model.add(Activation('relu'))

    model.add(Dense(16, init='lecun_uniform'))
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

else:
    # Load an existing, trained model.
    model = load_model('model250.h5')

#INITIALIZE UNIVERSE ENVIRONMENT
games = 0 #always init. to 0
done_n = [False]
env = gym.make('internet.SlitherIO-v0')
env.configure(remotes=1)
state = env.reset()

while games < epochs:
    rounds = 0 #keep track of how long snake is alive

    while True: #Wait for Universe to finish initializing.
        action_n = [[('PointerEvent', 200, 200, False)]]
        state, reward_n, done_n, info = env.step(action_n)
        env.render()

        try:
            state[0]['vision']
            break
        except:
            pass

    state = simplify(state[0]['vision']) #

    while not done_n[0]: #While the game is still in progress

        # Store expected value for all possible actions.
        qval = model.predict(state.reshape(1, 530, 470, 1), batch_size=1)

        # Take a random or the predicted best action.
        if (random.random() < epsilon):
            action = np.random.randint(0,16)
        else:
            action = (np.argmax(qval))
        #Take an action and observe the new state
        new_state, reward, done_n, info = makeMove(state, action)
        if done_n[0]:
            new_state = state
        else:
            new_state = simplify(new_state[0]['vision'])

        # Rerender the screen. This can be skipped if the
        # developer doesn't care about watching the snake.
        env.render()

        newQ = model.predict(new_state.reshape(1, 530, 470, 1), batch_size=1)
        maxQ = np.max(newQ)
        y = np.zeros((1,16))
        y[:] = qval[:]
        if not done_n[0]: #non-terminal state
            update = ((reward[0] - 0.1) + (gamma * maxQ)) #algorithm for q-learning
            if action >= 8:
                update -= 0.1 #Penalize for boosting
        else: #terminal state
            update = (DEATH_COST * rounds + (gamma * maxQ))
        y[0][action] = update
        model.fit(state.reshape(1, 530, 470, 1), y, batch_size=1, nb_epoch=1, verbose=1)
        state = new_state
        clear_output(wait=True)

        rounds += 1


    if epsilon > 0.1: #decreases the value of epsilon because as the model learns more, it should be taking less random actions
        epsilon -= (1/(epochs))

    games += 1

    if games % 10 == 0:
        model.save('model250.h5')

while PLAY_AFTER:
    while True: # Wait for Universe to finish initializing.
        action_n = [[('PointerEvent', 200, 200, False)]]
        state, reward_n, done_n, info = env.step(action_n)
        env.render()

        try:
            state[0]['vision']
            break
        except:
            pass

    state = simplify(state[0]['vision'])

    while(not done_n[0]): #while game is still in progress
        qval = model.predict(state.reshape(1, 530, 470, 1), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,16)
        else:
            action = (np.argmax(qval))
        new_state, reward, done_n, info = makeMove(state, action)
        if done_n[0]:
            new_state = state
        else:
            new_state = simplify(new_state[0]['vision'])
        env.render()
        state = new_state
        clear_output(wait=True)

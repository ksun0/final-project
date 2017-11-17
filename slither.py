# CS630: Machine Learning Final Project
# Lior Hirschfeld, Jihoun Im, Kevin Sun, and Henry Desai
# IMPORTS
import gym
import universe  # register the universe environments
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output
import random
import numpy as np
import pickle
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import load_model

# HYPERPARAMATERS
DEATH_COST = -1/600     # Sets how much should the bot be harmed for dying.
LOAD = False            # Sets whether a previous model should be loaded.
PLAY_AFTER = True       # Sets whether the model plays after it finishes training.
BATCH_SIZE = 10         # Sets the batch size for catastrophic forgetting defence.
LEARNING_RATE=0.0001    # Sets the speed at which the model learns.
EQUALIZE = False        # Sets whether the model should EQUALIZE rewards.
epochs = 3              # Sets the number of games the model trains on.
gamma = 0.9             # Sets the amount the model consider future reward.
epsilon = 1             # Sets exploration vs exploitation value.

# HELPER METHODS
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
    data = rgb2gray(data)                  # Collapse RGB
    data = resize(data, (53, 47))          # BIN
    return np.array(data)

if not LOAD:
    # Construct a new model.
    # The size an # of hidden layers can be messed with here.
    # We found that these numbers worked moderately well on the GPU.
    model = Sequential()
    model.add(Dense(2000, init='uniform', input_shape=(53*47,)))
    model.add(Activation('tanh'))

    model.add(Dense(16, init='uniform'))
    model.add(Activation('linear'))
    rms = RMSprop()

    model.compile(loss='mse', optimizer=RMSprop(lr=LEARNING_RATE))

else:
    # Load an existing, trained model.
    model = load_model('model250.h5')

# Store data recording the success of each model run in a list.
try:
	pelletsEarnedList = pickle.load(open('pelletsearned.p', 'rb'))
	pelletsEarnedList.append([])
except:
	pelletsEarnedList = [[]]

# INITIALIZE UNIVERSE ENVIRONMENT
games = 0
done_n = [False]
env = gym.make('internet.SlitherIO-v0')
env.configure(remotes=1)  # automatically creates a local docker container
state = env.reset()

while games < epochs:
    pelletsEarned = 0 # Keeps track of # of pellets earned in this epoch.
    rounds = 0        # Records the number of frames the model has learned.
    replay = []       # Stores a list of old state, rewards, and actions
    buff = 80         # Sets the max length of replay.
    h = 0
    while True: #Wait for Universe to finish initializing.
        action_n = [[('PointerEvent', 200, 200, False)]]
        state, reward_n, done_n, info = env.step(action_n)
        env.render()

        try:
            state[0]['vision']
            break
        except:
            pass

    state = simplify(state[0]['vision'])

    while not done_n[0]: # While the game is still in progress.

        # Store expected value for all possible actions.
        qval = model.predict(state.reshape(1,53*47), batch_size=1)

        # Take a random or the predicted best action.
        if (random.random() < epsilon):
            action = np.random.randint(0,16)
        else:
            action = (np.argmax(qval))
        #Take an action and observe the new state.
        new_state, reward, done_n, info = makeMove(state, action)
        pelletsEarned += reward[0]

        if done_n[0]:
            new_state = state
            reward[0] = None
        else:
            new_state = simplify(new_state[0]['vision'])

        # Rerender the screen. This can be skipped if the
        # developer doesn't care about watching the snake.
        env.render()

        if (len(replay) < buff):
            # While the buffer is being filled, don't train.
            replay.append((state, action, reward, new_state))
        else:
            # Now, take a random selection of replay and train off of that.
            if (h < (buff - 1)):
                h += 1
            else:
                h = 0
            replay[h] = (state, action, reward, new_state) # Add current situation to replays.
            minibatch = random.sample(replay, BATCH_SIZE)
            X_train = []
            Y_train = []

            for memory in minibatch:
                old_state, action, reward, new_state = memory
                oldQ = model.predict(old_state.reshape(1,53*47), batch_size=1)
                newQ = model.predict(new_state.reshape(1,53*47), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,16))
                y[:] = oldQ[:]

                # Calculate how the reward should be updated.
                if reward[0]: # non-terminal state
                    update = ((reward[0]) + (gamma * maxQ))
                    if action >= 8:
                        update -= 0.1 #Penalize the model for boosting
                else: # terminal state
                    update = (DEATH_COST * rounds + (gamma * maxQ))

                y[0][action] = update
                X_train.append(old_state.reshape(53*47))
                Y_train.append(y.reshape(16,))
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=1, verbose=0)
        state = new_state
        clear_output(wait=True)
        rounds += 1

    # Record the number of pellents earned in this epoch.
    # NOTE: This does not reflect the final score of the snake
    # as it does not consider how many of the pellets earned
    # were lost in boosting.
    pelletsEarnedList[len(pelletsEarnedList)-1].append(pelletsEarned)

    if epsilon > 0.1:
        # Now that the model has trained, make it more certain in the future.
        epsilon -= (1/(epochs))

    games += 1

    if games % 10 == 0:
        # Every 10 runs, resave the model (in case of internet loss).
        model.save('model250.h5')
        pickle.dump(pelletsEarnedList, open('pelletsearned.p', 'wb'))

pickle.dump(pelletsEarnedList, open('pelletsearned.p', 'wb'))

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

    if EQUALIZE:
        # When we first started running our model, we noticed that it would
        # always prefer one direction after it had finished training.
        # Hypothetically, on average, all directional movements should be
        # equally likely, so we established this method.
        # By calculating the average reward given to each action,
        # which should hypothetically be the same (for 0-7) in a proper model,
        # and then subtracting that from all future predictions, we can get
        # an idea of how 'relatively' good the model thinks the action is in
        # any state.
        averages = np.zeros((1, 16))
        i = 0
        while(not done_n[0] and i < 100):
            i += 1
            qval = model.predict(state.reshape(1,53*47), batch_size=1)
            action = (np.argmax(qval))
            new_state, reward, done_n, info = makeMove(state, action)
            averages += qval
        averages /= 100

    while(not done_n[0]):
        qval = model.predict(state.reshape(1,53*47), batch_size=1)
        if EQUALIZE:
            action = np.argmax(qval - averages)
        else:
            action = np.argmax(qval)
        new_state, reward, done_n, info = makeMove(state, action)
        if done_n[0]:
            new_state = state
        else:
            new_state = simplify(new_state[0]['vision'])
        env.render()
        state = new_state
        clear_output(wait=True)

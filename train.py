
# train.py
import random
from snake_game import SnakeGameAI
from agent import Agent
import matplotlib.pyplot as plt

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.pause(0.1)

if __name__ == '__main__':
    agent = Agent()
    game = SnakeGameAI()
    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    while True:
        state_old = agent.get_state(game)

        final_move = [0, 0, 0]
        move = random.randint(0, 2)
        final_move[move] = 1

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # save model if needed

            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)

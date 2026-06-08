import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    returns = [[] for _ in range(n_states)]

    for episode in episodes:
        visited = set()

        for t, (state, _) in enumerate(episode):
            if state in visited:
                continue

            visited.add(state)

            G = 0.0
            power = 1.0

            for k in range(t, len(episode)):
                G += power * episode[k][1]
                power *= gamma

            returns[state].append(G)

    V = np.zeros(n_states)

    for s in range(n_states):
        if returns[s]:
            V[s] = np.mean(returns[s])

    return V
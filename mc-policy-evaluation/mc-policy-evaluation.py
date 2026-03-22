import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    returns = [[] for _ in range(n_states)]  # store returns for each state
    
    for episode in episodes:
        states = [s for (s, r) in episode]
        rewards = [r for (s, r) in episode]
        
        G = 0
        visited = set()
        
        # go backward to compute returns
        for t in reversed(range(len(episode))):
            s = states[t]
            r = rewards[t]
            G = gamma * G + r
            
            # first visit check
            if s not in states[:t]:
                returns[s].append(G)
    
    # compute value function
    V = np.zeros(n_states)
    for s in range(n_states):
        if returns[s]:
            V[s] = np.mean(returns[s])
    
    return V
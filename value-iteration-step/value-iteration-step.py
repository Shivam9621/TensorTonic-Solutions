def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    
    n_states = len(values)
    new_values = []

    for s in range(n_states):
        action_values = []

        # Iterate over actions available for state s
        for a in range(len(rewards[s])):
            q_value = rewards[s][a]

            # Add discounted future rewards
            future_value = 0
            for s_next in range(n_states):
                future_value += transitions[s][a][s_next] * values[s_next]

            q_value += gamma * future_value
            action_values.append(q_value)

        # Take maximum Q-value
        new_values.append(max(action_values))

    return new_values
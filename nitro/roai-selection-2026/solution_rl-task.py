
from __future__ import annotations
import time
import numpy as np

# ---------------------------------------------------------------------------
# Part 1: one-step TD update formulas
# ---------------------------------------------------------------------------

def td_update(v_s, reward, v_next, alpha, gamma, done):
    if done:
        target = reward                      
    else:
        target = reward + gamma * v_next     
    return float(v_s + alpha * (target - v_s))


def q_learning_update(q_s_a, reward, q_next, alpha, gamma, done):
    if done:
        target = reward
    else:
        target = reward + gamma * max(q_next)
    return float(q_s_a + alpha * (target - q_s_a))


def sarsa_update(q_s_a, reward, q_next_a_next, alpha, gamma, done):
    if done:
        target = reward
    else:
        target = reward + gamma * q_next_a_next  
    return float(q_s_a + alpha * (target - q_s_a))


# ---------------------------------------------------------------------------
# Part 2: train and evaluate an agent
# ---------------------------------------------------------------------------

class Agent:
    """You may replace this class with any pickle-free/simple object you prefer."""

    def __init__(self):
        # TODO: store a learned Q-table or another policy representation here.
        pass


def train_agent(env_factory, seed: int, episodes: int):
    env = env_factory()
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    Q = np.zeros((env.observation_space.n, n_actions))

    alpha = 0.1
    gamma = 0.99
    start = time.time()

    for ep in range(episodes):
        if time.time() - start > 10:                    
            break
        eps = max(0.05, 1.0 - ep / (0.7 * episodes))

        s, _ = env.reset(seed=seed + ep)                
        done = False
        while not done:                                 
            if rng.random() < eps:
                a = int(rng.integers(n_actions))        
            else:
                a = int(np.argmax(Q[s]))                 
            s2, r, terminated, truncated, _ = env.step(a)   
            Q[s, a] = q_learning_update(Q[s, a], r, Q[s2], alpha, gamma, terminated)
            s = s2
            done = terminated or truncated
    
    return Q


def select_action(agent, observation, deterministic: bool = True) -> int:
    return int(np.argmax(agent[observation]))


# ---------------------------------------------------------------------------
# Local flow demo
# ---------------------------------------------------------------------------

def _demo_env_factory():
    """Create a public environment for local debugging only."""
    from smart_warehouse.public_scenarios import make_public_env

    return make_public_env("public_easy")


def _run_local_flow_demo() -> None:
    """
    Demonstrate the exact high-level flow used by the evaluator.

    This function intentionally calls the empty TODO functions. Until you fill
    them in, the calls will raise NotImplementedError and the demo will print a
    short message explaining which part is still missing.
    """
    print("SmartWarehouse skeleton flow demo")
    print("=================================")

    env = _demo_env_factory()
    obs, info = env.reset(seed=0)

    print("\nCreated public environment: public_easy")
    print(f"Observation space size: {env.observation_space.n}")
    print(f"Action space size: {env.action_space.n}")
    print(f"Initial observation: {obs}")
    print(f"Decoded state: {env.decode(obs)}  # (row, col, battery, carrying)")
    print("\nInitial warehouse:")
    print(env.render_ascii())

    print("\n1) Calling td_update(...) on a tiny numeric example")
    try:
        value = td_update(v_s=1.0, reward=2.0, v_next=3.0, alpha=0.5, gamma=0.9, done=False)
        print(f"td_update returned: {value}")
    except NotImplementedError:
        print("td_update is still empty. Fill it in before submitting.")

    print("\n2) Calling train_agent(env_factory, seed=0, episodes=5)")
    try:
        agent = train_agent(_demo_env_factory, seed=0, episodes=5)
        print("train_agent returned an agent object.")
    except NotImplementedError:
        print("train_agent is still empty. Fill it in before submitting.")
        agent = Agent()

    print("\n3) Calling select_action(agent, observation, deterministic=True)")
    try:
        action = select_action(agent, obs, deterministic=True)
        print(f"select_action returned action: {action}")
    except NotImplementedError:
        print("select_action is still empty. Fill it in before submitting.")
        print("For this demo only, taking action 3 = right so that env.step(...) can be shown.")
        action = 3

    obs, reward, terminated, truncated, info = env.step(action)
    print("\n4) One environment step")
    print(f"Next observation: {obs}")
    print(f"Decoded state: {env.decode(obs)}")
    print(f"Reward: {reward}")
    print(f"terminated={terminated}, truncated={truncated}, info={info}")
    print("\nWarehouse after one step:")
    print(env.render_ascii())


#------------------------------
# Creating the final submission
#------------------------------
def _make_submission():
    import cloudpickle
    print("Packaging submission...")

    submission_dict = {
        "td_update": td_update,
        "q_learning_update": q_learning_update,
        "sarsa_update": sarsa_update,
        "train_agent": train_agent,
        "select_action": select_action
    }

    with open("submission.pkl", "wb") as f:
        cloudpickle.dump(submission_dict, f)

    print("Saved submission.pkl!")

if __name__ == "__main__":
    _run_local_flow_demo()
    _make_submission()

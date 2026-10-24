import random

GRID = (
    (0, 0, 0, "S"),
    (0, "X", 0, 0),
    (0, 0, 0, 0),
    (0, -1, 0, 10),
)

START = (0, 3)
GAMMA = 0.95
ACTIONS = ("N", "E", "S", "W")
DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
STATES = [(r, c) for r in range(4) for c in range(4) if GRID[r][c] != "X"]
TERMINALS = {s: float(GRID[s[0]][s[1]]) for s in STATES
             if GRID[s[0]][s[1]] in (-1, 10)}


def move(state, direction):
    dr, dc = DELTA[direction]
    target = (state[0] + dr, state[1] + dc)
    return target if target in STATES else state


def outcomes(state, action):
    i = ACTIONS.index(action)
    return [(0.8, action, move(state, action)),
            (0.1, ACTIONS[(i - 1) % 4], move(state, ACTIONS[(i - 1) % 4])),
            (0.1, ACTIONS[(i + 1) % 4], move(state, ACTIONS[(i + 1) % 4]))]


def reward(next_state, living_cost):
    return TERMINALS.get(next_state, living_cost)


def q_value(state, action, values, living_cost):
    return sum(p * (reward(ns, living_cost) + GAMMA * values[ns])
               for p, _, ns in outcomes(state, action))


def solve(living_cost, tolerance=1e-10):
    values = dict.fromkeys(STATES, 0.0)
    for iteration in range(1, 10001):
        updated = {
            s: (0.0 if TERMINALS.get(s) == 10
                else -1.0 + GAMMA * values[s] if TERMINALS.get(s) == -1
                else max(q_value(s, a, values, living_cost) for a in ACTIONS))
            for s in STATES
        }
        change = max(abs(updated[s] - values[s]) for s in STATES)
        values = updated
        if change < tolerance:
            policy = {s: max(ACTIONS, key=lambda a: q_value(s, a, values, living_cost))
                      for s in STATES if s not in TERMINALS}
            return values, policy, iteration
    raise RuntimeError("Value iteration did not converge")


def render(policy, position=None):
    for r in range(4):
        row = []
        for c in range(4):
            s = (r, c)
            label = ("R" if s == position else "X" if s not in STATES else
                     str(int(TERMINALS[s])) if s in TERMINALS else policy[s])
            row.append(f"{label:>3}")
        print(" ".join(row))


def simulate(policy, living_cost, seed, verbose=False, delay=0, max_steps=500):
    rng = random.Random(seed)
    state, path, total, discounted = START, [START], 0.0, 0.0
    for step in range(1, max_steps + 1):
        if state in TERMINALS and TERMINALS[state] == -1:
            action, actual, ns = "STUCK", "STUCK", state
        else:
            action = policy[state]
            u, cumulative = rng.random(), 0.0
            for probability, actual, ns in outcomes(state, action):
                cumulative += probability
                if u < cumulative:
                    break
        earned = reward(ns, living_cost)
        total += earned
        discounted += GAMMA ** (step - 1) * earned
        if verbose:
            print(f"{step:03d}: {state} intended={action} actual={actual} -> {ns} reward={earned:+.2f}")
        state = ns
        path.append(state)
        if verbose and delay:
            render(policy, state)
            print()
        if TERMINALS.get(state) == 10:
            break
    status = "RECHARGED" if TERMINALS.get(state) == 10 else "TRAPPED" if state in TERMINALS else "TIMEOUT"
    if verbose:
        print("Path:", " -> ".join(map(str, path)))
        print(f"{status}; steps={len(path)-1}; total={total:.2f}; discounted={discounted:.6f}")
    return status, len(path)-1, total, discounted


def main():
    seed = 7
    delay = 0.5
    trials = 1000

    print("Terminal reward replaces living cost on entry; gamma=0.95.")

    for cost in (-0.04, -1.0, -5.0):
        values, policy, iterations = solve(cost)
        print(f"\n=== Living cost {cost:+.2f}; iterations={iterations}; V(start)={values[START]:.6f} ===")
        print("Policy (N/E/S/W); R marks robot:")
        render(policy, START)
        simulate(policy, cost, seed, verbose=True, delay=delay)

        results = [simulate(policy, cost, trial_seed) for trial_seed in range(trials)]
        counts = {name: sum(result[0] == name for result in results)
                  for name in ("RECHARGED", "TRAPPED", "TIMEOUT")}
        mean_steps = sum(result[1] for result in results) / trials
        mean_return = sum(result[3] for result in results) / trials
        print(f"Trials={trials}: {counts}; mean steps={mean_steps:.3f}; "
              f"mean discounted return={mean_return:.6f}")


if __name__ == "__main__":
    main()

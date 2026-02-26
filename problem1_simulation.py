"""
EE 5393 - Homework #1, Question 1
Stochastic Simulation of Chemical Reaction Networks (Gillespie Algorithm)
Reactions:
    R1: 2X1 + X2 --> 4X3    k1 = 1
    R2: X1 + 2X3 --> 3X2    k2 = 2
    R3: X2 + X3  --> 2X1    k3 = 3
Part (a): From [110, 26, 55], estimate Pr(C1), Pr(C2), Pr(C3)
Part (b): From [9, 8, 7], compute mean and variance after 7 steps
AI Disclosure: Simulation code generated with AI assistance.
"""
import numpy as np
import random
# ──────────────────────────────────────────────
#  Propensity Function (Discrete Model)
# ──────────────────────────────────────────────
def propensities(x1, x2, x3):
    """
    Discrete propensities:

        a1 = 0.5 * x1*(x1-1) * x2
        a2 = x1 * x3*(x3-1)
        a3 = 3 * x2 * x3
    """
    a1 = 0.5 * x1 * (x1 - 1) * x2 if x1 >= 2 else 0.0
    a2 = x1 * x3 * (x3 - 1) if x3 >= 2 else 0.0
    a3 = 3.0 * x2 * x3
    return a1, a2, a3

def gillespie_step(state):
    """Performs one discrete Gillespie reaction step."""
    x1, x2, x3 = state
    a1, a2, a3 = propensities(x1, x2, x3)
    total = a1 + a2 + a3

    if total <= 0:
        return state[:], None

    r = random.random() * total

    if r < a1:
        return [x1 - 2, x2 - 1, x3 + 4], 1
    elif r < a1 + a2:
        return [x1 - 1, x2 + 3, x3 - 2], 2
    else:
        return [x1 + 2, x2 - 1, x3 - 1], 3
# ──────────────────────────────────────────────
#  Part (a): Estimate Outcome Probabilities
# ──────────────────────────────────────────────
def part_a(n_runs=50_000, seed=42):
    random.seed(seed)

    INIT = [110, 26, 55]
    counts = {'C1': 0, 'C2': 0, 'C3': 0}

    for _ in range(n_runs):
        state = INIT[:]

        for _ in range(20_000):
            x1, x2, x3 = state

            if x1 >= 150:
                counts['C1'] += 1
                break
            if x2 < 10:
                counts['C2'] += 1
                break
            if x3 > 100:
                counts['C3'] += 1
                break

            new_state, rxn = gillespie_step(state)

            if rxn is None or any(v < 0 for v in new_state):
                break

            state = new_state

    print("\nPart (a): Outcome Probabilities (50,000 runs)")
    print("---------------------------------------------")
    print(f"Pr(C1: x1 >= 150) = {counts['C1']/n_runs:.6f}")
    print(f"Pr(C2: x2 < 10)   = {counts['C2']/n_runs:.6f}")
    print(f"Pr(C3: x3 > 100)  = {counts['C3']/n_runs:.6f}")

    return {k: counts[k]/n_runs for k in counts}
# ──────────────────────────────────────────────
#  Part (b): Mean and Variance after 7 Steps
# ──────────────────────────────────────────────

def part_b(n_runs=100_000, n_steps=7, seed=42):
    random.seed(seed)

    INIT = [9, 8, 7]
    results = np.zeros((n_runs, 3), dtype=int)

    for i in range(n_runs):
        state = INIT[:]

        for _ in range(n_steps):
            new_state, rxn = gillespie_step(state)

            if rxn is None or any(v < 0 for v in new_state):
                break

            state = new_state

        results[i] = state

    print("\nPart (b): After 7 Steps (100,000 runs)")
    print("---------------------------------------")

    for i, name in enumerate(['X1', 'X2', 'X3']):
        mean = np.mean(results[:, i])
        var  = np.var(results[:, i])
        print(f"{name}: Mean = {mean:.6f}, Variance = {var:.6f}")

    return results
# ──────────────────────────────────────────────
#  Main Execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    part_a()
    part_b()

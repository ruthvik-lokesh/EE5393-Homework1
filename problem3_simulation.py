"""
EE 5393 - Homework 1 | Problem 3 Verification
==============================================
Verifies both CRNs from Problem 3 using the Gillespie SSA:
  Part (a): Z_inf = X0 * log2(Y0)
  Part (b): Y_inf = 2^(log2(X0))  [mathematically equals X0]

Each CRN is composed of two modules with time-scale separation:
  slow < medium < fast < faster (rate constants below)
"""

import random
import math
import numpy as np
from collections import defaultdict

# ─── Rate constants (time-scale separated) ───────────────────────────────────
SLOW    = 0.01
MEDIUM  = 1.0
FAST    = 10.0
FASTER  = 100.0
SLOWER  = 0.001   # for multiplication / exponentiation trigger

TRIALS  = 100
MAX_STEPS = 500_000   # safety cap per trial


# ═════════════════════════════════════════════════════════════════════════════
# Generic Gillespie SSA
# ═════════════════════════════════════════════════════════════════════════════

def gillespie(state: dict, reactions: list, max_steps: int = MAX_STEPS) -> dict:
    """
    Run the Gillespie SSA until no reactions are enabled or max_steps reached.

    reactions: list of (propensity_fn, update_dict)
      propensity_fn(state) -> float
      update_dict          -> {species: delta}
    """
    state = state.copy()
    for _ in range(max_steps):
        props = [max(0.0, r[0](state)) for r in reactions]
        a0 = sum(props)
        if a0 == 0:
            break
        # choose reaction
        r_val = random.random() * a0
        cumulative = 0.0
        chosen = len(reactions) - 1
        for i, p in enumerate(props):
            cumulative += p
            if r_val <= cumulative:
                chosen = i
                break
        # apply update
        for species, delta in reactions[chosen][1].items():
            state[species] = state.get(species, 0) + delta
    return state


# ═════════════════════════════════════════════════════════════════════════════
# Part (a): Z_inf = X0 * log2(Y0)
# ═════════════════════════════════════════════════════════════════════════════
#
# MODULE 1 – Logarithm (computes R = log2(Y0))
#   S  -slow->  T
#   T + 2Y  -faster->  H + Y' + T      (halving round: consumes 2Y, produces H, Y')
#   2H  -faster->  H                   (annihilation: net removes one H)
#   T  -fast->  ∅                      (token degradation)
#   Y' -medium-> Y                     (restore Y count)
#   H  -medium-> R                     (H converts to output R)
#
# MODULE 2 – Multiplication (computes Z = X0 * R)
#   X  -slower->  U
#   U + R  -faster->  U + R' + Z       (each U catalyses copying of R into Z)
#   U  -fast->  ∅
#   R' -medium-> R                     (restore R)

def build_log_reactions(Y_species='Y', Yprime='Yp', S_species='S',
                         T_species='T', H_species='H', R_species='R'):
    """Returns the 6 logarithm-module reactions."""
    return [
        # S -> T
        (lambda s, S=S_species, T=T_species:
            SLOW * s.get(S, 0),
         {S_species: -1, T_species: +1}),

        # T + 2Y -> H + Y' + T  (propensity: FASTER * T * C(Y,2))
        (lambda s, Y=Y_species, T=T_species, H=H_species, Yp=Yprime:
            FASTER * s.get(T, 0) * s.get(Y, 0) * max(0, s.get(Y, 0) - 1) / 2,
         {Y_species: -2, H_species: +1, Yprime: +1}),   # T is catalytic, unchanged

        # 2H -> H  (propensity: FASTER * C(H,2))
        (lambda s, H=H_species:
            FASTER * s.get(H, 0) * max(0, s.get(H, 0) - 1) / 2,
         {H_species: -1}),

        # T -> ∅
        (lambda s, T=T_species:
            FAST * s.get(T, 0),
         {T_species: -1}),

        # Y' -> Y
        (lambda s, Yp=Yprime, Y=Y_species:
            MEDIUM * s.get(Yp, 0),
         {Yprime: -1, Y_species: +1}),

        # H -> R
        (lambda s, H=H_species, R=R_species:
            MEDIUM * s.get(H, 0),
         {H_species: -1, R_species: +1}),
    ]


def simulate_part_a(X0: int, Y0: int) -> tuple:
    """Returns (avg_Z, avg_R) over TRIALS runs."""
    Z_results = []
    R_results = []

    for _ in range(TRIALS):
        # ── Phase 1: Logarithm module ─────────────────────────────────────
        state = {'S': 40, 'Y': Y0, 'Yp': 0, 'T': 0, 'H': 0, 'R': 0}
        log_rxns = build_log_reactions()
        state = gillespie(state, log_rxns)
        R_val = state.get('R', 0)

        # ── Phase 2: Multiplication module ────────────────────────────────
        # Start fresh with X=X0, R=R_val from phase 1
        state2 = {'X': X0, 'R': R_val, 'Rp': 0, 'U': 0, 'Z': 0}
        mul_rxns = [
            # X -> U
            (lambda s: SLOWER * s.get('X', 0),
             {'X': -1, 'U': +1}),

            # U + R -> U + R' + Z  (propensity: FASTER * U * R, U catalytic)
            (lambda s: FASTER * s.get('U', 0) * s.get('R', 0),
             {'R': -1, 'Rp': +1, 'Z': +1}),

            # U -> ∅
            (lambda s: FAST * s.get('U', 0),
             {'U': -1}),

            # R' -> R
            (lambda s: MEDIUM * s.get('Rp', 0),
             {'Rp': -1, 'R': +1}),
        ]
        state2 = gillespie(state2, mul_rxns)
        Z_results.append(state2.get('Z', 0))
        R_results.append(R_val)

    return np.mean(Z_results), np.mean(R_results)


# ═════════════════════════════════════════════════════════════════════════════
# Part (b): Y_inf = 2^(log2(X0))   [= X0 mathematically]
# ═════════════════════════════════════════════════════════════════════════════
#
# MODULE 1 – Logarithm on X (same structure, computes R = log2(X0))
#
# MODULE 2 – Exponentiation (computes Y = 2^R, starting from Y=1)
#   R  -slower->  D
#   D + Y  -faster->  2Y* + D    (D catalytic: doubles Y)
#   D  -fast->  ∅
#   Y* -medium-> Y

def simulate_part_b(X0: int) -> tuple:
    """Returns (avg_Y, avg_R) over TRIALS runs."""
    Y_results = []
    R_results = []

    for _ in range(TRIALS):
        # ── Phase 1: Logarithm module (on X) ─────────────────────────────
        state = {'S': 40, 'X': X0, 'Xp': 0, 'T': 0, 'H': 0, 'R': 0}
        log_rxns = build_log_reactions(
            Y_species='X', Yprime='Xp',
            S_species='S', T_species='T', H_species='H', R_species='R'
        )
        state = gillespie(state, log_rxns)
        R_val = state.get('R', 0)

        # ── Phase 2: Exponentiation module ───────────────────────────────
        state2 = {'R': R_val, 'D': 0, 'Y': 1, 'Ys': 0}
        exp_rxns = [
            # R -> D
            (lambda s: SLOWER * s.get('R', 0),
             {'R': -1, 'D': +1}),

            # D + Y -> 2Y* + D  (D catalytic, propensity: FASTER * D * Y)
            (lambda s: FASTER * s.get('D', 0) * s.get('Y', 0),
             {'Y': -1, 'Ys': +2}),

            # D -> ∅
            (lambda s: FAST * s.get('D', 0),
             {'D': -1}),

            # Y* -> Y
            (lambda s: MEDIUM * s.get('Ys', 0),
             {'Ys': -1, 'Y': +1}),
        ]
        state2 = gillespie(state2, exp_rxns)
        Y_results.append(state2.get('Y', 0))
        R_results.append(R_val)

    return np.mean(Y_results), np.mean(R_results)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    random.seed(42)

    # ── Part (a) ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Part (a): Z_inf = X0 * log2(Y0)")
    print("=" * 60)
    print(f"{'X0':>4} {'Y0':>4} {'Expected Z':>12} {'Avg Z':>10} {'Avg R':>10}")
    print("-" * 60)

    test_cases_a = [(3, 8), (4, 16), (2, 32), (5, 4), (1, 64)]
    for X0, Y0 in test_cases_a:
        expected_Z = X0 * math.log2(Y0)
        avg_Z, avg_R = simulate_part_a(X0, Y0)
        print(f"{X0:>4} {Y0:>4} {expected_Z:>12.2f} {avg_Z:>10.2f} {avg_R:>10.2f}")

    # ── Part (b) ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Part (b): Y_inf = 2^(log2(X0))  [= X0 mathematically]")
    print("=" * 60)
    print(f"{'X0':>4} {'Expected Y':>12} {'Avg Y':>10} {'Avg R':>10}")
    print("-" * 60)

    test_cases_b = [4, 8, 16, 32]
    for X0 in test_cases_b:
        expected_Y = X0   # 2^log2(X0) = X0
        avg_Y, avg_R = simulate_part_b(X0)
        print(f"{X0:>4} {expected_Y:>12.2f} {avg_Y:>10.2f} {avg_R:>10.2f}")

    print()
    print("Note: Small positive bias in Part (a) is expected due to")
    print("stochastic variation in the halving steps of the log module.")
    print("In Part (b), exponentiation amplifies that bias for larger X0.")

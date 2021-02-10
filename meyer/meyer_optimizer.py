import numpy as np
from constants import P21, V, P, Q


def cached_function(func):

    def inner(*args):
        if args not in inner.cache.keys():
            inner.cache[args] = func(*args)
        return inner.cache[args]

    inner.cache = {}
    return inner

# assumption: only metric = my points vs the average


@cached_function
def p_lie(n, last_claim):
    p_had_to_lie = Q[last_claim]/(1-P21)
    return p_had_to_lie


@cached_function
def mu_throw(n, last_claim, players_until_me):
    """ Expected outcome for me, if the player throws. """
    loss_21 = 1/(n-1)
    if players_until_me == 0:
        loss_21 = -1

    mu_best_lie, mu_throw_true_claim = calculate_lines(n, last_claim, players_until_me)
    _mu_throw = P21 * loss_21 + Q[last_claim] * mu_best_lie + np.inner(P, mu_throw_true_claim)
    return _mu_throw


@cached_function
def calculate_lines(n, last_claim, players_until_me):
    next_players_until_me = players_until_me - 1
    loss_proc_doubt_correct = -1/(n-1)
    loss_proc_doubt_miss = -1/(n-1)
    if players_until_me == 0:
        next_players_until_me = n - 1
        loss_proc_doubt_correct = 1
    elif players_until_me == 1:
        loss_proc_doubt_miss = 1
    mu_throw_false_claim = np.ones(20)*loss_proc_doubt_correct
    mu_throw_true_claim = np.zeros(20)

    for claim in V[last_claim + 1:]:
        mu_proc_throw = mu_throw(n, claim, next_players_until_me)
        p_proc_doubt = do_doubt(n, last_claim, claim)
        weighted_mu_proc_throw = (1 - p_proc_doubt) * mu_proc_throw
        mu_throw_true_claim[claim] = weighted_mu_proc_throw + p_proc_doubt * loss_proc_doubt_miss
        mu_throw_false_claim[claim] = weighted_mu_proc_throw + p_proc_doubt * loss_proc_doubt_correct
    # assumption: always use the optimal lie
    mu_best_lie = min(mu_throw_false_claim)
    return mu_best_lie, mu_throw_true_claim


@cached_function
def mu_doubt(n, claim_m2):
    """
    Expected outcome, if I doubt the last last claim.
    :param n: number of players
    :param claim_m2: claim made by P_-2
    """
    return 1 - n/(n-1) * p_lie(n, claim_m2)


@cached_function
def do_doubt(n, claim_m2, claim_m1):
    """
    :param n: number of players
    :param claim_m2: claim made by the player before the previous one
    :param claim_m1: claim made by the previous player
    """
    if claim_m2 is claim_m1 is None:
        return False
    elif claim_m1 not in V[claim_m2 + 1:]:
        # penalize rule breaking
        return True
    # assumption: always take the optimal action
    return mu_doubt(n, claim_m2) < mu_throw(n, claim_m1, 0)


@cached_function
def mu(n, claim_m2, claim_m1):
    """
    :param n: number of players
    :param claim_m2: claim made by the player before the previous one
    :param claim_m1: claim made by the previous player
    """
    if claim_m2 is claim_m1 is None:
        return mu_throw(n, claim_m1, 0)
    elif claim_m1 not in V[claim_m2 + 1:]:
        return mu_doubt(n, claim_m2)
    # assumption: always take the optimal action
    return min((mu_doubt(n, claim_m2), mu_throw(n, claim_m1, 0)))

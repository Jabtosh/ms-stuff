import numpy as np

from constants import P21, V, P, Q


def cached_function(func):
    """ Currently, only made compatible with positional arguments. """

    def inner(*args):
        if args not in inner.cache.keys():
            inner.cache[args] = func(*args)
        return inner.cache[args]

    inner.cache = {}
    return inner


""" Assumption: only metric = my points vs the average """


@cached_function
def p_lie(n: int, last_claim: int) -> float:
    """ Probability, that a player lied, given the last claim. """
    # TODO: include needless lies
    p_had_to_lie = Q[last_claim] / (1 - P21)
    return p_had_to_lie


@cached_function
def mu_throw(n: int, last_claim: int, players_until_me: int, rounds_remaining: int) -> float:
    """ Expected outcome for me, if the acting player throws. """
    loss_21 = 1 / (n - 1)
    if players_until_me == 0:
        loss_21 = -1

    mu_best_lie, mu_throw_true_claim = calculate_lines(n, last_claim, players_until_me, rounds_remaining)
    _mu_throw = P21 * loss_21 + Q[last_claim] * mu_best_lie + np.inner(P, mu_throw_true_claim)
    return _mu_throw


@cached_function
def calculate_lines(n: int, last_claim: int, players_until_me: int, rounds_remaining: int) -> (float, np.ndarray):
    """ Collapse the subbranches of the tree diagram. """
    new_players_until_me = players_until_me - 1
    loss_proc_doubt_correct = -1 / (n - 1)
    loss_proc_doubt_miss = -1 / (n - 1)
    if players_until_me == 0:
        new_players_until_me = n - 1
        loss_proc_doubt_correct = 1
    elif players_until_me == 1:
        loss_proc_doubt_miss = 1
    mu_throw_false_claim = np.ones(21) * loss_proc_doubt_correct
    mu_throw_true_claim = np.zeros(21)

    for claim in V[last_claim + 1:]:
        mu_throw_true_claim[claim], mu_throw_false_claim[claim] = _mu_claim_possibilities(
            n, last_claim, claim, new_players_until_me, rounds_remaining, loss_proc_doubt_miss, loss_proc_doubt_correct)

    # assumption: always use the optimal lie
    mu_best_lie = min(mu_throw_false_claim)
    return mu_best_lie, mu_throw_true_claim


@cached_function
def _mu_claim_possibilities(n: int, last_claim: int, claim: int, next_players_until_me: int, rounds_remaining: int,
                            loss_proc_doubt_miss: float, loss_proc_doubt_correct: float) -> (float, float):
    if do_doubt(n, last_claim, claim, rounds_remaining):
        mu_proc_restart = mu(n, 0, 0, next_players_until_me, rounds_remaining - 1)
        _mu_throw_true_claim = mu_proc_restart + loss_proc_doubt_miss
        _mu_throw_false_claim = mu_proc_restart + loss_proc_doubt_correct
    else:
        mu_proc_throw = mu_throw(n, claim, next_players_until_me, rounds_remaining)
        _mu_throw_true_claim = mu_proc_throw
        _mu_throw_false_claim = mu_proc_throw
    return _mu_throw_true_claim, _mu_throw_false_claim


@cached_function
def mu_doubt(n: int, claim_m2: int, players_until_me: int, rounds_remaining: int) -> float:
    """ Expected outcome, if I doubt the last claim. """
    loss_doubt_correct = -1 / (n - 1)
    loss_doubt_miss = -1 / (n - 1)
    if players_until_me == 0:
        loss_doubt_miss = 1
    elif players_until_me == n - 1:
        loss_doubt_correct = 1
    mu_restart = mu(n, 0, 0, players_until_me, rounds_remaining - 1)
    return loss_doubt_correct * p_lie(n, claim_m2) + loss_doubt_miss * (1 - p_lie(n, claim_m2)) + mu_restart


@cached_function
def do_doubt(n: int, claim_m2: int, claim_m1: int, rounds_remaining: int) -> bool:
    """ Is it optimal to doubt claim_m1? """
    if claim_m2 == claim_m1 == 0:
        # not allowed to doubt
        return False
    elif claim_m1 not in V[claim_m2 + 1:]:
        raise RuleException("Previous player broke the rules.")
    # assumption: always take the optimal action
    return mu_doubt(n, claim_m2, 0, rounds_remaining) < mu_throw(n, claim_m1, 0, rounds_remaining)


@cached_function
def mu(n: int, claim_m2: int, claim_m1: int, players_until_me: int, rounds_remaining: int) -> float:
    """
    :param n: number of players
    :param claim_m2: claim made by the player before the previous one
    :param claim_m1: claim made by the previous player
    :param players_until_me: players to act before me (0 if it's my turn)
    :param rounds_remaining: rounds remaining after the current one
    """
    # follow rules
    if rounds_remaining < 0:
        return 0
    if claim_m2 == claim_m1 == 0:
        return mu_throw(n, claim_m1, players_until_me, rounds_remaining)
    elif claim_m1 not in V[claim_m2 + 1:]:
        raise RuleException("Previous player broke the rules.")
    return min(mu_doubt(n, claim_m2, players_until_me, rounds_remaining),
               mu_throw(n, claim_m1, players_until_me, rounds_remaining))


class RuleException(Exception):
    pass

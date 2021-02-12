import numpy as np

from constants import P21, V, P, Q


def cache_results(func):
    """ Currently, optimized for positional arguments. """

    def inner(*args, **kwargs):
        if kwargs:
            args = (*args, str(kwargs))
        if args not in inner.cache.keys():
            inner.cache[args] = func(*args)
        return inner.cache[args]

    inner.cache = {}
    return inner


""" Assumption: only metric = my points vs the average """


@cache_results
def p_lie(n: int, last_claim: int) -> float:
    """ Probability, that a player lied, given the last claim. """
    # TODO: include needless lies (conditional probabilities: include input "claim")
    return p_has_to_lie(last_claim)


@cache_results
def p_has_to_lie(last_claim: int) -> float:
    """ Probability, that a has to lie, given the last claim. """
    return Q[last_claim] / (1 - P21)


@cache_results
def mu_throw(n: int, last_claim: int, players_until_me: int, rounds_remaining: int) -> float:
    """ Expected outcome for me, if the acting player throws. """
    loss_21 = 1. / (n - 1)
    if players_until_me == 0:
        loss_21 = -1.

    mu_throw_v = calculate_lines(n, last_claim, players_until_me, rounds_remaining)
    _mu_throw = P21 * loss_21 + P @ mu_throw_v
    return _mu_throw


@cache_results
def calculate_lines(n: int, last_claim: int, players_until_me: int, rounds_remaining: int) -> (float, np.ndarray):
    """ Collapse the subbranches of the tree. """
    new_players_until_me = players_until_me - 1
    loss_proc_doubt_correct = -1. / (n - 1)
    loss_proc_doubt_miss = -1. / (n - 1)
    if players_until_me == 0:
        new_players_until_me = n - 1
        loss_proc_doubt_correct = 1.
    elif players_until_me == 1:
        loss_proc_doubt_miss = 1.

    # perspective of the thrower
    # assumption: always use the same lie
    # pick best under the assumption
    # TODO: conditional probabilities depending on claim
    # TODO: refactor
    _p_lie = p_has_to_lie(last_claim)
    loss_doubt = (1. * _p_lie - 1. / (n - 1) * (1 - _p_lie))
    mu_throw_false_claim = np.ones(21)
    for claim in V[last_claim + 1:]:
        mu_throw_false_claim[claim] = _mu_claim_possibilities(n, last_claim, claim, n - 1, rounds_remaining,
                                                              loss_doubt)
    best_lie = mu_throw_false_claim.argmin()

    _p_lie = p_lie(n, last_claim)
    loss_doubt = (loss_proc_doubt_correct * _p_lie + loss_proc_doubt_miss * (1 - _p_lie))
    mu_throw_true_claim = np.ones(21) * loss_doubt
    for claim in V[last_claim + 1:]:
        mu_throw_true_claim[claim] = _mu_claim_possibilities(n, last_claim, claim, new_players_until_me,
                                                             rounds_remaining, loss_doubt)
    for claim in V[:last_claim + 1]:
        mu_throw_true_claim[claim] = mu_throw_true_claim[best_lie]
    return mu_throw_true_claim


@cache_results
def _mu_claim_possibilities(n: int, last_claim: int, claim: int, next_players_until_me: int, rounds_remaining: int,
                            loss_doubt: float) -> (float, float):
    _p_doubt = do_doubt(n, last_claim, claim, rounds_remaining)
    _q_believe = 1 - _p_doubt
    mu_proc_throw = mu_throw(n, claim, next_players_until_me, rounds_remaining)
    mu_proc_restart = mu(n, 0, 0, next_players_until_me, rounds_remaining - 1)
    _mu_throw_v = _q_believe * mu_proc_throw + _p_doubt * (mu_proc_restart + loss_doubt)
    return _mu_throw_v


@cache_results
def mu_doubt(n: int, claim_m2: int, players_until_me: int, rounds_remaining: int) -> float:
    """ Expected outcome, if I doubt the last claim. """
    loss_doubt_correct = -1. / (n - 1)
    loss_doubt_miss = -1. / (n - 1)
    if players_until_me == 0:
        loss_doubt_miss = 1.
    elif players_until_me == n - 1:
        loss_doubt_correct = 1.
    mu_restart = mu(n, 0, 0, players_until_me, rounds_remaining - 1)
    return loss_doubt_correct * p_lie(n, claim_m2) + loss_doubt_miss * (1 - p_lie(n, claim_m2)) + mu_restart


@cache_results
def do_doubt(n: int, claim_m2: int, claim_m1: int, rounds_remaining: int) -> float:
    """ Is it optimal to doubt claim_m1? """
    _mu_throw = 0
    if claim_m2 == claim_m1 == 0:
        # not allowed to doubt
        _mu_doubt = 1
    elif claim_m1 not in V[claim_m2 + 1:]:
        _mu_doubt = -1
    else:
        # assumption: always take the optimal action
        _mu_doubt = mu_doubt(n, claim_m2, 0, rounds_remaining)
        _mu_throw = mu_throw(n, claim_m1, 0, rounds_remaining)

    if _mu_doubt == _mu_throw:
        return .5
    else:
        return float(_mu_doubt < _mu_throw)


@cache_results
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

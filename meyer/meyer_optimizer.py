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


""" Assumption: only metric = minimize my points compared to the average among the others """


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
    loss_21 = -1. if players_until_me == 0 else 1. / (n - 1)

    _p_lie = p_lie(n, last_claim)
    _mu_throw_v = mu_throw_v(n, last_claim, _p_lie, players_until_me, rounds_remaining)
    _best_lie = best_lie(n, last_claim, rounds_remaining)
    for claim in V[:last_claim + 1]:
        _mu_throw_v[claim] = _mu_throw_v[_best_lie]

    return P21 * loss_21 + P @ _mu_throw_v


def best_lie(n, last_claim, rounds_remaining):
    # TODO: sanity check
    _p_lie = p_has_to_lie(last_claim)
    _mu_throw_v_thrower_perspective = mu_throw_v(n, last_claim, _p_lie, 0, rounds_remaining)
    return _mu_throw_v_thrower_perspective.argmin()


def mu_throw_v(n: int, last_claim: int, _p_lie: float, players_until_me: int, rounds_remaining: int) -> np.ndarray:
    loss_doubt, new_players_until_me = _perspective_parameters(n, _p_lie, players_until_me)
    _mu_throw_v = np.ones(len(V)) * loss_doubt
    for claim in V[last_claim + 1:]:
        _p_doubt = do_doubt(n, last_claim, claim, rounds_remaining)
        mu_proc_throw = mu_throw(n, claim, new_players_until_me, rounds_remaining)
        mu_proc_restart = mu(n, 0, 0, new_players_until_me, rounds_remaining - 1)
        _mu_throw_v[claim] = _p_doubt * (mu_proc_restart + loss_doubt) + (1 - _p_doubt) * mu_proc_throw
    return _mu_throw_v


@cache_results
def _perspective_parameters(n: int, _p_lie: float, players_until_me: int) -> (float, int):
    new_players_until_me = players_until_me - 1
    loss_proc_doubt_correct = -1. / (n - 1)
    loss_proc_doubt_miss = -1. / (n - 1)
    if players_until_me == 0:
        new_players_until_me = n - 1
        loss_proc_doubt_correct = 1.
    elif players_until_me == 1:
        loss_proc_doubt_miss = 1.
    loss_doubt = (loss_proc_doubt_correct * _p_lie + loss_proc_doubt_miss * (1 - _p_lie))
    return loss_doubt, new_players_until_me


@cache_results
def mu_doubt(n: int, claim_m2: int, players_until_me: int, rounds_remaining: int) -> float:
    """ Expected outcome for me, if the last claim is doubted. """
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
        return 0.
    elif claim_m1 not in V[claim_m2 + 1:]:
        return 1.
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
    p_doubt = do_doubt(n, claim_m2, claim_m1, rounds_remaining)
    return (p_doubt * mu_doubt(n, claim_m2, players_until_me, rounds_remaining) +
            (1 - p_doubt) * mu_throw(n, claim_m1, players_until_me, rounds_remaining))


class RuleException(Exception):
    pass

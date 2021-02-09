import numpy as np

# Constants
V = np.arange(20)
P = np.zeros(20)
P[V <14] = 2/36
P[V>=14] = 1/36
Q = P.cumsum()
P21 = 2/36

# assumption: only metric: my points vs the average

def p_lie_unnecessarily(N, last_claim):
    # approx:
    return 0

def p_lie(N, last_claim):
    p_had_to_lie = Q[last_claim]/(1-P21)
    return p_had_to_lie + (1-p_had_to_lie)*p_lie_unnecessarily(N, last_claim)

def mu_throw(N, last_claim, players_until_me):
    """ Expected outcome, if the proceeding player throws. """
    _players_until_me = players_until_me - 1
    loss_21 = 1/(N-1)
    loss_proc_doubt_correct = -1/(N-1)
    loss_proc_doubt_miss = -1/(N-1)
    if players_until_me == 0:
        _players_until_me = N - 1
        loss_21 = -1
        loss_proc_doubt_correct = 1
    elif players_until_me == 1:
        loss_proc_doubt_miss = 1

    _mu_throw = P21 * loss_21
    if last_claim == V[-1]:
        _mu_throw += (1 - P21) * loss_proc_doubt_correct
    else:
        mu_throw_false_claim = []
        for claim in V[last_claim + 1:]:
            _mu_proc_throw = mu_throw(N, claim, _players_until_me)
            p_proc_doubt = do_doubt(N, last_claim, claim)
            mu_throw_true_claim = (1 - p_proc_doubt) * _mu_proc_throw + p_proc_doubt * loss_proc_doubt_miss
            mu_throw_false_claim.append((1 - p_proc_doubt) * _mu_proc_throw + p_proc_doubt * loss_proc_doubt_correct)
            _mu_throw += P[claim] * mu_throw_true_claim
        # assumption: always use the optimal lie
        _mu_throw += Q[last_claim] * min(mu_throw_false_claim)
    return _mu_throw

def mu_doubt(N, claim_m2):
    """
    :param N: number of players
    :param claim_m2: claim made by P_-2
    """
    return 1 - N/(N-1) * p_lie(N, claim_m2)

def do_doubt(N, claim_m2, claim_m1):
    """
    :param N: number of players
    :param claim_m2: claim made by the player before the previous one
    :param claim_m1: claim made by the previous player
    """
    # assumption: always take the optimal action
    return mu_doubt(N, claim_m2) < mu_throw(N, claim_m1, 0)

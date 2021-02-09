import numpy as np

# Constants
V = np.arange(20)
P = np.zeros(20)
P[V <14] = 2/36
P[V>=14] = 1/36
Q = P.cumsum()
P21 = 2/36

# assumption: only metric: my points vs the average

def init_loss_parameters(N, players_until_me):
    next_players_until_me = players_until_me - 1
    loss_21 = 1/(N-1)
    loss_proc_doubt_correct = -1/(N-1)
    loss_proc_doubt_miss = -1/(N-1)
    if players_until_me == 0:
        next_players_until_me = N - 1
        loss_21 = -1
        loss_proc_doubt_correct = 1
    elif players_until_me == 1:
        loss_proc_doubt_miss = 1
    return loss_21, loss_proc_doubt_correct, loss_proc_doubt_miss, next_players_until_me

def p_lie_unnecessarily(N, last_claim, players_until_me):
    _, loss_proc_doubt_correct, loss_proc_doubt_miss, next_players_until_me = init_loss_parameters(N, players_until_me)
    mu_best_lie, mu_throw_true_claim = calculate_lines(N, last_claim, next_players_until_me, loss_proc_doubt_miss, loss_proc_doubt_correct)
    return sum(P[mu_throw_true_claim > mu_best_lie])

def p_lie(N, last_claim, players_until_me):
    p_had_to_lie = Q[last_claim]/(1-P21)
    return p_had_to_lie + p_lie_unnecessarily(N, last_claim, players_until_me)

def mu_throw(N, last_claim, players_until_me):
    """ Expected outcome for me, if the player throws. """
    loss_21, loss_proc_doubt_correct, loss_proc_doubt_miss, next_players_until_me = init_loss_parameters(N, players_until_me)

    _mu_throw = P21 * loss_21
    if last_claim == V[-1]:
        _mu_throw += (1 - P21) * loss_proc_doubt_correct
    else:
        mu_best_lie, mu_throw_true_claim = calculate_lines(N, last_claim, next_players_until_me, loss_proc_doubt_miss, loss_proc_doubt_correct)
        mu_throw_true_claim[mu_throw_true_claim > mu_best_lie] = mu_best_lie
        _mu_throw += Q[last_claim] * mu_best_lie
        _mu_throw += np.inner(P, mu_throw_true_claim)
    return _mu_throw

def calculate_lines(N, last_claim, next_players_until_me, loss_proc_doubt_miss, loss_proc_doubt_correct):
    mu_throw_false_claim = np.ones(20)
    mu_throw_true_claim = np.zeros(20)
    for claim in V[last_claim + 1:]:
        mu_proc_throw = mu_throw(N, claim, next_players_until_me)
        p_proc_doubt = do_doubt(N, last_claim, claim)
        weighted_mu_proc_throw = (1 - p_proc_doubt) * mu_proc_throw
        mu_throw_true_claim[claim] = weighted_mu_proc_throw + p_proc_doubt * loss_proc_doubt_miss
        mu_throw_false_claim[claim] = weighted_mu_proc_throw + p_proc_doubt * loss_proc_doubt_correct
    # assumption: always use the optimal lie
    mu_best_lie = min(mu_throw_false_claim)
    return mu_best_lie, mu_throw_true_claim

def mu_doubt(N, claim_m2, players_until_me):
    """
    Expected outcome, if I doubt the last last claim.
    :param N: number of players
    :param claim_m2: claim made by P_-2
    """
    return 1 - N/(N-1) * p_lie(N, claim_m2, players_until_me)

def do_doubt(N, claim_m2, claim_m1):
    """
    :param N: number of players
    :param claim_m2: claim made by the player before the previous one
    :param claim_m1: claim made by the previous player
    """
    if not claim_m1 in V[claim_m2 + 1:]:
        return True
    # assumption: always take the optimal action
    return mu_doubt(N, claim_m2, 0) < mu_throw(N, claim_m1, 0)

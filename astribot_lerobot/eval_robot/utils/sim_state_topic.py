class SimStateSubscriber:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Simulation state subscription is not available in the Astribot S1 package.")


class SimRewardSubscriber:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Simulation reward subscription is not available in the Astribot S1 package.")


def start_sim_state_subscribe(*args, **kwargs) -> SimStateSubscriber:
    return SimStateSubscriber(*args, **kwargs)


def start_sim_reward_subscribe(*args, **kwargs) -> SimRewardSubscriber:
    return SimRewardSubscriber(*args, **kwargs)

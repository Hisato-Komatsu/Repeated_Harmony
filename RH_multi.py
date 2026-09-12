# Repeated 2x2 game
# Simulate multiple environments simultaneously
# both agents 0 and 1 learn the policy
# geometric decay of epsilon 
import sys
import time
import csv

import numpy as np

# seed of randm value
# seed = 1 and n_env = 20 or 100 (20 or 100 trials) when hyperparameter tuning, and
# seed = 42 and n_env = 1000 (1000 trials) when calculating the reportrd data
seed = 42
rng = np.random.default_rng(seed=seed)

class hyperparameters:
    def __init__(self):
        self.gamma = 0.95
        self.epsilon_init = 0.02
        self.epsilon_decay_rate = 1.0e-6 # per one step
        self.epsilon_decay = 1.0 - self.epsilon_decay_rate
        self.epsilon_min = 0.002
        self.learning_rate = 0.1
        self.cooperate = 3.0 # R
        self.temptation = 2.0 # T
        self.sucker = 1.0 # S
        self.punishment = 0.0 # P

        self.n_env = 1000

        self.t_episode = 100
        self.t_observe = 10000
        self.t_whole = 10_000_000

        #self.Q_init = 'f_handshake'
        #self.Q_init = 'wsls'
        self.Q_init = 'zeros'
        self.Q_init_strength = 1.0


class Agent:
    def __init__(self, n_obs, n_act, hyperparameters):
        self.n_observation = n_obs
        self.n_action = n_act
        self.n_env = hyperparameters.n_env
        self.env_ids = np.arange(self.n_env)

        #initialization of the Q-table
        Q_init_strength = hyperparameters.Q_init_strength
        if hyperparameters.Q_init == 'f_handshake':
            self.Q_table = np.zeros((self.n_env, self.n_observation, self.n_action))
            self.Q_table[:, 0:3, 1] = Q_init_strength
            self.Q_table[:, 3, 0] = Q_init_strength
        elif hyperparameters.Q_init == 'wsls':
            self.Q_table = np.zeros((self.n_env, self.n_observation, self.n_action))
            self.Q_table[:, 1:3, 1] = Q_init_strength
            self.Q_table[:, 0, 0] = Q_init_strength
            self.Q_table[:, 3, 0] = Q_init_strength
        elif hyperparameters.Q_init == 'zeros':
            self.Q_table = np.zeros((self.n_env, self.n_observation, self.n_action))
        else:
            raise ValueError('invalid option for Q_init')

        self.gamma = hyperparameters.gamma
        self.epsilon = hyperparameters.epsilon_init
        self.epsilon_decay = hyperparameters.epsilon_decay
        self.epsilon_min = hyperparameters.epsilon_min
        self.learning_rate = hyperparameters.learning_rate

    def choose_action(self, s0):
        #choose action using epsilon-greedy method
        p = rng.uniform(low=0.0, high=1.0, size=(self.n_env, ))

        # random tie-breaking rule for choosing argmax
        Q_max = np.max(
            self.Q_table[self.env_ids, s0, :],
            axis=1,
        )
        mask = (self.Q_table[self.env_ids, s0, :] == Q_max[:, None])
        rand_mask = rng.uniform(size=mask.shape)
        rand_mask[~mask] = -np.inf
        action_argmax = np.argmax(
            rand_mask, axis=1,
        )

        action_rand = rng.integers(
            self.n_action,
            size=(self.n_env, ),
        ) 

        action_chosen = np.where(
            p < self.epsilon, action_rand, action_argmax,
        )
        return action_chosen

    def update_Q(self, s0, a0, r0, ns0, d0):
        TD = (
            r0
            + self.gamma * (1.0 - float(d0))
            * np.max(self.Q_table[self.env_ids, ns0, :], axis=1)
            - self.Q_table[self.env_ids, s0, a0]
        )
        self.Q_table[self.env_ids, s0, a0] = (
            self.Q_table[self.env_ids, s0, a0]
            + self.learning_rate * TD
        )

    def decaying_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)


class MakeEnv:
    def __init__(self, hyperparameters):
        self.temptation = hyperparameters.temptation
        self.cooperate = hyperparameters.cooperate
        self.punishment = hyperparameters.punishment
        self.sucker = hyperparameters.sucker

        self.payoff0 = np.array([
            [self.cooperate, self.sucker],
            [self.temptation, self.punishment]
        ])
        self.payoff1 = np.array([
            [self.cooperate, self.temptation],
            [self.sucker, self.punishment]
        ])

        self.n_env = hyperparameters.n_env

    def step(self, a0, a1):
        ns_number = 2 * a0 + a1
        r0 = self.payoff0[a0, a1]
        r1 = self.payoff1[a0, a1]
        return r0, r1, ns_number

    def reset(self):
        return rng.integers(2 ** 2, size=(self.n_env, ))


if __name__ == '__main__':
    start_time = time.time()
    #preparation of environment and agent
    hyper = hyperparameters()
    t_episode = hyper.t_episode
    t_observe = hyper.t_observe
    t_whole = hyper.t_whole
    env = MakeEnv(hyper)
    agent = [None, None]
    agent[0] = Agent(2 ** 2, 2, hyper)
    agent[1] = Agent(2 ** 2, 2, hyper)
    mem_rewards = []

    name_condition = (
        '_gamma_' + str(hyper.gamma)
        + '_lr_' + str(hyper.learning_rate)
        + '_RTSP_' + str(hyper.cooperate) + '_' + str(hyper.temptation)
        + '_' + str(hyper.sucker) + '_' + str(hyper.punishment) 
        + '_epsilon_' + str(hyper.epsilon_init)
        + '_' + str(hyper.epsilon_decay_rate)
        + '_' + str(hyper.epsilon_min)
        + '_Qinit_' + hyper.Q_init + '_' + str(hyper.Q_init_strength)
        + '_seed_' + str(seed) + '_' + str(hyper.n_env) + 'trials'
    )
    filename_reward = 'reward' + name_condition + '.tsv'
    file_reward = open(filename_reward, "w")
    writer_reward = csv.writer(file_reward, delimiter="\t")
    filename_Q0 = 'Qfunc0' + name_condition + '.tsv'
    file_Q0 = open(filename_Q0, "w")
    writer_Q0 = csv.writer(file_Q0, delimiter="\t")
    filename_Q1 = 'Qfunc1' + name_condition + '.tsv'
    file_Q1 = open(filename_Q1, "w")
    writer_Q1 = csv.writer(file_Q1, delimiter="\t")
    filename_Q_th = 'Qfunc_theory' + name_condition + '.tsv'
    file_Q_th = open(filename_Q_th, "w")
    writer_Q_th = csv.writer(file_Q_th, delimiter="\t")
    filename_info = 'info' + name_condition + '.txt'
    file_info = open(filename_info, "w")
    filename_rawQ = 'rawQ' + name_condition + '.txt'
    file_rawQ = open(filename_rawQ, "w")
    print('Python', sys.version, file=file_info)
    print('NumPy', np.__version__, file=file_info)
    
    state = env.reset()
    for t in range(t_whole):
        action = [None, None]
        reward = [None, None]

        for i_agent in range(2):
            action[i_agent] = agent[i_agent].choose_action(state)
        reward[0], reward[1], next_state = env.step(action[0], action[1])
        mem_rewards.append(
            np.stack([reward[0], reward[1]])
        )

        for i_agent in range(2):
            # Truncation is not treated as an episode termination when updating Q.
            agent[i_agent].update_Q(state, action[i_agent], reward[i_agent], next_state, False)
            # epsilon-dacaying is executed every step.
            agent[i_agent].decaying_epsilon()

        if (t + 1) % t_episode == 0:
            state = env.reset()
        else:
            state = next_state

        if (t + 1) % t_observe == 0:
            # moving average
            # np.stack(mem_rewards) ... t_observe * 2 (agent 0 and 1) * n_env array
            ma_reward = np.mean(np.stack(mem_rewards), axis=0)
            ma_rew_mean = np.mean(ma_reward, axis=1)
            ma_rew_std = np.std(ma_reward, axis=1, ddof=1)
            mem_rewards = []
            writer_reward.writerow(
                [t + 1, ma_rew_mean[0], ma_rew_std[0], ma_rew_mean[1], ma_rew_std[1]],
            )
            # Q-function at each time (NOT a moving average)
            Q_table_0 = np.mean(agent[0].Q_table, axis=0)
            Q_table_0_std = np.std(agent[0].Q_table, axis=0, ddof=1)
            Q_table_0_quant = np.quantile(agent[0].Q_table, [0.25, 0.5, 0.75], axis=0)
            writer_Q0.writerow([
                t + 1, Q_table_0[0, 0], Q_table_0_std[0, 0],
                Q_table_0_quant[0, 0, 0], Q_table_0_quant[1, 0, 0], Q_table_0_quant[2, 0, 0],
                Q_table_0[0, 1], Q_table_0_std[0, 1], 
                Q_table_0_quant[0, 0, 1], Q_table_0_quant[1, 0, 1], Q_table_0_quant[2, 0, 1],
                Q_table_0[1, 0], Q_table_0_std[1, 0],
                Q_table_0_quant[0, 1, 0], Q_table_0_quant[1, 1, 0], Q_table_0_quant[2, 1, 0],
                Q_table_0[1, 1], Q_table_0_std[1, 1],
                Q_table_0_quant[0, 1, 1], Q_table_0_quant[1, 1, 1], Q_table_0_quant[2, 1, 1],
                Q_table_0[2, 0], Q_table_0_std[2, 0],
                Q_table_0_quant[0, 2, 0], Q_table_0_quant[1, 2, 0], Q_table_0_quant[2, 2, 0],
                Q_table_0[2, 1], Q_table_0_std[2, 1],
                Q_table_0_quant[0, 2, 1], Q_table_0_quant[1, 2, 1], Q_table_0_quant[2, 2, 1],
                Q_table_0[3, 0], Q_table_0_std[3, 0],
                Q_table_0_quant[0, 3, 0], Q_table_0_quant[1, 3, 0], Q_table_0_quant[2, 3, 0],
                Q_table_0[3, 1], Q_table_0_std[3, 1],
                Q_table_0_quant[0, 3, 1], Q_table_0_quant[1, 3, 1], Q_table_0_quant[2, 3, 1],
            ])
            Q_table_1 = np.mean(agent[1].Q_table, axis=0)
            Q_table_1_std = np.std(agent[1].Q_table, axis=0, ddof=1)
            Q_table_1_quant = np.quantile(agent[1].Q_table, [0.25, 0.5, 0.75], axis=0)
            writer_Q1.writerow([
                t + 1, Q_table_1[0, 0], Q_table_1_std[0, 0],
                Q_table_1_quant[0, 0, 0], Q_table_1_quant[1, 0, 0], Q_table_1_quant[2, 0, 0],
                Q_table_1[0, 1], Q_table_1_std[0, 1],
                Q_table_1_quant[0, 0, 1], Q_table_1_quant[1, 0, 1], Q_table_1_quant[2, 0, 1],
                Q_table_1[1, 0], Q_table_1_std[1, 0],
                Q_table_1_quant[0, 1, 0], Q_table_1_quant[1, 1, 0], Q_table_1_quant[2, 1, 0],
                Q_table_1[1, 1], Q_table_1_std[1, 1],
                Q_table_1_quant[0, 1, 1], Q_table_1_quant[1, 1, 1], Q_table_1_quant[2, 1, 1],
                Q_table_1[2, 0], Q_table_1_std[2, 0],
                Q_table_1_quant[0, 2, 0], Q_table_1_quant[1, 2, 0], Q_table_1_quant[2, 2, 0],
                Q_table_1[2, 1], Q_table_1_std[2, 1],
                Q_table_1_quant[0, 2, 1], Q_table_1_quant[1, 2, 1], Q_table_1_quant[2, 2, 1],
                Q_table_1[3, 0], Q_table_1_std[3, 0],
                Q_table_1_quant[0, 3, 0], Q_table_1_quant[1, 3, 0], Q_table_1_quant[2, 3, 0],
                Q_table_1[3, 1], Q_table_1_std[3, 1],
                Q_table_1_quant[0, 3, 1], Q_table_1_quant[1, 3, 1], Q_table_1_quant[2, 3, 1],
            ])

    Q_table_0 = np.mean(agent[0].Q_table, axis=0)
    Q_table_1 = np.mean(agent[1].Q_table, axis=0)
    print(
        Q_table_0, "\n",
        Q_table_1,
        file=file_info,
    )
    print(
        Q_table_0[:, 0] > Q_table_0[:, 1], "\n",
        Q_table_1[:, 0] > Q_table_1[:, 1],
        file=file_info,    
    )

    Q_00_theory_allC = hyper.cooperate / (1.0 - hyper.gamma)
    Q_01_theory_allC = hyper.temptation + hyper.gamma * Q_00_theory_allC
    writer_Q_th.writerow([
        'allC',
        Q_00_theory_allC, Q_01_theory_allC,
        Q_00_theory_allC, Q_01_theory_allC,
        Q_00_theory_allC, Q_01_theory_allC,
        Q_00_theory_allC, Q_01_theory_allC,
    ])

    if hyper.gamma * (hyper.cooperate - hyper.punishment) > hyper.sucker - hyper.punishment:
        # The theoretical Q-vale under mutual WSLS
        Q_00_theory_wsls = hyper.cooperate / (1.0 - hyper.gamma)
        Q_11_theory_wsls = hyper.punishment + hyper.gamma * Q_00_theory_wsls
        Q_01_theory_wsls = hyper.temptation + hyper.gamma * Q_11_theory_wsls
        Q_10_theory_wsls = hyper.sucker + hyper.gamma * Q_11_theory_wsls
        writer_Q_th.writerow([
            'WSLS',
            Q_00_theory_wsls, Q_01_theory_wsls,
            Q_10_theory_wsls, Q_11_theory_wsls,
            Q_10_theory_wsls, Q_11_theory_wsls,
            Q_00_theory_wsls, Q_01_theory_wsls,
        ])
    if hyper.gamma * (hyper.cooperate - hyper.sucker) > hyper.sucker - hyper.punishment:
        # The theoretical Q-vale under mutual FH
        Q_01_theory_fh = (hyper.punishment + hyper.gamma * hyper.cooperate) / (1.0 - hyper.gamma ** 2)
        Q_00_theory_fh = hyper.sucker + hyper.gamma * Q_01_theory_fh
        Q_30_theory_fh = (hyper.cooperate + hyper.gamma * hyper.punishment) / (1.0 - hyper.gamma ** 2)
        Q_31_theory_fh = hyper.temptation + hyper.gamma * Q_01_theory_fh
        writer_Q_th.writerow([
            'FH',
            Q_00_theory_fh, Q_01_theory_fh,
            Q_00_theory_fh, Q_01_theory_fh,
            Q_00_theory_fh, Q_01_theory_fh,
            Q_30_theory_fh, Q_31_theory_fh,
        ])

    Cop = [None, None]
    Def = [None, None]
    wsls = [None, None]
    allC = [None, None]
    f_handshake = [None, None]
    for i_agent in range(2):
        Cop[i_agent] = agent[i_agent].Q_table[:, :, 0] > agent[i_agent].Q_table[:, :, 1]
        Def[i_agent] = agent[i_agent].Q_table[:, :, 0] < agent[i_agent].Q_table[:, :, 1]

        wsls[i_agent] = np.logical_and.reduce([
            Cop[i_agent][:, 0], 
            Def[i_agent][:, 1],
            Def[i_agent][:, 2],
            Cop[i_agent][:, 3],
        ])
        allC[i_agent] = np.logical_and.reduce([
            Cop[i_agent][:, 0], 
            Cop[i_agent][:, 1],
            Cop[i_agent][:, 2],
            Cop[i_agent][:, 3],
        ])
        f_handshake[i_agent] = np.logical_and.reduce([
            Def[i_agent][:, 0], 
            Def[i_agent][:, 1],
            Def[i_agent][:, 2],
            Cop[i_agent][:, 3],
        ])

    print('allC',
        np.count_nonzero(allC[0]), np.count_nonzero(allC[1]),
        np.count_nonzero(allC[0] & allC[1]),
        file=file_info,
    )
    where_allC = (allC[0] & allC[1])[:, None, None]
    if np.any(where_allC):
        Q_table_0_allC = np.mean(agent[0].Q_table, axis=0, where=where_allC)
        Q_table_0_std_allC = (
            np.std(agent[0].Q_table, axis=0, where=where_allC, ddof=1) if where_allC.sum() > 1
            else np.nan
        )
        Q_table_1_allC = np.mean(agent[1].Q_table, axis=0, where=where_allC)
        Q_table_1_std_allC = (
            np.std(agent[1].Q_table, axis=0, where=where_allC, ddof=1) if where_allC.sum() > 1
            else np.nan
        )
        print(
            Q_table_0_allC, "\n",
            Q_table_0_std_allC, "\n",
            Q_table_1_allC, "\n",
            Q_table_1_std_allC,    
            file=file_info,
        )
    
    print('WSLS',
        np.count_nonzero(wsls[0]), np.count_nonzero(wsls[1]),
        np.count_nonzero(wsls[0] & wsls[1]),
        file=file_info,
    )
    where_wsls = (wsls[0] & wsls[1])[:, None, None]
    if np.any(where_wsls):
        Q_table_0_wsls = np.mean(agent[0].Q_table, axis=0, where=where_wsls)
        Q_table_0_std_wsls = (
            np.std(agent[0].Q_table, axis=0, where=where_wsls, ddof=1) if where_wsls.sum() > 1
            else np.nan
        )
        Q_table_1_wsls = np.mean(agent[1].Q_table, axis=0, where=where_wsls)
        Q_table_1_std_wsls = (
            np.std(agent[1].Q_table, axis=0, where=where_wsls, ddof=1) if where_wsls.sum() > 1
            else np.nan
        )
        print(
            Q_table_0_wsls, "\n",
            Q_table_0_std_wsls, "\n",
            Q_table_1_wsls, "\n",
            Q_table_1_std_wsls,    
            file=file_info,
        )
    
    print('FH',
        np.count_nonzero(f_handshake[0]), np.count_nonzero(f_handshake[1]),
        np.count_nonzero(f_handshake[0] & f_handshake[1]),
        file=file_info,
    )
    where_fh = (f_handshake[0] & f_handshake[1])[:, None, None]
    if np.any(where_fh):
        Q_table_0_fh = np.mean(agent[0].Q_table, axis=0, where=where_fh)
        Q_table_0_std_fh = (
            np.std(agent[0].Q_table, axis=0, where=where_fh, ddof=1) if where_fh.sum() > 1
            else np.nan
        )
        Q_table_1_fh = np.mean(agent[1].Q_table, axis=0, where=where_fh)
        Q_table_1_std_fh = (
            np.std(agent[1].Q_table, axis=0, where=where_fh, ddof=1) if where_fh.sum() > 1
            else np.nan
        )
        print(
            Q_table_0_fh, "\n",
            Q_table_0_std_fh, "\n",
            Q_table_1_fh, "\n",
            Q_table_1_std_fh,    
            file=file_info,
        )
    
    for i_env in range(hyper.n_env):
        print(
            i_env, 
            'allC', allC[0][i_env], allC[1][i_env],
            'WSLS', wsls[0][i_env], wsls[1][i_env],
            'FH', f_handshake[0][i_env], f_handshake[1][i_env],
            file=file_rawQ,
        )
        print(
            agent[0].Q_table[i_env], "\n",
            Cop[0][i_env], "\n",
            agent[1].Q_table[i_env], "\n",
            Cop[1][i_env],
            file=file_rawQ,
        )

    file_reward.close()
    file_Q0.close()
    file_Q1.close()
    file_Q_th.close()
    file_info.close()
    file_rawQ.close()
    print(time.time() - start_time)

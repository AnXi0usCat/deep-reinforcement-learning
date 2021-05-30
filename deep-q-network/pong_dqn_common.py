import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from types import SimpleNamespace
import ptan
import ptan.ignite as ptan_ignite
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from lib import model


SEED = 123

HYPERPARAMS = {
    'pong': SimpleNamespace(**{
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    })
}


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    
    for exp in batch:

        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        
        if exp.last_state is None:
            lstate = state
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
            
    return np.array(states, copy=False), np.array(actions), \ 
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    dones_mask = torch.BoolTensor(dones).to(device)
    
    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    
    with torch.no_grad():
        next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[dones_mask] = 0.0
        
    
    expected_action_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, expected_action_vals)


def setup_ignite(engine: Engine, params: SimpleNamespace,
                 exp_source, run_name: str,
                 extra_metrics: Iterable[str] = ()):
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    handler = ptan_ignite.EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.0f, steps=%s, "
              "speed=%.1f f/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed))))

    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes "
              "and %d iterations!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    now = datetime.now().isoformat(timespec='minutes').replace(':', '')
    logdir = f"runs/{now}-{params.run_name}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    metrics = ['reward', 'steps', 'avg_reward']
    handler = tb_logger.OutputHandler(
        tag="episodes", metric_names=metrics)
    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # write to tensorboard every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(
        tag="train", metric_names=metrics,
        output_transform=lambda a: a)
    event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)
    

class EpsilonTracker:
    
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - \
              (frame_idx / self.params.epsilon_frames)
        self.selector.epsilon = max(self.params.epsilon_final, eps)


if __name__ == '__main__':

    random.seed(SEED)
    torch.manual_seed(SEED)
    params = common.HYPERPARAMS['pong']
    
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(SEED)
    
    net = model.DQN(env.observation_space.shape,
                    env.action_space.n).to(device)
    
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params.epsilon_start)
    epsilon_tracker = EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(),
                           lr=params.learning_rate)
    
    
    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = calc_loss_dqn(
            batch, net, tgt_net.target_model, 
            params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME)
    engine.run(common.batch_generator(buffer, params.replay_initial,
                                      params.batch_size))
    
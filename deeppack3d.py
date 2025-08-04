def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('method', metavar='method', 
                        type=str, choices=['rl', 'bl', 'baf', 'bssf', 'blsf'], 
                        help='choose the method from {"rl", "bl", "baf", "bssf", "blsf"}.')
    
    parser.add_argument('lookahead', metavar='lookahead', 
                        type=int,
                        help='choose the lookahead value.')
    
    parser.add_argument('--data', metavar='', 
                        type=str, default='custom', choices=['generated', 'input', 'file', 'custom'], 
                        help='choose the input source from {"generated", "input", "file", "custom"} (default: custom).')
    
    parser.add_argument('--path', metavar='', 
                        type=str, default=None, 
                        help='set the file path, only used if --data is "file" (default: None).')
    
    parser.add_argument('--n_iterations', metavar='', 
                        type=int, default=100, 
                        help='set the number of iterations, only used if --data is "generated" (default: 100).')
    
    parser.add_argument('--seed', metavar='', 
                        type=str, default=None, 
                        help='set the random seed for reproducibility, only used if --data is "generated" (default: None).')
    
    parser.add_argument('--verbose', metavar='', 
                        type=int, default=1, 
                        help='set verbose level (default: 1).')
    
    parser.add_argument('--train', 
                        action='store_true', 
                        help='enable training mode, only used if method is "rl" (default: False).')
    
    parser.add_argument('--batch_size', metavar='', 
                        type=int, default=32, 
                        help='set batch_size, only used if train is True (default: 32).')
    
    parser.add_argument('--visualize', 
                        action='store_true', 
                        help='enable visualization mode (default: False).')
    
    return parser.parse_args()

import numpy as np
import os, shutil, time
import matplotlib.pyplot as plt
import tensorflow as tf

# Optional imports for training
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from env import *
from agent import *

# Custom bin size and packages
CUSTOM_BIN_SIZE = (100, 140, 120)  # Width, Height, Depth

CUSTOM_PACKAGES = [
    (24, 8, 17),   # 240x80x170 -> 24x8x17
    (24, 16, 17),  # 240x160x170 -> 24x16x17
    (34, 8, 24),   # 340x80x240 -> 34x8x24
    (34, 16, 24),  # 340x160x240 -> 34x16x24
    (40, 10, 40),  # 400x100x400 -> 40x10x40
    (40, 12, 40),  # 400x120x400 -> 40x12x40
    (40, 14, 40),  # 400x140x400 -> 40x14x40
    (40, 16, 40),  # 400x160x400 -> 40x16x40
    (40, 18, 40),  # 400x180x400 -> 40x18x40
    (40, 20, 40),  # 400x200x400 -> 40x20x40
    (60, 10, 40),  # 600x100x400 -> 60x10x40
    (60, 12, 40),  # 600x120x400 -> 60x12x40
    (60, 14, 40),  # 600x140x400 -> 60x14x40
    (60, 16, 40),  # 600x160x400 -> 60x16x40
    (60, 18, 40),  # 600x180x400 -> 60x18x40
    (60, 20, 40),  # 600x200x400 -> 60x20x40
]

heuristics = {
    'bl': bottom_left,
    'baf': best_area_fit, 
    'bssf': best_short_side_fit, 
    'blsf': best_long_side_fit, 
}

def deeppack3d(method, lookahead, *, n_iterations=100, seed=None, verbose=1, data='custom', path=None, train=False, visualize=False, batch_size=32):
    reset_rng(seed)
    
    env = MultiBinPackerEnv(n_bins=1, 
                            max_bins=1, 
                            size=CUSTOM_BIN_SIZE, 
                            k=lookahead, 
                            prealloc_items=100, 
                            verbose=verbose)

    if data == 'file':
        if path is not None:
            env.conveyor = FileConveyor(k=env.k, path=path).reset()
        else:
            raise ValueError("Path must be provided when data='file'")
    elif data == 'input':
        env.conveyor = InputConveyor(k=env.k).reset()
    elif data == 'custom':
        # Use custom packages
        repeated_packages = CUSTOM_PACKAGES * (n_iterations // len(CUSTOM_PACKAGES) + 10)
        np.random.shuffle(repeated_packages)
        env.conveyor = Conveyor(k=env.k, assigned_items=repeated_packages)

    if visualize:
        if os.path.exists('./outputs'):
            shutil.rmtree('./outputs')
        os.makedirs('./outputs')

    if train:
        print(f'Training with method "{method}" and lookahead {lookahead}...')
        
        if method != 'rl':
            raise Exception('training mode can only be used if method is "rl"')

        # env = BinPackerEnv(size=(32, 32, 32), k=env.k, bin_size=(32, 32, 32))
        agent = Agent(env, train=True, verbose=verbose > 0, visualize=visualize, batch_size=batch_size)

        agent.eps = 1.0
        for i in range(n_iterations):
            print(f'Iteration {i}')
            start_time = time.time()
            yield from agent.run(100, verbose=verbose > 1)
            agent.eps = max(agent.eps * 0.95, 0.025)
            
        data = np.asarray([utils for utils, n_bins, ep_reward in agent.ep_history])
        # y = np.ones(100)
        # data = np.convolve(data, y, 'valid') / len(y)
        if HAS_SEABORN:
            sns.lineplot(data=data)
            plt.savefig(f'./util.jpg')
            plt.show()
        else:
            plt.plot(data)
            plt.savefig(f'./util.jpg')
            plt.show()
        
        data = np.asarray([ep_reward for utils, n_bins, ep_reward in agent.ep_history])
        # y = np.ones(100)
        # data = np.convolve(data, y, 'valid') / len(y)
        if HAS_SEABORN:
            sns.lineplot(data=data)
            plt.savefig(f'./ep_reward.jpg')
            plt.show()
        else:
            plt.plot(data)
            plt.savefig(f'./ep_reward.jpg')
            plt.show()

        import uuid
        uid = uuid.uuid4()
        print(f'saved model at ./{uid}.h5')
        if agent.q_net is not None:
            agent.q_net.save(f'{uid}.h5')
        else:
            print("Warning: No model to save (q_net is None)")
    else:
        if verbose > 0:
            print(f'Testing with method "{method}" and lookahead {lookahead}...')
            print(f'Custom bin size: {CUSTOM_BIN_SIZE}')
            print(f'Package types: {len(CUSTOM_PACKAGES)}')
        
        if method == 'rl':
            model_path = f'./models/k={lookahead}.h5'
            agent = Agent(env, train=False, verbose=verbose > 0, visualize=visualize, batch_size=batch_size)
            try:
                agent.q_net = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Please train a model first or use a heuristic method")
                return
            agent.eps = 0.0
        else:
            agent = HeuristicAgent(heuristics[method], env, verbose=verbose > 0, visualize=visualize)
        
        start_time = time.time()
        
        try:
            yield from agent.run(n_iterations, verbose=verbose > 1)
        except Exception as e:
            if np.all(np.array(env.conveyor.reset().peek()) == None):
                if verbose > 0:
                    print('\n=====the end of conveyor line=====')
            else:
                print(e)

        if verbose > 0:
            print()
            next_items = np.array(env.conveyor.reset().peek()).tolist()
            avg_util = np.mean([util for utils, n_bins, ep_reward in agent.ep_history[:] for util in utils[:]])
            used_items = np.sum([n_bins for utils, n_bins, ep_reward in agent.ep_history[:] for util in utils[:]])
            
            print(f'Used time: {int(time.time() - start_time)} seconds')
            print(f'Next items: {next_items}')
            print(f'Average space util: {avg_util}')
            print(f'Used bins: {used_items}')

def main():
    args = parse_args()
    
    reset_rng(args.seed)

    for _ in deeppack3d(args.method, 
                        args.lookahead, 
                        n_iterations=args.n_iterations, 
                        seed=args.seed, 
                        train=args.train, 
                        verbose=args.verbose, 
                        data=args.data, 
                        path=args.path,
                        visualize=args.visualize, 
                        batch_size=args.batch_size):
        pass

if __name__ == "__main__":
    main()
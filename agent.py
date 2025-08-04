import numpy as np

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LeakyReLU, Lambda, Conv2D, GlobalAveragePooling2D, Flatten, Dense, MaxPooling2D, Conv2DTranspose, RepeatVector, Reshape, concatenate, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import orthogonal

from env import *
import collections, itertools

def q_net(k=1, bin_size=(100, 140, 120)):
    weight_decay = 0.0005
    
    # Calculate input dimensions based on bin size
    # Use depth and width for the 2D height map (D, W)
    input_d = bin_size[2]  # depth = 120
    input_w = bin_size[0]  # width = 100
    
    # Downsample for manageable input size
    downsample_factor = 4
    map_d = max(8, input_d // downsample_factor)  # At least 8x8
    map_w = max(8, input_w // downsample_factor)
    
    hmap_in = Input((map_d, map_w, 1))
    amap_in = Input((map_d, map_w, 1))
    imap_in = Input((k, 3))
    imap_x = Flatten()(imap_in)
    const_in = Input((map_d, map_w, 1))
    
    x = concatenate([hmap_in, amap_in, const_in], axis=-1)
    
    # Adjusted architecture for different input sizes
    x = Conv2D(32, 5, strides=2, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, strides=2, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, strides=2, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, 3, strides=1, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x)
    
    emb = Dense(128, kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(imap_x)
    
    x = concatenate([x, emb], axis=-1)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    
    x = Dense(1, activation='linear')(x)
    
    outputs = x
    model = Model([const_in, hmap_in, amap_in, imap_in], outputs)
    return model

class Agent:
    def __init__(self, env=MultiBinPackerEnv(n_bins=2, max_bins=-1, size=(32, 32, 32), k=10, verbose=True), train=True, verbose=True, visualize=False, batch_size=32):
        self.env = env
        
        self.gamma = 0.95
        
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.99
        
        self.ep_history = []
        
        self.warmup_epochs = 20
        self.warmup_lr = 1e-5
        self.learning_rate = 1e-3
        self.lr_min = 1e-5
        self.lr_drop = 10000
        self.epoch = 0
        self.update_epochs = 10

        self.batch_size = batch_size
        
        self.__train = train

        self.verbose = verbose
        self.visualize = visualize
        
        # Calculate downsampling for the new bin size
        self.bin_size = env.size
        self.downsample_factor = 4
        self.map_d = max(8, self.bin_size[2] // self.downsample_factor)  # depth
        self.map_w = max(8, self.bin_size[0] // self.downsample_factor)  # width
        
        if self.__train:
            self.q_net = q_net(k=env.k - 1, bin_size=env.size)
            self.q_net_target = q_net(k=env.k - 1, bin_size=env.size)
            self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=self.warmup_lr)
            self.memory = collections.deque(maxlen=1000000)
        else:
            self.q_net = None
            self.q_net_target = None
            self.q_optimizer = None
            self.memory = None
    
    def select(self, state):
        items, h_maps, actions = state
        action_space = indices(actions)
        
#         print('actions: ', len(action_space))
        
        r = np.random.random()
        if r < self.eps:
            action = action_space[np.random.choice(len(action_space))]
        else:
            q = self.Q(state)
            action = action_space[np.argmax(q)]
            r = np.max(q)
            
        return action, r
    
    def Q_inputs(self, state, action=None):
        W, H, D = self.env.size
        
        items, h_maps, actions = state
        if action is None:
            action_space = indices(actions)
        else:
            i, j, k = action
            action_space = [(i, j, k)]
            
        imaps = [self.env.i_map(i, items) for i in range(len(self.env.packers))]
            
        hmap_in = []
        amap_in = []
        imap_in = []
        
        # item, bin, rotation_placement
        for i, j, k in action_space:
            _, (x, y, z), (w, h, d), _ = actions[i][j][k]
            
            # Create action map
            amap = np.zeros((D, W))  # height map is (depth, width)
            if z + d <= D and x + w <= W:
                amap[z:z + d, x:x + w] = (y + h) / H
            
            # Get height map and normalize
            hmap = h_maps[j].copy().astype(float) / H
            
            # Ensure dimensions match
            if hmap.shape != amap.shape:
                min_d, min_w = min(hmap.shape[0], amap.shape[0]), min(hmap.shape[1], amap.shape[1])
                hmap = hmap[:min_d, :min_w]
                amap = amap[:min_d, :min_w]
            
            # Combine maps
            combined_amap = np.where(amap == 0, hmap, amap)
            
            # Downsample for neural network
            ds_d = max(1, hmap.shape[0] // self.map_d)
            ds_w = max(1, hmap.shape[1] // self.map_w)
            
            hmap_ds = hmap[::ds_d, ::ds_w]
            amap_ds = combined_amap[::ds_d, ::ds_w]
            
            # Ensure exact target size
            if hmap_ds.shape[0] > self.map_d:
                hmap_ds = hmap_ds[:self.map_d, :]
                amap_ds = amap_ds[:self.map_d, :]
            if hmap_ds.shape[1] > self.map_w:
                hmap_ds = hmap_ds[:, :self.map_w]
                amap_ds = amap_ds[:, :self.map_w]
            
            # Pad if necessary
            if hmap_ds.shape[0] < self.map_d or hmap_ds.shape[1] < self.map_w:
                pad_d = self.map_d - hmap_ds.shape[0]
                pad_w = self.map_w - hmap_ds.shape[1]
                hmap_ds = np.pad(hmap_ds, ((0, pad_d), (0, pad_w)), mode='edge')
                amap_ds = np.pad(amap_ds, ((0, pad_d), (0, pad_w)), mode='edge')

            imap = imaps[j][np.arange(len(items)) != i]

            hmap_in.append(hmap_ds)
            amap_in.append(amap_ds)
            imap_in.append(imap)
            
        hmap_in, amap_in, imap_in = map(np.asarray, (hmap_in, amap_in, imap_in))
        hmap_in = hmap_in[..., None]
        amap_in = amap_in[..., None]
        const_in = np.ones(hmap_in.shape)
        
        return [const_in, hmap_in, amap_in, imap_in]
    
    def Q(self, state, action=None):
        const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)
        
        batch_size = self.batch_size
        sections = np.cumsum([self.batch_size] * int(np.ceil(const_in.shape[0] / batch_size) - 1))
        batches = map(lambda data: map(lambda x: x.copy(), np.split(data, sections, axis=0)), (const_in, hmap_in, amap_in, imap_in))
        
        outputs = []
        for const_in, hmap_in, amap_in, imap_in in zip(*batches):
            # print(const_in.shape)
            q = self.q_net([const_in, hmap_in, amap_in, imap_in])
#             print('const_in.shape')
            outputs.append(q)
        q = np.concatenate(outputs, axis=0)
#         print(q.shape)
        return q

    def lr_scheduler(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr
        else:
            lr = self.learning_rate * (0.5 ** (epoch / self.lr_drop))
        return max(self.lr_min, lr)
            
    def train(self, history):
        q_inputs = []
        q_targets = []
        
        for state, action, next_state, reward, done in history:
            const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)
            q_inputs.append([const_in[0], hmap_in[0], amap_in[0], imap_in[0]])
            if done:
                q_target = reward
            else:
#                 print(np.amax(self.Q(next_state)))
                q_target = reward + self.gamma * np.amax(self.Q(next_state))
#                 print(q_target)
            q_targets.append([q_target])
            
        const_in, hmap_in, amap_in, imap_in = zip(*q_inputs)
        q_inputs = [np.asarray(const_in), np.asarray(hmap_in), np.asarray(amap_in), np.asarray(imap_in)]
        q_targets = np.asarray(q_targets)
#         print('q_targets', q_targets)
#         print([result for result in map(lambda inps: (inps.shape, np.amin(inps), np.amax(inps), np.mean(inps)), q_inputs)], q_targets)
        return self.fit(q_inputs, q_targets)
    
    def fit(self, q_inputs, q_targets):
        with tf.GradientTape() as tape:
#             print(q_inputs)
            q = self.q_net_target(q_inputs)
            # print(tf.keras.losses.MeanSquaredError()(q_targets, q))
            loss = tf.reduce_mean(tf.square(q_targets - q))
            loss = tf.keras.losses.MeanSquaredError()(q_targets, q)
#             print('q', q)
#             print('loss', loss)
            
#         print(list(zip(q, q_targets)))
        grad = tape.gradient(loss, self.q_net_target.trainable_variables)
        
#         gradient clipping
#         if self.epoch < self.warmup_epochs:
#             grad = [tf.clip_by_value(value, -1e-5, 1e-5) for value in grad]
        
        self.q_optimizer.apply_gradients(zip(grad, self.q_net_target.trainable_variables))
        
        # Fixed learning rate assignment for newer TensorFlow
        new_lr = self.lr_scheduler(self.epoch)
        try:
            # Try newer TensorFlow syntax first
            self.q_optimizer.learning_rate.assign(new_lr)
        except AttributeError:
            try:
                # Try older TensorFlow syntax
                self.q_optimizer.lr.assign(new_lr)
            except AttributeError:
                # If both fail, skip learning rate update
                pass
        
        self.epoch += 1
    
        if self.epoch % self.update_epochs == 0:
            print('update')
            self.q_net.set_weights([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        return loss
    
    def run(self, max_ep=1, verbose=False, train=None):
        if train is None:
            train = self.__train
            
        iters = (i for i, _ in enumerate(iter(bool, True))) if max_ep == -1 else range(max_ep)

        for ep in iters:
            if verbose:
                print(f'ep {ep}:')
                
            state = self.env.reset()
            ep_reward = 0
            
            history = []
            
            for step in itertools.count():
                if verbose:
                    print(f'\nstep {step}')
                    
                items, h_map, actions = state
                if len(actions) == 0: raise Exception('0 actions')
                action, r = self.select(state)
                
                if verbose:
                    print(f'possible actions: {len(actions)}')
                    print(f'action: {action}')
                    print(f'placement: {actions[action[0]][action[1]][action[2]]}')
                
                yield actions[action[0]][action[1]][action[2]]
                
                next_state, reward, done = self.env.step(action)
                
                history.append((state, action, next_state, reward, done))
                
                if self.visualize:
                    for i, packer in enumerate(self.env.packers):
                        packer.render().savefig(f'./outputs/{ep}_{step}_{i}.jpg')
                    
                ep_reward += reward
                if done:
                    break
                state = next_state
                
            loss = None
            if train:
                self.memory.extend(history)
                if len(self.memory) > 1000:
                    print('update model')
                    history = [self.memory[i] for i in np.random.choice(len(self.memory), 128)]
                    loss = self.train(history)
            
            self.ep_history.append(([packer.space_utilization() for packer in self.env.used_packers], self.env.used_bins, ep_reward))
            
            yield None
            
            utils = [round(packer.space_utilization() * 100, 2) for packer in self.env.used_packers]
            
            # Get learning rate safely for different TensorFlow versions
            current_lr = None
            if self.q_optimizer is not None:
                try:
                    current_lr = self.q_optimizer.learning_rate.numpy()
                except AttributeError:
                    try:
                        current_lr = self.q_optimizer.lr.numpy()
                    except AttributeError:
                        current_lr = "N/A"
            
            if self.verbose: 
                print(f'Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}, memory: {len(self.memory) if self.memory is not None else None}, eps: {self.eps:.2f}, loss: {loss}, lr: {current_lr}')

class HeuristicAgent:
    def __init__(self, heuristic, env=MultiBinPackerEnv(n_bins=2, max_bins=-1, size=(32, 32, 32), k=10, verbose=True), verbose=True, visualize=False):
        self.env = env
        
        self.heuristic = heuristic
        self.ep_history = []

        self.verbose = verbose
        self.visualize = visualize
    
    def select(self, state):
        # state = (items, h_map, actions)
        
        items, h_map, actions = state
#         print(len(indices(actions)))
        action = self.heuristic(actions)
            
        return action
    
    def run(self, max_ep=1, verbose=False):
        iters = (i for i, _ in enumerate(iter(bool, True))) if max_ep == -1 else range(max_ep)
        
        for ep in iters:
            if verbose:
                print(f'ep {ep}:')
                
            state = self.env.reset()
            ep_reward = 0
            
            history = []
            
            for step in itertools.count():
                if verbose:
                    print(f'\nstep {step}')
                    
                items, h_map, actions = state
                if len(actions) == 0: raise Exception('0 actions')
                
                action = self.select(state)
                
                if verbose:
                    print(f'action: {action}')
                
                next_state, reward, done = self.env.step(action)
                
                history.append((state, action, next_state, reward, done))
                
                if verbose:
                    print(f'actions: {actions}')
                    print(f'reward: {reward}, done: {done}')
                    print(f'placement: {actions[action[0]][action[1]][action[2]]}')
                
                yield actions[action[0]][action[1]][action[2]]
                
                ep_reward += reward
                
                if self.visualize:
                    for i, packer in enumerate(self.env.packers):
                        packer.render().savefig(f'./outputs/{ep}_{step}_{i}.jpg')
                        
                if done:
                    break
                
                state = next_state
            
            self.ep_history.append(([packer.space_utilization() for packer in self.env.used_packers], self.env.used_bins, ep_reward))
            
            yield None
            
            utils = [round(packer.space_utilization() * 100, 2) for packer in self.env.used_packers]
            if self.verbose: print(f'Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}')

def bottom_left(actions):
    scores = []
    for i, item in enumerate(actions):
        for j, bin_ in enumerate(item):
            for k, placement in enumerate(bin_):
                item, (x, y, z), (w, h, d), _ = placement
                y = y + h
                x = x + w
                z = z + d
                scores.append(([y, x, z, i, j, k], [i, j, k]))
        
    indices = sorted(range(len(scores)), key=lambda i: scores[i][0])
    return scores[indices[0]][1]

def best_short_side_fit(actions):
    scores = []
    for i, item in enumerate(actions):
        for j, bin_ in enumerate(item):
            for k, placement in enumerate(bin_):
                item, (x, y, z), (w, h, d), split = placement
                W, H = split.width, split.height
                scores.append(((min(W - w, H - h), i, j, k), [i, j, k]))
        
#     print(scores)
    indices = sorted(range(len(scores)), key=lambda i: scores[i][0])
    return scores[indices[0]][1]

def best_area_fit(actions):
    scores = []
    for i, item in enumerate(actions):
        for j, bin_ in enumerate(item):
            for k, placement in enumerate(bin_):
                item, (x, y, z), (w, h, d), split = placement
                W, H = split.width, split.height
                scores.append(((split.volume, min(W - w, H - h), i, j, k), [i, j, k]))
        
#     print(scores)
    indices = sorted(range(len(scores)), key=lambda i: scores[i][0])
    return scores[indices[0]][1]

def best_long_side_fit(actions):
    scores = []
    for i, item in enumerate(actions):
        for j, bin_ in enumerate(item):
            for k, placement in enumerate(bin_):
                item, (x, y, z), (w, h, d), split = placement
                W, H = split.width, split.height
                scores.append(((max(W - w, H - h), i, j, k), [i, j, k]))
        
    indices = sorted(range(len(scores)), key=lambda i: scores[i][0])
    return scores[indices[0]][1]
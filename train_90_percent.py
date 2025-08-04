#!/usr/bin/env python3
"""
ADVANCED TRAINING STRATEGY FOR 90% SPACE UTILIZATION
"""

import numpy as np
import os, time, json
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import collections
from dataclasses import dataclass
from typing import List, Tuple

from env import *
from agent import *
from deeppack3d import CUSTOM_BIN_SIZE, CUSTOM_PACKAGES

@dataclass
class TrainingConfig:
    """Advanced training configuration"""
    target_utilization: float = 90.0
    max_iterations: int = 10000
    curriculum_stages: int = 4
    ensemble_size: int = 3
    priority_replay: bool = True
    double_dqn: bool = True
    noisy_networks: bool = True

class CurriculumScheduler:
    """Curriculum learning for progressive difficulty"""
    
    def __init__(self, stages=4):
        self.stages = stages
        self.current_stage = 0
        
    def get_package_set(self, stage: int) -> List[Tuple[int, int, int]]:
        """Get packages for current curriculum stage"""
        
        if stage == 0:  # Stage 1: Easy - only small packages
            return [(24, 8, 17), (24, 16, 17), (34, 8, 24), (34, 16, 24)]
            
        elif stage == 1:  # Stage 2: Medium - small + medium  
            return CUSTOM_PACKAGES[:10]
            
        elif stage == 2:  # Stage 3: Hard - all packages
            return CUSTOM_PACKAGES
            
        else:  # Stage 4: Expert - all packages + challenging combinations
            challenging = []
            for pkg in CUSTOM_PACKAGES:
                # Add original and all rotations
                w, h, d = pkg
                challenging.extend([
                    (w, h, d), (h, w, d), (w, d, h),
                    (d, h, w), (h, d, w), (d, w, h)
                ])
            return list(set(challenging))
    
    def should_advance(self, recent_performance: List[float]) -> bool:
        """Check if ready to advance to next stage"""
        if len(recent_performance) < 50:
            return False
            
        avg_recent = np.mean(recent_performance[-50:])
        thresholds = [45.0, 60.0, 75.0, 85.0]  # % utilization thresholds
        
        return avg_recent > thresholds[self.current_stage]

class AdvancedQNetwork:
    """Enhanced Q-Network with modern techniques"""
    
    @staticmethod
    def create_dueling_network(input_shape_maps, input_shape_items, noisy=True):
        """Create Dueling DQN with noisy layers"""
        
        # Map inputs
        hmap_in = Input(input_shape_maps, name='height_map')
        amap_in = Input(input_shape_maps, name='action_map') 
        const_in = Input(input_shape_maps, name='const_map')
        imap_in = Input(input_shape_items, name='item_map')
        
        # Spatial feature extraction
        x = layers.concatenate([hmap_in, amap_in, const_in], axis=-1)
        
        # Enhanced convolutional layers
        x = layers.Conv2D(128, 5, strides=2, activation='relu', 
                         kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv2D(256, 3, strides=2, activation='relu',
                         kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv2D(512, 3, strides=2, activation='relu',
                         kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        
        # Item processing  
        imap_x = layers.Flatten()(imap_in)
        if noisy:
            item_emb = AdvancedQNetwork.noisy_dense(512, 'item_embedding')(imap_x)
        else:
            item_emb = layers.Dense(512, activation='relu', 
                                  kernel_initializer='he_uniform')(imap_x)
        
        # Combine features
        combined = layers.concatenate([x, item_emb])
        
        if noisy:
            shared = AdvancedQNetwork.noisy_dense(1024, 'shared')(combined)
        else:
            shared = layers.Dense(1024, activation='relu')(combined)
        
        shared = layers.Dropout(0.3)(shared)
        
        # Dueling architecture
        if noisy:
            value_stream = AdvancedQNetwork.noisy_dense(512, 'value_hidden')(shared)
            advantage_stream = AdvancedQNetwork.noisy_dense(512, 'advantage_hidden')(shared)
        else:
            value_stream = layers.Dense(512, activation='relu')(shared)
            advantage_stream = layers.Dense(512, activation='relu')(shared)
        
        # Output layers
        value = layers.Dense(1, name='value')(value_stream)
        advantage = layers.Dense(1, name='advantage')(advantage_stream)
        
        # Dueling combination
        q_value = layers.Add(name='q_value')([
            value,
            layers.Subtract()([advantage, layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)])
        ])
        
        model = Model([const_in, hmap_in, amap_in, imap_in], q_value, name='DuelingDQN')
        return model
    
    @staticmethod
    def noisy_dense(units, name, sigma_init=0.017):
        """Noisy network layer for better exploration"""
        def noisy_layer(inputs):
            input_dim = inputs.shape[-1]
            
            # Learnable parameters
            mu_w = tf.Variable(tf.random.uniform([input_dim, units], -1/np.sqrt(input_dim), 1/np.sqrt(input_dim)), name=f'{name}_mu_w')
            sigma_w = tf.Variable(tf.fill([input_dim, units], sigma_init), name=f'{name}_sigma_w')
            mu_b = tf.Variable(tf.zeros([units]), name=f'{name}_mu_b') 
            sigma_b = tf.Variable(tf.fill([units], sigma_init), name=f'{name}_sigma_b')
            
            # Noise
            eps_w = tf.random.normal([input_dim, units])
            eps_b = tf.random.normal([units])
            
            # Noisy weights and biases
            w = mu_w + sigma_w * eps_w
            b = mu_b + sigma_b * eps_b
            
            return tf.matmul(inputs, w) + b
        
        return layers.Lambda(noisy_layer, name=name)

class PriorityReplayBuffer:
    """Priority Experience Replay"""
    
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        
    def add(self, experience, td_error):
        """Add experience with priority"""
        priority = (abs(td_error) + 1e-6) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch with priority"""
        if len(self.buffer) < batch_size:
            return None, None, None
            
        # Calculate sampling probabilities
        probs = self.priorities[:len(self.buffer)]
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        
        batch = [self.buffer[i] for i in indices]
        
        return batch, indices, weights

class AdvancedAgent(Agent):
    """Enhanced agent with modern RL techniques"""
    
    def __init__(self, env, config: TrainingConfig):
        super().__init__(env, train=True, verbose=True, visualize=False, batch_size=32)
        
        self.config = config
        self.target_utilization = config.target_utilization
        
        # Enhanced hyperparameters
        self.gamma = 0.99  # Higher discount factor
        self.eps = 1.0 if config.noisy_networks else 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.9995
        
        # Learning parameters
        self.learning_rate = 1e-4
        self.lr_scheduler_patience = 1000
        self.update_freq = 4
        self.target_update_freq = 1000
        
        # Advanced components
        if config.priority_replay:
            self.memory = PriorityReplayBuffer(capacity=200000)
        else:
            self.memory = collections.deque(maxlen=200000)
            
        self.curriculum = CurriculumScheduler(config.curriculum_stages)
        
        # Create advanced network
        input_shape_maps = (self.map_d, self.map_w, 1)
        input_shape_items = (env.k, 3)
        
        self.q_net = AdvancedQNetwork.create_dueling_network(
            input_shape_maps, input_shape_items, config.noisy_networks
        )
        self.q_net_target = AdvancedQNetwork.create_dueling_network(
            input_shape_maps, input_shape_items, config.noisy_networks
        )
        
        # Optimizers with learning rate scheduling
        self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        
        # Performance tracking
        self.utilization_history = []
        self.best_utilization = 0
        self.stage_performance = []
        
    def enhanced_reward_function(self, state, action, next_state, done):
        """Advanced reward function targeting 90% utilization"""
        
        # Extract packer state
        items, h_maps, actions = state
        i, j, k = action
        _, (x, y, z), (w, h, d), _ = actions[i][j][k]
        
        packer = self.env.packers[j]
        
        # 1. Base utilization reward (primary objective)
        utilization = packer.space_utilization()
        util_reward = utilization ** 2  # Quadratic reward for high utilization
        
        # 2. Progressive utilization bonus
        if utilization > 0.8:
            util_reward += (utilization - 0.8) * 10  # Huge bonus for >80%
        if utilization > 0.85:
            util_reward += (utilization - 0.85) * 20  # Even bigger bonus for >85%
        if utilization > 0.9:
            util_reward += (utilization - 0.9) * 50   # Massive bonus for >90%
        
        # 3. Compactness reward (minimize wasted height)
        h_map = h_maps[j]
        if np.max(h_map) > 0:
            volume_used = np.sum([split.volume for split in packer.splits])
            max_height = np.max(h_map)
            base_area = packer.size[0] * packer.size[2]
            height_efficiency = volume_used / (max_height * base_area)
            util_reward += height_efficiency * 2
        
        # 4. Gap penalty (avoid creating unusable spaces)
        gap_penalty = self.calculate_gap_penalty(h_map, (x, y, z), (w, h, d))
        util_reward -= gap_penalty * 0.5
        
        # 5. Strategic placement bonuses
        edge_bonus = self.calculate_edge_bonus((x, y, z), packer.size)
        util_reward += edge_bonus * 0.3
        
        # 6. Curriculum-based reward scaling
        stage_multiplier = 1.0 + (self.curriculum.current_stage * 0.2)
        util_reward *= stage_multiplier
        
        # 7. End-of-episode bonus for high utilization
        if done and utilization > 0.85:
            util_reward += 20 * (utilization - 0.85)
        
        return util_reward
    
    def train_to_90_percent(self):
        """Main training loop targeting 90% utilization"""
        
        print(f"🎯 TRAINING TO {self.target_utilization}% UTILIZATION")
        print("=" * 60)
        print(f"Strategy: Curriculum Learning + Advanced DQN")
        print(f"Max iterations: {self.config.max_iterations}")
        print(f"Curriculum stages: {self.config.curriculum_stages}")
        print("=" * 60)
        
        start_time = time.time()
        
        for iteration in range(self.config.max_iterations):
            
            # Curriculum learning
            current_packages = self.curriculum.get_package_set(self.curriculum.current_stage)
            
            # Create training environment for current stage
            training_items = current_packages * (100 // len(current_packages) + 5)
            np.random.shuffle(training_items)
            self.env.conveyor = Conveyor(k=1, assigned_items=training_items)
            
            # Run episode
            episode_utilization = self.run_training_episode()
            
            if episode_utilization is not None:
                self.utilization_history.append(episode_utilization)
                self.stage_performance.append(episode_utilization)
                
                # Update best performance
                if episode_utilization > self.best_utilization:
                    self.best_utilization = episode_utilization
                    self.save_best_model()
                    
                    if episode_utilization >= self.target_utilization:
                        print(f"🎉 TARGET ACHIEVED! {episode_utilization:.1f}% utilization!")
                        break
            
            # Curriculum advancement
            if len(self.stage_performance) >= 100:
                if self.curriculum.should_advance(self.stage_performance):
                    self.curriculum.current_stage = min(
                        self.curriculum.current_stage + 1, 
                        self.curriculum.stages - 1
                    )
                    print(f"📈 Advanced to curriculum stage {self.curriculum.current_stage + 1}")
                    self.stage_performance = []
            
            # Progress reporting
            if iteration % 100 == 0 and iteration > 0:
                self.report_progress(iteration, start_time)
            
            # Learning rate decay
            if iteration % 1000 == 0 and iteration > 0:
                self.decay_learning_rate()
        
        # Final results
        self.print_final_results(time.time() - start_time)
    
    def run_training_episode(self):
        """Run single training episode with enhanced learning"""
        
        state = self.env.reset()
        episode_experiences = []
        episode_utilization = None
        
        for step in range(1000):  # Max steps per episode
            # Select action
            if self.config.noisy_networks:
                action, _ = self.select(state)  # Noisy networks handle exploration
            else:
                action, _ = self.epsilon_greedy_select(state)
            
            # Execute action
            next_state, reward, done = self.env.step(action)
            
            # Enhanced reward
            enhanced_reward = self.enhanced_reward_function(state, action, next_state, done)
            
            # Store experience
            experience = (state, action, next_state, enhanced_reward, done)
            episode_experiences.append(experience)
            
            if done:
                episode_utilization = self.env.packers[0].space_utilization() * 100
                break
                
            state = next_state
        
        # Add experiences to replay buffer
        for experience in episode_experiences:
            if isinstance(self.memory, PriorityReplayBuffer):
                # Calculate TD error for priority
                td_error = self.calculate_td_error(experience)
                self.memory.add(experience, td_error)
            else:
                self.memory.append(experience)
        
        # Training step
        if len(self.memory) > 1000:
            self.advanced_training_step()
        
        return episode_utilization
    
    def save_best_model(self):
        """Save best performing model"""
        model_name = f'advanced_k1_best_{self.best_utilization:.1f}pct.h5'
        self.q_net.save(model_name)
        
        # Also update standard location
        self.q_net.save('./models/k=1.h5')
        
        print(f"💾 Saved new best model: {model_name}")

def main():
    """Run advanced training to reach 90% utilization"""
    
    # Configuration
    config = TrainingConfig(
        target_utilization=90.0,
        max_iterations=15000,
        curriculum_stages=4,
        priority_replay=True,
        double_dqn=True,
        noisy_networks=True
    )
    
    # Create environment
    env = MultiBinPackerEnv(
        n_bins=1,
        max_bins=1,
        size=CUSTOM_BIN_SIZE,
        k=1,
        verbose=False
    )
    
    # Create advanced agent
    agent = AdvancedAgent(env, config)
    
    # Start training
    agent.train_to_90_percent()

if __name__ == "__main__":
    main()
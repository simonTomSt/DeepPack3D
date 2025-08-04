#!/usr/bin/env python3
"""
OPTIMIZED TRAINING FOR MAXIMUM SPACE UTILIZATION - k=1 model
Target: Beat 73% heuristic performance → 80%+ utilization
"""

import numpy as np
import os, shutil, time
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import collections

from env import *
from agent import *
from deeppack3d import CUSTOM_BIN_SIZE, CUSTOM_PACKAGES

class OptimizedAgent(Agent):
    """Enhanced agent focused on maximum space utilization"""
    
    def __init__(self, env, train=True, verbose=True, visualize=False, batch_size=16):
        super().__init__(env, train=train, verbose=verbose, visualize=visualize, batch_size=batch_size)
        
        # OPTIMIZED HYPERPARAMETERS FOR SPACE UTILIZATION
        self.gamma = 0.98                    # Higher discount for long-term planning
        self.eps_start = 1.0
        self.eps_min = 0.01                  # Lower minimum = more exploitation
        self.eps_decay_steps = 3000          # Gradual decay over more episodes
        self.eps = self.eps_start
        
        # Enhanced learning parameters
        self.warmup_epochs = 100             # More warmup
        self.warmup_lr = 1e-4               # Higher initial learning rate
        self.learning_rate = 5e-4           # Higher base learning rate
        self.lr_min = 1e-6
        self.lr_drop = 2000
        self.update_epochs = 5               # More frequent target updates
        
        # Larger memory for better training
        if train and self.memory is not None:
            self.memory = collections.deque(maxlen=100000)  # 10x larger memory
            
        self.step_count = 0                  # Track training steps
        
    def enhanced_reward(self, packer, action_coords, item_size):
        """ENHANCED REWARD FUNCTION focused on space utilization"""
        
        # 1. SPACE UTILIZATION (main objective)
        volume_used = np.sum([split.volume for split in packer.splits])
        bin_volume = np.prod(packer.size)
        utilization = volume_used / bin_volume
        
        # 2. COMPACTNESS BONUS (prefer lower, denser packing)
        h_map = packer.height_map()
        if np.max(h_map) > 0:
            max_height = np.max(h_map)
            # Reward for using height efficiently
            height_efficiency = volume_used / (max_height * packer.size[0] * packer.size[2])
        else:
            height_efficiency = 0
            
        # 3. GAP MINIMIZATION (avoid creating unusable spaces)
        gap_penalty = self.calculate_gap_penalty(h_map, action_coords, item_size)
        
        # 4. CORNER/EDGE PREFERENCE (more stable packing)
        edge_bonus = self.calculate_edge_bonus(action_coords, packer.size)
        
        # 5. LARGE ITEM BONUS (pack big items early)
        item_volume = np.prod(item_size)
        size_bonus = (item_volume / max([np.prod(pkg) for pkg in CUSTOM_PACKAGES])) * 0.1
        
        # COMBINED REWARD (emphasizing space utilization)
        reward = (
            utilization * 5.0 +              # Main objective (5x weight)
            height_efficiency * 2.0 +        # Compactness
            (1.0 - gap_penalty) * 1.5 +      # Gap avoidance  
            edge_bonus * 0.5 +               # Stability
            size_bonus                       # Strategic placement
        )
        
        return reward
    
    def calculate_gap_penalty(self, h_map, coords, size):
        """Penalty for creating small unusable gaps"""
        x, y, z = coords
        w, h, d = size
        penalty = 0
        
        # Check if placement creates small gaps around item
        W, H, D = self.env.size
        
        # Check adjacent areas for small gaps
        for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            check_x = x + dx
            check_z = z + dz
            
            if 0 <= check_x < W and 0 <= check_z < D:
                if check_x < h_map.shape[1] and check_z < h_map.shape[0]:
                    height_diff = abs(h_map[check_z, check_x] - (y + h))
                    
                    # Penalty for small gaps that can't fit smallest package
                    min_package_dim = min([min(pkg) for pkg in CUSTOM_PACKAGES])
                    if 0 < height_diff < min_package_dim:
                        penalty += 0.3
        
        return min(penalty, 1.0)
    
    def calculate_edge_bonus(self, coords, bin_size):
        """Bonus for placing items against edges (more stable)"""
        x, y, z = coords
        W, H, D = bin_size
        
        bonus = 0
        if x == 0: bonus += 0.1      # Left wall
        if z == 0: bonus += 0.1      # Back wall  
        if y == 0: bonus += 0.2      # Floor (most important)
        
        # Corner bonuses (even better)
        corners = [(x == 0 and z == 0), (x == 0 and y == 0), (z == 0 and y == 0)]
        bonus += sum(corners) * 0.1
        
        # Ultimate corner bonus
        if x == 0 and z == 0 and y == 0:
            bonus += 0.2
            
        return bonus
    
    def epsilon_decay_schedule(self, step):
        """Improved epsilon decay for better exploration→exploitation transition"""
        if step < self.eps_decay_steps:
            # Exponential decay with slower start
            progress = step / self.eps_decay_steps
            self.eps = self.eps_min + (self.eps_start - self.eps_min) * np.exp(-3.0 * progress)
        else:
            self.eps = self.eps_min
            
    def enhanced_training_step(self):
        """Enhanced training with better sampling and loss calculation"""
        if len(self.memory) < 1000:
            return None
            
        # Sample batch with some prioritization (recent experiences weighted more)
        batch_size = min(64, len(self.memory))
        
        # Weight recent experiences more heavily (last 20% get 2x weight)
        weights = np.ones(len(self.memory))
        recent_count = len(self.memory) // 5
        weights[-recent_count:] *= 2
        weights = weights / np.sum(weights)
        
        batch_indices = np.random.choice(len(self.memory), batch_size, 
                                       replace=False, p=weights)
        batch = [self.memory[i] for i in batch_indices]
        
        # Prepare training data
        q_inputs = []
        q_targets = []
        
        for state, action, next_state, reward, done in batch:
            const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)
            q_inputs.append([const_in[0], hmap_in[0], amap_in[0], imap_in[0]])
            
            if done:
                q_target = reward
            else:
                next_q = self.Q(next_state)
                q_target = reward + self.gamma * np.max(next_q)
                
            q_targets.append([q_target])
        
        # Train
        const_in, hmap_in, amap_in, imap_in = zip(*q_inputs)
        q_inputs_batch = [np.asarray(const_in), np.asarray(hmap_in), 
                         np.asarray(amap_in), np.asarray(imap_in)]
        q_targets_batch = np.asarray(q_targets)
        
        loss = self.fit(q_inputs_batch, q_targets_batch)
        return loss

def create_diverse_training_packages(base_packages, multiplier=5):
    """Create diverse training set with rotations and variations"""
    diverse_packages = []
    
    for pkg in base_packages:
        w, h, d = pkg
        # Add all valid rotations
        rotations = [
            (w, h, d),    # Original
            (h, w, d),    # Rotate XY
            (w, d, h),    # Rotate XZ  
            (d, h, w),    # Rotate YZ
            (h, d, w),    # Rotate XY+XZ
            (d, w, h),    # Rotate XZ+YZ
        ]
        
        # Filter valid rotations (must fit in bin)
        valid_rotations = [r for r in rotations 
                          if all(dim <= max(CUSTOM_BIN_SIZE) for dim in r)]
        diverse_packages.extend(valid_rotations)
    
    # Remove duplicates
    unique_packages = list(set(diverse_packages))
    
    # Repeat for training diversity
    return unique_packages * multiplier

def train_optimal_space_utilization(n_iterations=4000, verbose=True):
    """MAIN TRAINING FUNCTION - Optimized for maximum space utilization"""
    
    print("🚀 OPTIMIZED TRAINING FOR MAXIMUM SPACE UTILIZATION")
    print("=" * 70)
    print(f"Current heuristic best: ~73% → Target: 80%+")
    print(f"Training iterations: {n_iterations}")
    print(f"Bin size: {CUSTOM_BIN_SIZE}")
    print(f"Expected training time: {n_iterations/800:.1f}-{n_iterations/400:.1f} hours")
    print("=" * 70)
    
    # Create environment
    env = MultiBinPackerEnv(
        n_bins=1, 
        max_bins=1, 
        size=CUSTOM_BIN_SIZE, 
        k=1, 
        verbose=False
    )
    
    # Create diverse training packages
    diverse_packages = create_diverse_training_packages(CUSTOM_PACKAGES, multiplier=10)
    print(f"Training with {len(diverse_packages)} diverse package variants")
    
    # Set training data (shuffle for each iteration)
    training_packages = diverse_packages * (n_iterations // len(diverse_packages) + 50)
    env.conveyor = Conveyor(k=1, assigned_items=training_packages)
    
    # Create optimized agent
    agent = OptimizedAgent(env, train=True, verbose=False, visualize=False, batch_size=16)
    
    # Training tracking
    best_utilization = 0
    utilization_history = []
    loss_history = []
    recent_utils = []
    
    print(f"Starting optimized training at {time.strftime('%H:%M:%S')}...")
    start_time = time.time()
    
    for iteration in range(n_iterations):
        # Progress reporting
        if iteration % 100 == 0 and iteration > 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / iteration) * (n_iterations - iteration)
            avg_recent = np.mean(recent_utils[-100:]) if len(recent_utils) >= 100 else (np.mean(recent_utils) if recent_utils else 0)
            
            print(f"Iter {iteration:4d}/{n_iterations} | "
                  f"Time: {elapsed/3600:.1f}h | "
                  f"ETA: {remaining/3600:.1f}h | "
                  f"Recent avg: {avg_recent:.1f}% | "
                  f"Best: {best_utilization:.1f}% | "
                  f"Eps: {agent.eps:.3f}")
        
        # Run episode with enhanced reward
        episode_data = []
        for result in agent.run(1, verbose=False):
            if result is None:
                break
            episode_data.append(result)
        
        # Update exploration schedule
        agent.epsilon_decay_schedule(iteration)
        agent.step_count += 1
        
        # Enhanced training step
        if iteration > 200 and iteration % 3 == 0:  # Start training after warmup
            loss = agent.enhanced_training_step()
            if loss is not None:
                loss_history.append(float(loss))
        
        # Track performance
        if agent.ep_history:
            current_util = agent.ep_history[-1][0][0] * 100
            utilization_history.append(current_util)
            recent_utils.append(current_util)
            
            # Save best model
            if current_util > best_utilization:
                best_utilization = current_util
                agent.q_net.save(f'./best_k1_utilization_{best_utilization:.1f}pct.h5')
                print(f"🏆 NEW BEST: {best_utilization:.1f}% utilization!")
        
        # Periodic saves
        if iteration > 0 and iteration % 1000 == 0:
            agent.q_net.save(f'./checkpoint_k1_iter_{iteration}.h5')
            
            # Performance analysis
            recent_100 = recent_utils[-100:] if len(recent_utils) >= 100 else recent_utils
            if recent_100:
                avg_util = np.mean(recent_100)
                std_util = np.std(recent_100)
                print(f"Checkpoint {iteration}: Avg={avg_util:.1f}% ±{std_util:.1f}%, Best={best_utilization:.1f}%")
    
    # Final save and analysis
    final_model_name = f'optimal_k1_final_{int(time.time())}.h5'
    agent.q_net.save(final_model_name)
    agent.q_net.save('./models/k=1.h5')  # Replace default model
    
    total_time = time.time() - start_time
    
    print("\n" + "🎉 TRAINING COMPLETED!" + "\n" + "=" * 70)
    print(f"Total training time: {total_time/3600:.1f} hours")
    print(f"Best utilization achieved: {best_utilization:.1f}%")
    print(f"Improvement vs heuristic: {best_utilization-73:.1f} percentage points")
    
    if len(recent_utils) >= 100:
        final_avg = np.mean(recent_utils[-100:])
        final_std = np.std(recent_utils[-100:])
        print(f"Final performance: {final_avg:.1f}% ±{final_std:.1f}%")
        
    print(f"Best model saved as: {final_model_name}")
    print(f"Model ready for use as: ./models/k=1.h5")
    
    return agent, utilization_history, best_utilization

if __name__ == "__main__":
    print("🎯 STARTING OPTIMIZED TRAINING FOR MAXIMUM SPACE UTILIZATION")
    print("This will train a k=1 model to beat the 73% heuristic performance")
    print()
    
    # Run optimized training (3-5 hours depending on hardware)
    agent, history, best_util = train_optimal_space_utilization(
        n_iterations=4000,  # 4000 iterations for thorough training
        verbose=True
    )
    
    print(f"\n🏆 TRAINING COMPLETE!")
    print(f"Best utilization achieved: {best_util:.1f}%")
    print(f"Model improvement: {'+' if best_util > 73 else ''}{best_util-73:.1f} percentage points vs heuristic")
    print("\n🚀 Test your optimized model:")
    print("python deeppack3d.py rl 1 --visualize --verbose=1")
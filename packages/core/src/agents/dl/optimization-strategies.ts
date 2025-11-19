/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview Collection of optimization strategies for deep learning training.
 * Includes learning rate schedules, advanced optimizers, and gradient handling techniques.
 */

import { z } from 'zod';

/**
 * Schema for optimizer configuration.
 */
const OptimizerConfigSchema = z.object({
  name: z.string(),
  description: z.string(),
  hyperparameters: z.record(z.any()),
  implementation: z.string(),
  useCases: z.array(z.string()),
  advantages: z.array(z.string()),
  disadvantages: z.array(z.string()),
  references: z.array(z.string()),
});

export type OptimizerConfig = z.infer<typeof OptimizerConfigSchema>;

/**
 * Schema for learning rate schedule.
 */
const LRScheduleConfigSchema = z.object({
  name: z.string(),
  description: z.string(),
  formula: z.string(),
  hyperparameters: z.record(z.any()),
  implementation: z.string(),
  useCases: z.array(z.string()),
  tips: z.array(z.string()),
});

export type LRScheduleConfig = z.infer<typeof LRScheduleConfigSchema>;

// =============================================================================
// WARMUP STRATEGIES
// =============================================================================

export const WarmupStrategies: Record<string, LRScheduleConfig> = {
  /**
   * Linear warmup - most commonly used
   */
  linear_warmup: {
    name: 'Linear Warmup',
    description: 'Gradually increases learning rate from 0 to peak over warmup steps',
    formula: 'lr = peak_lr * (step / warmup_steps)',
    hyperparameters: {
      warmup_steps: 'Number of warmup steps (typically 1-10% of total)',
      peak_lr: 'Target learning rate after warmup',
    },
    implementation: `
def get_linear_warmup_lr(step: int, warmup_steps: int, peak_lr: float) -> float:
    """Linear warmup schedule."""
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    return peak_lr

# PyTorch implementation
class LinearWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps
                    for base_lr in self.base_lrs]
        return self.base_lrs

# Usage
scheduler = LinearWarmup(optimizer, warmup_steps=2000)
`,
    useCases: [
      'Transformer training (BERT, GPT, ViT)',
      'Large batch training',
      'Preventing initial training instability',
      'Fine-tuning pretrained models',
    ],
    tips: [
      'Use 1-10% of total training steps for warmup',
      'Longer warmup for larger batch sizes',
      'Shorter warmup for smaller models',
      'Can combine with any decay schedule',
    ],
  },

  /**
   * Exponential warmup
   */
  exponential_warmup: {
    name: 'Exponential Warmup',
    description: 'Exponentially increases learning rate during warmup phase',
    formula: 'lr = peak_lr * (gamma ^ (warmup_steps - step))',
    hyperparameters: {
      warmup_steps: 'Number of warmup steps',
      peak_lr: 'Target learning rate',
      gamma: 'Base of exponential (typically 0.9-0.99)',
    },
    implementation: `
def get_exponential_warmup_lr(step: int, warmup_steps: int,
                               peak_lr: float, gamma: float = 0.99) -> float:
    """Exponential warmup - slower initial increase, faster near end."""
    if step < warmup_steps:
        return peak_lr * (1 - gamma ** (step + 1)) / (1 - gamma ** warmup_steps)
    return peak_lr

# Alternative formulation
def get_exp_warmup_v2(step: int, warmup_steps: int,
                      peak_lr: float, start_lr: float = 1e-8) -> float:
    """Exponential warmup from start_lr to peak_lr."""
    if step < warmup_steps:
        return start_lr * (peak_lr / start_lr) ** (step / warmup_steps)
    return peak_lr
`,
    useCases: [
      'Very large learning rates',
      'Sensitive optimization landscapes',
      'When linear warmup causes instability',
    ],
    tips: [
      'More gradual start than linear warmup',
      'Good for very sensitive models',
      'May need longer warmup period',
    ],
  },

  /**
   * Gradual warmup (square root)
   */
  sqrt_warmup: {
    name: 'Square Root Warmup',
    description: 'Learning rate increases proportionally to square root of step',
    formula: 'lr = peak_lr * sqrt(step / warmup_steps)',
    hyperparameters: {
      warmup_steps: 'Number of warmup steps',
      peak_lr: 'Target learning rate',
    },
    implementation: `
import math

def get_sqrt_warmup_lr(step: int, warmup_steps: int, peak_lr: float) -> float:
    """Square root warmup - aggressive start, gradual finish."""
    if step < warmup_steps:
        return peak_lr * math.sqrt(step / warmup_steps)
    return peak_lr
`,
    useCases: [
      'When you want faster initial warmup',
      'Short training runs',
      'Models that converge quickly',
    ],
    tips: [
      'Faster than linear in early steps',
      'Use when early steps are wasted with linear warmup',
      'Good for time-constrained training',
    ],
  },

  /**
   * RAdam-style warmup (variance-based)
   */
  radam_warmup: {
    name: 'RAdam Variance Warmup',
    description: 'Automatic warmup based on Adam variance rectification',
    formula: 'Derived from adaptive learning rate variance correction',
    hyperparameters: {
      beta2: 'Adam beta2 parameter (used to compute warmup length)',
    },
    implementation: `
import math

class RAdam(torch.optim.Optimizer):
    """Rectified Adam with automatic warmup."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decayed gradients
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Maximum length of approximated SMA
                rho_inf = 2 / (1 - beta2) - 1
                # Current length of approximated SMA
                rho_t = rho_inf - 2 * state['step'] * (beta2 ** state['step']) / bias_correction2

                # Variance rectification
                if rho_t > 5:
                    # Variance is tractable
                    rect = math.sqrt(
                        (rho_t - 4) * (rho_t - 2) * rho_inf /
                        ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    )
                    adaptive_lr = rect * math.sqrt(bias_correction2) / exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcmul_(exp_avg, adaptive_lr, value=-group['lr'] / bias_correction1)
                else:
                    # Variance is intractable, use unadapted lr
                    p.data.add_(exp_avg, alpha=-group['lr'] / bias_correction1)
`,
    useCases: [
      'When you want automatic warmup without tuning',
      'Adam-based training',
      'Simplifying hyperparameter search',
    ],
    tips: [
      'No warmup_steps hyperparameter needed',
      'Warmup length derived from beta2',
      'Typically ~5 steps before variance is tractable',
      'Drop-in replacement for Adam',
    ],
  },
};

// =============================================================================
// LEARNING RATE DECAY STRATEGIES
// =============================================================================

export const LearningRateSchedules: Record<string, LRScheduleConfig> = {
  /**
   * Cosine annealing
   */
  cosine_annealing: {
    name: 'Cosine Annealing',
    description: 'Decays learning rate following cosine curve',
    formula: 'lr = min_lr + (max_lr - min_lr) * (1 + cos(pi * t / T)) / 2',
    hyperparameters: {
      T_max: 'Total number of steps/epochs',
      eta_min: 'Minimum learning rate (default 0)',
    },
    implementation: `
import math
import torch

class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]

# With warmup
def get_cosine_with_warmup(step, warmup_steps, total_steps, peak_lr, min_lr=0):
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
`,
    useCases: [
      'Vision Transformers',
      'Language model pretraining',
      'Any long training run',
      'Default choice for modern deep learning',
    ],
    tips: [
      'Set eta_min to 10% of peak_lr for stability',
      'Works well with warmup',
      'Smooth decay helps find better minima',
      'Most widely used schedule',
    ],
  },

  /**
   * Cosine annealing with warm restarts (SGDR)
   */
  cosine_warm_restarts: {
    name: 'Cosine Annealing with Warm Restarts (SGDR)',
    description: 'Periodic restarts of learning rate following cosine curve',
    formula: 'lr = min_lr + (max_lr - min_lr) * (1 + cos(pi * T_cur / T_i)) / 2',
    hyperparameters: {
      T_0: 'Initial cycle length',
      T_mult: 'Cycle length multiplier (1 = fixed, 2 = doubling)',
      eta_min: 'Minimum learning rate',
    },
    implementation: `
import math
import torch

class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            # Find which cycle we're in
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.T_i = self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# Usage: Doubling cycles starting at 10 epochs
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
`,
    useCases: [
      'Snapshot ensembles',
      'Exploring multiple local minima',
      'Long training with periodic exploration',
      'When single decay is insufficient',
    ],
    tips: [
      'Save checkpoints at restart points for ensembles',
      'T_mult=2 doubles cycle length each restart',
      'T_mult=1 for fixed-length cycles',
      'Can escape local minima',
    ],
  },

  /**
   * One Cycle Policy
   */
  one_cycle: {
    name: 'One Cycle Policy (Super-Convergence)',
    description: 'Warmup to peak, then cosine decay with final annealing phase',
    formula: 'Warmup + Cosine decay + Final annealing',
    hyperparameters: {
      max_lr: 'Peak learning rate',
      total_steps: 'Total training steps',
      pct_start: 'Percentage of cycle spent increasing LR (default 0.3)',
      anneal_strategy: 'Annealing type: cos or linear',
      div_factor: 'Initial LR = max_lr / div_factor',
      final_div_factor: 'Final LR = initial_lr / final_div_factor',
    },
    implementation: `
import torch

# PyTorch built-in
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    total_steps=10000,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25,        # Initial LR = max_lr/25
    final_div_factor=1e4  # Final LR = initial_lr/1e4
)

# Custom implementation with momentum
class OneCycleLR:
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3,
                 div_factor=25, final_div_factor=1e4,
                 max_momentum=0.95, min_momentum=0.85):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.step_num = 0
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum

    def step(self):
        self.step_num += 1
        pct = self.step_num / self.total_steps

        if pct <= self.pct_start:
            # Warmup phase
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * pct / self.pct_start
            momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * pct / self.pct_start
        else:
            # Annealing phase
            pct_decay = (pct - self.pct_start) / (1 - self.pct_start)
            lr = self.final_lr + (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * pct_decay)) / 2
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * pct_decay

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if 'momentum' in param_group:
                param_group['momentum'] = momentum
            elif 'betas' in param_group:
                param_group['betas'] = (momentum, param_group['betas'][1])
`,
    useCases: [
      'Fast training with SGD',
      'Image classification',
      'When you can only train for limited epochs',
      'Transfer learning',
    ],
    tips: [
      'Use LR range test to find max_lr',
      'Can train in 1/5 to 1/10 the epochs',
      'Momentum annealing is important',
      'Works best with SGD, okay with Adam',
      'pct_start=0.3 is a good default',
    ],
  },

  /**
   * Polynomial decay
   */
  polynomial_decay: {
    name: 'Polynomial Decay',
    description: 'Decays learning rate following polynomial curve',
    formula: 'lr = (initial_lr - end_lr) * (1 - step/total_steps)^power + end_lr',
    hyperparameters: {
      power: 'Polynomial power (1=linear, 2=quadratic)',
      end_lr: 'Final learning rate',
    },
    implementation: `
def polynomial_decay(step, total_steps, initial_lr, end_lr=0, power=1.0):
    """Polynomial decay schedule."""
    if step >= total_steps:
        return end_lr
    return (initial_lr - end_lr) * (1 - step / total_steps) ** power + end_lr

class PolynomialDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, end_lr=0, power=1.0, last_epoch=-1):
        self.total_steps = total_steps
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.total_steps:
            return [self.end_lr for _ in self.base_lrs]
        return [
            (base_lr - self.end_lr) * (1 - self.last_epoch / self.total_steps) ** self.power + self.end_lr
            for base_lr in self.base_lrs
        ]
`,
    useCases: [
      'BERT and other transformers (power=1)',
      'Object detection (power=0.9)',
      'When you want controllable decay speed',
    ],
    tips: [
      'power=1 is linear decay',
      'power>1 for faster initial decay',
      'power<1 for slower initial decay',
      'Commonly used with warmup',
    ],
  },

  /**
   * Inverse square root (Transformer schedule)
   */
  inverse_sqrt: {
    name: 'Inverse Square Root Schedule',
    description: 'Learning rate decays as inverse square root of step (original Transformer)',
    formula: 'lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))',
    hyperparameters: {
      warmup_steps: 'Number of warmup steps',
      d_model: 'Model dimension for scaling (optional)',
    },
    implementation: `
import math

def get_inverse_sqrt_lr(step, warmup_steps, peak_lr):
    """Inverse square root schedule with linear warmup."""
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    return peak_lr * math.sqrt(warmup_steps / step)

# Original Transformer formulation
def transformer_lr(step, d_model, warmup_steps):
    """Original Attention Is All You Need schedule."""
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

class InverseSquareRootLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        return [base_lr * math.sqrt(self.warmup_steps / step) for base_lr in self.base_lrs]
`,
    useCases: [
      'Original Transformer training',
      'Machine translation',
      'When you don\'t know total training steps',
    ],
    tips: [
      'Doesn\'t require knowing total steps',
      'Infinite training friendly',
      'Slower decay than cosine',
      'May undertrain with limited steps',
    ],
  },

  /**
   * REX (Revised Exponential) schedule
   */
  rex_schedule: {
    name: 'REX (Revised Exponential) Schedule',
    description: 'Improved exponential decay with theoretical guarantees',
    formula: 'lr = initial_lr * exp(-gamma * step) with revised gamma',
    hyperparameters: {
      total_steps: 'Total training steps',
      min_lr_ratio: 'Ratio of final to initial LR',
    },
    implementation: `
import math

def rex_schedule(step, total_steps, initial_lr, min_lr_ratio=0.01):
    """REX schedule - theoretically motivated exponential decay."""
    # Compute gamma to reach min_lr_ratio at total_steps
    gamma = -math.log(min_lr_ratio) / total_steps
    return initial_lr * math.exp(-gamma * step)

class REXScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, min_lr_ratio=0.01, last_epoch=-1):
        self.total_steps = total_steps
        self.gamma = -math.log(min_lr_ratio) / total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * math.exp(-self.gamma * self.last_epoch)
                for base_lr in self.base_lrs]

# With warmup
def rex_with_warmup(step, warmup_steps, total_steps, peak_lr, min_lr_ratio=0.01):
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    decay_step = step - warmup_steps
    decay_total = total_steps - warmup_steps
    gamma = -math.log(min_lr_ratio) / decay_total
    return peak_lr * math.exp(-gamma * decay_step)
`,
    useCases: [
      'When exponential decay is preferred',
      'Theoretical optimization research',
      'Alternative to cosine for comparison',
    ],
    tips: [
      'Theoretically motivated unlike ad-hoc schedules',
      'Decays faster initially than cosine',
      'Good for convergence analysis',
    ],
  },
};

// =============================================================================
// ADVANCED OPTIMIZERS
// =============================================================================

export const AdvancedOptimizers: Record<string, OptimizerConfig> = {
  /**
   * LARS (Layer-wise Adaptive Rate Scaling)
   */
  lars: {
    name: 'LARS (Layer-wise Adaptive Rate Scaling)',
    description: 'Enables large batch training by scaling LR per layer based on weight/gradient ratio',
    hyperparameters: {
      lr: 'Base learning rate',
      momentum: 'SGD momentum',
      weight_decay: 'Weight decay',
      trust_coefficient: 'Trust ratio coefficient (default 0.001)',
      epsilon: 'Small constant for numerical stability',
    },
    implementation: `
import torch

class LARS(torch.optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling for large batch training."""

    def __init__(self, params, lr=1.0, momentum=0.9, weight_decay=0,
                 trust_coefficient=0.001, epsilon=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       trust_coefficient=trust_coefficient, epsilon=epsilon)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Compute trust ratio
                weight_norm = p.norm(2)
                grad_norm = grad.norm(2)

                if weight_norm > 0 and grad_norm > 0:
                    trust_ratio = group['trust_coefficient'] * weight_norm / (
                        grad_norm + group['weight_decay'] * weight_norm + group['epsilon']
                    )
                else:
                    trust_ratio = 1.0

                # Apply trust ratio to learning rate
                scaled_lr = group['lr'] * trust_ratio

                # Momentum
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(grad)
                    grad = buf

                p.add_(grad, alpha=-scaled_lr)

        return loss

# Usage for large batch ImageNet training
optimizer = LARS(model.parameters(), lr=0.1 * batch_size / 256,
                 momentum=0.9, weight_decay=1e-4)
`,
    useCases: [
      'Large batch training (4K-32K)',
      'ImageNet with ResNet',
      'When scaling SGD fails',
    ],
    advantages: [
      'Enables very large batch sizes',
      'Per-layer learning rate adaptation',
      'Based on theoretical analysis',
    ],
    disadvantages: [
      'Adds computational overhead',
      'Requires tuning trust coefficient',
      'May not help for small batches',
    ],
    references: [
      'Large Batch Training of Convolutional Networks (You et al., 2017)',
    ],
  },

  /**
   * LAMB (Layer-wise Adaptive Moments for Batch training)
   */
  lamb: {
    name: 'LAMB (Layer-wise Adaptive Moments)',
    description: 'LARS principle applied to Adam for large batch training',
    hyperparameters: {
      lr: 'Base learning rate',
      betas: 'Adam betas (momentum, variance)',
      weight_decay: 'Weight decay',
      trust_ratio_clip: 'Maximum trust ratio (default 10)',
      epsilon: 'Adam epsilon',
    },
    implementation: `
import torch
import math

class LAMB(torch.optim.Optimizer):
    """LAMB optimizer for large batch training with Adam."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, trust_ratio_clip=10.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       trust_ratio_clip=trust_ratio_clip)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute Adam update
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2

                adam_step = exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + group['eps'])

                # Add weight decay
                if group['weight_decay'] != 0:
                    adam_step.add_(p, alpha=group['weight_decay'])

                # Compute trust ratio
                weight_norm = p.norm(2).clamp(min=1e-8)
                adam_norm = adam_step.norm(2).clamp(min=1e-8)
                trust_ratio = weight_norm / adam_norm
                trust_ratio = torch.clamp(trust_ratio, max=group['trust_ratio_clip'])

                # Apply update
                p.add_(adam_step, alpha=-group['lr'] * trust_ratio)

        return loss

# Usage for BERT large batch training
optimizer = LAMB(model.parameters(), lr=0.00176, betas=(0.9, 0.999),
                 weight_decay=0.01)
`,
    useCases: [
      'BERT/Transformer pretraining',
      'Large batch training (up to 64K)',
      'When AdamW fails to scale',
    ],
    advantages: [
      'Enables massive batch sizes',
      'Maintains Adam\'s adaptive learning',
      'Layer-wise trust ratio',
    ],
    disadvantages: [
      'More hyperparameters',
      'Computational overhead',
      'Requires careful tuning',
    ],
    references: [
      'Large Batch Optimization for Deep Learning (You et al., 2019)',
    ],
  },

  /**
   * SAM (Sharpness-Aware Minimization)
   */
  sam: {
    name: 'SAM (Sharpness-Aware Minimization)',
    description: 'Seeks flat minima by minimizing both loss and sharpness',
    hyperparameters: {
      base_optimizer: 'Base optimizer (SGD, Adam, etc.)',
      rho: 'Neighborhood size for sharpness (default 0.05)',
      adaptive: 'Whether to use adaptive SAM',
    },
    implementation: `
import torch

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer."""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Compute gradient at w + epsilon (adversarial point)."""
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None:
                    continue

                # Store original parameters
                self.state[p]['old_p'] = p.data.clone()

                # Compute epsilon
                if group['adaptive']:
                    e_w = (torch.pow(p, 2) * p.grad * scale).to(p)
                else:
                    e_w = p.grad * scale

                # Move to adversarial point
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Update weights with gradient at adversarial point."""
        # Restore original parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']

        # Perform base optimizer step
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm

# Usage
base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)

for batch in dataloader:
    # First forward-backward pass
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # Second forward-backward pass
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.second_step(zero_grad=True)
`,
    useCases: [
      'Improving generalization',
      'Image classification',
      'When test performance lags training',
    ],
    advantages: [
      'Better generalization',
      'Finds flatter minima',
      'Works with any base optimizer',
    ],
    disadvantages: [
      '2x forward-backward passes per step',
      'Doubles compute time',
      'rho requires tuning',
    ],
    references: [
      'Sharpness-Aware Minimization for Efficiently Improving Generalization (Foret et al., 2020)',
    ],
  },

  /**
   * ASAM (Adaptive SAM)
   */
  asam: {
    name: 'ASAM (Adaptive Sharpness-Aware Minimization)',
    description: 'SAM with adaptive perturbation sizes per parameter',
    hyperparameters: {
      base_optimizer: 'Base optimizer',
      rho: 'Base neighborhood size (default 0.5)',
      eta: 'Normalization coefficient (default 0.01)',
    },
    implementation: `
import torch

class ASAM(torch.optim.Optimizer):
    """Adaptive SAM with parameter-wise normalization."""

    def __init__(self, params, base_optimizer, rho=0.5, eta=0.01):
        defaults = dict(rho=rho, eta=eta)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Adaptive perturbation step."""
        wgrads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Store original weights
                self.state[p]['old_p'] = p.data.clone()

                # Compute T_w (normalization operator)
                t_w = torch.pow(p, 2)
                t_w.add_(group['eta'])

                # Weight gradient for norm computation
                wgrads.append((t_w * p.grad).view(-1))

        # Compute weighted gradient norm
        wgrad_norm = torch.cat(wgrads).norm(2)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Compute T_w
                t_w = torch.pow(p, 2)
                t_w.add_(group['eta'])

                # Adaptive epsilon
                e_w = t_w * p.grad * group['rho'] / (wgrad_norm + 1e-12)

                # Perturb weights
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

# Usage (larger rho than standard SAM)
base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = ASAM(model.parameters(), base_optimizer, rho=0.5)
`,
    useCases: [
      'When SAM underperforms',
      'Heterogeneous parameter scales',
      'Better than SAM in many cases',
    ],
    advantages: [
      'More robust than SAM',
      'Larger effective rho',
      'Parameter-wise adaptation',
    ],
    disadvantages: [
      'Still 2x compute',
      'More hyperparameters',
      'More complex implementation',
    ],
    references: [
      'ASAM: Adaptive Sharpness-Aware Minimization (Kwon et al., 2021)',
    ],
  },

  /**
   * AdaFactor
   */
  adafactor: {
    name: 'AdaFactor',
    description: 'Memory-efficient adaptive optimizer with factored second moments',
    hyperparameters: {
      lr: 'Learning rate (optional if using relative_step)',
      eps: 'Epsilon terms (tuple)',
      clip_threshold: 'Gradient clipping threshold',
      decay_rate: 'Second moment decay rate',
      scale_parameter: 'Scale LR by root mean square of parameter',
      relative_step: 'Use relative step size',
      warmup_init: 'Use warmup initialization',
    },
    implementation: `
import torch
import math

class Adafactor(torch.optim.Optimizer):
    """Memory-efficient optimizer with factored second moments."""

    def __init__(self, params, lr=None, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, beta1=None, weight_decay=0.0,
                 scale_parameter=True, relative_step=True, warmup_init=False):

        if lr is not None and relative_step:
            raise ValueError("Cannot use both lr and relative_step")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step")

        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                       decay_rate=decay_rate, beta1=beta1, weight_decay=weight_decay,
                       scale_parameter=scale_parameter, relative_step=relative_step,
                       warmup_init=warmup_init)
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        """Compute learning rate."""
        rel_step_sz = param_group['lr']
        if param_group['relative_step']:
            min_step = 1e-6 * param_state['step'] if param_group['warmup_init'] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))

        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps'][1], param_state['RMS'])

        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        """Determine if factored or full second moments."""
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                grad_shape = grad.shape
                factored, use_first_moment = self._get_options(group, grad_shape)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if use_first_moment:
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1])
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[:-2] + grad_shape[-1:])
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                    state['RMS'] = 0
                else:
                    if use_first_moment:
                        state['exp_avg'] = state['exp_avg'].to(grad)
                    if factored:
                        state['exp_avg_sq_row'] = state['exp_avg_sq_row'].to(grad)
                        state['exp_avg_sq_col'] = state['exp_avg_sq_col'].to(grad)
                    else:
                        state['exp_avg_sq'] = state['exp_avg_sq'].to(grad)

                state['step'] += 1
                state['RMS'] = p.data.pow(2).mean().sqrt().item()
                lr = self._get_lr(group, state)

                # ... (remaining computation follows original Adafactor)

        return loss
`,
    useCases: [
      'Large language models (T5, etc.)',
      'Memory-constrained training',
      'When Adam uses too much memory',
    ],
    advantages: [
      'O(n) memory vs O(2n) for Adam',
      'Automatic LR scaling',
      'Built-in gradient clipping',
    ],
    disadvantages: [
      'More complex implementation',
      'May be slower per step',
      'Less intuitive hyperparameters',
    ],
    references: [
      'Adafactor: Adaptive Learning Rates with Sublinear Memory Cost (Shazeer & Stern, 2018)',
    ],
  },
};

// =============================================================================
// GRADIENT HANDLING STRATEGIES
// =============================================================================

export const GradientStrategies: Record<string, {
  name: string;
  description: string;
  implementation: string;
  useCases: string[];
  tips: string[];
}> = {
  /**
   * Gradient clipping by norm
   */
  clip_grad_norm: {
    name: 'Gradient Clipping by Norm',
    description: 'Scale down gradients when total norm exceeds threshold',
    implementation: `
import torch

# PyTorch built-in (most common)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Manual implementation
def clip_grad_norm(parameters, max_norm):
    """Clip gradients by global norm."""
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = torch.norm(
        torch.stack([p.grad.norm(2) for p in parameters]), 2
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.mul_(clip_coef)
    return total_norm

# Usage in training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()

    # Clip gradients before optimizer step
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optional: Log gradient norm for debugging
    if grad_norm > 1.0:
        print(f"Gradients clipped: {grad_norm:.2f} -> 1.0")

    optimizer.step()
`,
    useCases: [
      'RNN/LSTM training',
      'Transformer training',
      'Preventing gradient explosion',
      'Any deep network training',
    ],
    tips: [
      'max_norm=1.0 is a good default',
      'Use 0.5 for more aggressive clipping',
      'Log clipping frequency to detect instability',
      'Clip before optimizer.step()',
    ],
  },

  /**
   * Gradient clipping by value
   */
  clip_grad_value: {
    name: 'Gradient Clipping by Value',
    description: 'Clip individual gradient values to range [-clip_value, clip_value]',
    implementation: `
import torch

# PyTorch built-in
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Manual implementation
def clip_grad_value(parameters, clip_value):
    """Clip each gradient element to [-clip_value, clip_value]."""
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.clamp_(-clip_value, clip_value)

# Usage
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
optimizer.step()
`,
    useCases: [
      'When norm clipping is too aggressive',
      'Preserving gradient direction',
      'Specific numerical stability issues',
    ],
    tips: [
      'Less common than norm clipping',
      'Changes gradient direction',
      'Typical values: 0.5-1.0',
      'Consider norm clipping first',
    ],
  },

  /**
   * Adaptive gradient clipping (AGC)
   */
  adaptive_gradient_clipping: {
    name: 'Adaptive Gradient Clipping (AGC)',
    description: 'Clip based on gradient-to-weight ratio per parameter',
    implementation: `
import torch

def adaptive_gradient_clipping(parameters, clip_factor=0.01, eps=1e-3):
    """AGC: Clip gradients based on unit-wise gradient-to-weight ratio."""
    for p in filter(lambda p: p.grad is not None, parameters):
        # Compute norms
        p_norm = p.norm(2)
        g_norm = p.grad.norm(2)

        # Compute clipping threshold
        max_norm = p_norm * clip_factor

        # Clip if necessary
        trigger = g_norm > max_norm
        if trigger:
            clipped_grad = p.grad * (max_norm / (g_norm + eps))
            p.grad.copy_(clipped_grad)

    return

# Improved version (NFNet paper)
def agc_nfnet(parameters, clip_factor=0.01, eps=1e-3):
    """AGC as used in NFNet."""
    for p in filter(lambda p: p.grad is not None, parameters):
        # Skip 1D parameters (bias, normalization)
        if p.ndim == 1:
            continue

        # Unit-wise clipping
        p_norm = p.norm(2, dim=tuple(range(1, p.ndim)), keepdim=True).clamp(min=eps)
        g_norm = p.grad.norm(2, dim=tuple(range(1, p.ndim)), keepdim=True)

        max_norm = p_norm * clip_factor
        trigger = g_norm > max_norm
        clipped_grad = p.grad * (max_norm / g_norm.clamp(min=eps))
        p.grad.copy_(torch.where(trigger, clipped_grad, p.grad))

# Usage
optimizer.zero_grad()
loss.backward()
adaptive_gradient_clipping(model.parameters(), clip_factor=0.01)
optimizer.step()
`,
    useCases: [
      'NFNet and normalizer-free networks',
      'Very deep networks',
      'When standard clipping fails',
    ],
    tips: [
      'clip_factor=0.01 is standard',
      'Skip bias and norm parameters',
      'Works well without BatchNorm',
      'Prevents instability in deep networks',
    ],
  },

  /**
   * Gradient accumulation
   */
  gradient_accumulation: {
    name: 'Gradient Accumulation',
    description: 'Accumulate gradients over multiple steps for larger effective batch',
    implementation: `
import torch

accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    # Forward pass
    outputs = model(batch['input'])
    loss = criterion(outputs, batch['target'])

    # Scale loss for accumulation
    loss = loss / accumulation_steps
    loss.backward()

    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        # Optional: Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

        # Scheduler step (if per-step)
        scheduler.step()

# Handle remaining steps at end of epoch
if (i + 1) % accumulation_steps != 0:
    optimizer.step()
    optimizer.zero_grad()

# For mixed precision training
scaler = torch.cuda.amp.GradScaler()
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    with torch.cuda.amp.autocast():
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target']) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
`,
    useCases: [
      'Large models with limited GPU memory',
      'Simulating larger batch sizes',
      'Maintaining batch statistics',
    ],
    tips: [
      'Effective batch = micro_batch * accumulation_steps',
      'Scale loss by 1/accumulation_steps',
      'Be careful with BatchNorm statistics',
      'Sync gradients in distributed only at update step',
    ],
  },

  /**
   * Gradient centralization
   */
  gradient_centralization: {
    name: 'Gradient Centralization',
    description: 'Center gradients to have zero mean for faster convergence',
    implementation: `
import torch

def centralize_gradient(grad):
    """Remove mean from gradient (for weight matrices only)."""
    if grad.ndim > 1:
        grad.add_(-grad.mean(dim=tuple(range(1, grad.ndim)), keepdim=True))
    return grad

# Apply during training
for p in model.parameters():
    if p.grad is not None and p.ndim > 1:
        centralize_gradient(p.grad)

# Integrated into optimizer
class SGD_GC(torch.optim.SGD):
    """SGD with Gradient Centralization."""

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.ndim > 1:
                    # Centralize gradient
                    p.grad.add_(-p.grad.mean(dim=tuple(range(1, p.grad.ndim)), keepdim=True))

        return super().step(closure)

# Usage
optimizer = SGD_GC(model.parameters(), lr=0.1, momentum=0.9)
`,
    useCases: [
      'Faster convergence',
      'Better generalization',
      'Works with any optimizer',
    ],
    tips: [
      'Only apply to weights, not biases',
      'Skip for 1D parameters',
      'Can improve convergence by 10-20%',
      'No additional hyperparameters',
    ],
  },

  /**
   * Gradient noise injection
   */
  gradient_noise: {
    name: 'Gradient Noise Injection',
    description: 'Add Gaussian noise to gradients for regularization',
    implementation: `
import torch
import math

def add_gradient_noise(parameters, noise_stddev, t, gamma=0.55):
    """Add annealed gradient noise."""
    # Annealing schedule: stddev / (1 + t)^gamma
    stddev = noise_stddev / (1 + t) ** gamma

    for p in filter(lambda p: p.grad is not None, parameters):
        noise = torch.randn_like(p.grad) * stddev
        p.grad.add_(noise)

# Usage
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()

    # Add noise that decreases over training
    add_gradient_noise(model.parameters(), noise_stddev=0.01, t=step)

    optimizer.step()

# Constant noise (simpler)
def add_constant_noise(parameters, stddev):
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.add_(torch.randn_like(p.grad) * stddev)
`,
    useCases: [
      'Escaping local minima',
      'Regularization',
      'Exploration in optimization',
    ],
    tips: [
      'Anneal noise over training',
      'Start with small noise (0.01-0.1)',
      'Can help generalization',
      'Similar effect to larger batch size',
    ],
  },
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Get all optimization strategies by category.
 */
export function getAllOptimizationStrategies(): Record<string, unknown> {
  return {
    warmup: WarmupStrategies,
    lr_schedules: LearningRateSchedules,
    optimizers: AdvancedOptimizers,
    gradient_handling: GradientStrategies,
  };
}

/**
 * Recommend optimization strategy based on model and training setup.
 */
export function recommendStrategy(setup: {
  modelType: string;
  batchSize: number;
  totalSteps: number;
  gpuMemory: string;
}): {
  optimizer: string;
  scheduler: string;
  warmup: string;
  gradientHandling: string;
  rationale: string;
} {
  const isLargeBatch = setup.batchSize >= 1024;
  const isTransformer = setup.modelType.toLowerCase().includes('transformer') ||
                        setup.modelType.toLowerCase().includes('bert') ||
                        setup.modelType.toLowerCase().includes('gpt');

  let optimizer = 'AdamW';
  let scheduler = 'cosine_annealing';
  let warmup = 'linear_warmup';
  let gradientHandling = 'clip_grad_norm';
  let rationale = '';

  if (isLargeBatch) {
    optimizer = isTransformer ? 'LAMB' : 'LARS';
    rationale += `Using ${optimizer} for large batch training (${setup.batchSize}). `;
  }

  if (isTransformer) {
    warmup = 'linear_warmup';
    gradientHandling = 'clip_grad_norm';
    rationale += 'Linear warmup and gradient clipping for Transformer stability. ';
  }

  if (setup.totalSteps < 10000) {
    scheduler = 'one_cycle';
    rationale += 'One-cycle policy for fast training. ';
  } else {
    scheduler = 'cosine_annealing';
    rationale += 'Cosine annealing for long training. ';
  }

  return {
    optimizer,
    scheduler,
    warmup,
    gradientHandling,
    rationale: rationale.trim(),
  };
}

/**
 * Generate learning rate schedule visualization code.
 */
export function generateScheduleVisualization(
  scheduleName: string,
  totalSteps: number,
  peakLr: number
): string {
  return `
import matplotlib.pyplot as plt
import numpy as np

# Generate schedule
steps = np.arange(${totalSteps})
lrs = []

# Add your schedule computation here
# Example for cosine annealing:
warmup_steps = ${Math.floor(totalSteps * 0.1)}
for step in steps:
    if step < warmup_steps:
        lr = ${peakLr} * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (${totalSteps} - warmup_steps)
        lr = ${peakLr * 0.1} + (${peakLr} - ${peakLr * 0.1}) * (1 + np.cos(np.pi * progress)) / 2
    lrs.append(lr)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(steps, lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('${scheduleName} Learning Rate Schedule')
plt.grid(True, alpha=0.3)
plt.savefig('lr_schedule.png', dpi=150, bbox_inches='tight')
plt.show()
`;
}

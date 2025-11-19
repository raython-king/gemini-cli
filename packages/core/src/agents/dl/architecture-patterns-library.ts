/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Architecture Patterns Library
 *
 * A comprehensive collection of common architecture patterns used in deep learning.
 * This library serves as a knowledge base for architecture analysis and suggestion agents.
 */

/**
 * Attention pattern definition.
 */
export interface AttentionPattern {
  name: string;
  abbreviation: string;
  description: string;
  complexity: {
    time: string;
    memory: string;
  };
  kvCacheSize: string;
  pros: string[];
  cons: string[];
  useCases: string[];
  examples: string[];
  codePattern: string;
  papers: string[];
}

/**
 * Comprehensive attention patterns database.
 */
export const ATTENTION_PATTERNS: Record<string, AttentionPattern> = {
  MHA: {
    name: 'Multi-Head Attention',
    abbreviation: 'MHA',
    description:
      'Standard attention with multiple parallel attention heads, each with separate Q, K, V projections.',
    complexity: {
      time: 'O(n^2 * d)',
      memory: 'O(n^2 + n*d)',
    },
    kvCacheSize: 'O(layers * heads * seq_len * head_dim)',
    pros: [
      'High quality representations',
      'Well-understood behavior',
      'Stable training',
      'Good for most tasks',
    ],
    cons: [
      'Quadratic memory in sequence length',
      'Large KV cache for inference',
      'Can be slow for long sequences',
    ],
    useCases: [
      'General language modeling',
      'Short to medium sequences',
      'Tasks requiring high quality',
    ],
    examples: ['BERT', 'GPT-2', 'Original Transformer'],
    codePattern: `
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)`,
    papers: ['Attention Is All You Need (Vaswani et al., 2017)'],
  },

  MQA: {
    name: 'Multi-Query Attention',
    abbreviation: 'MQA',
    description:
      'Attention where all heads share the same K and V projections, reducing KV cache size.',
    complexity: {
      time: 'O(n^2 * d)',
      memory: 'O(n^2 + n*d/h)',
    },
    kvCacheSize: 'O(layers * seq_len * head_dim)',
    pros: [
      'Dramatically reduced KV cache (h times smaller)',
      'Faster inference with batched queries',
      'Lower memory bandwidth requirements',
    ],
    cons: [
      'Potential quality degradation vs MHA',
      'Needs careful training',
      'May require more parameters elsewhere',
    ],
    useCases: [
      'Large batch inference',
      'Memory-constrained deployment',
      'High-throughput serving',
    ],
    examples: ['PaLM', 'Falcon', 'StarCoder'],
    codePattern: `
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)  # Shared K
        self.v_proj = nn.Linear(d_model, self.head_dim)  # Shared V
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).unsqueeze(1)  # Broadcast across heads
        v = self.v_proj(x).unsqueeze(1)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)`,
    papers: ['Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019)'],
  },

  GQA: {
    name: 'Grouped-Query Attention',
    abbreviation: 'GQA',
    description:
      'Attention where heads are divided into groups, each group sharing K and V projections. Balances MHA quality with MQA efficiency.',
    complexity: {
      time: 'O(n^2 * d)',
      memory: 'O(n^2 + n*d*g/h)',
    },
    kvCacheSize: 'O(layers * n_kv_heads * seq_len * head_dim)',
    pros: [
      'Good balance of quality and efficiency',
      'Reduced KV cache (h/g times smaller than MHA)',
      'Better quality than MQA',
      'Flexible trade-off via group count',
    ],
    cons: [
      'Slightly more complex than MQA',
      'Still some quality loss vs MHA',
    ],
    useCases: [
      'Production LLM deployment',
      'Balancing quality and efficiency',
      'Scalable inference',
    ],
    examples: ['LLaMA-2', 'Mistral', 'Qwen-2'],
    codePattern: `
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV for each group
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(x)`,
    papers: ['GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)'],
  },

  MLA: {
    name: 'Multi-Head Latent Attention',
    abbreviation: 'MLA',
    description:
      'Attention using low-rank compression of KV cache through learned latent vectors. Achieves extreme KV cache compression.',
    complexity: {
      time: 'O(n^2 * d)',
      memory: 'O(n^2 + n*d_latent)',
    },
    kvCacheSize: 'O(layers * seq_len * latent_dim)',
    pros: [
      'Extreme KV cache compression (10x+ reduction)',
      'Maintains quality through learned compression',
      'Efficient for very long contexts',
    ],
    cons: [
      'More complex implementation',
      'Requires careful latent dimension tuning',
      'Novel technique with less adoption',
    ],
    useCases: [
      'Very long context models',
      'Extreme memory efficiency',
      'Large-scale deployment',
    ],
    examples: ['DeepSeek-V2', 'DeepSeek-V3'],
    codePattern: `
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, latent_dim):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.latent_dim = latent_dim

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_compress = nn.Linear(d_model, latent_dim)  # Compress to latent
        self.k_expand = nn.Linear(latent_dim, d_model)     # Expand K from latent
        self.v_expand = nn.Linear(latent_dim, d_model)     # Expand V from latent
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Compress to latent, then expand
        latent = self.kv_compress(x)  # This is cached
        k = self.k_expand(latent).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_expand(latent).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)`,
    papers: ['DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (DeepSeek, 2024)'],
  },

  FLASH: {
    name: 'Flash Attention',
    abbreviation: 'Flash',
    description:
      'IO-aware exact attention that computes attention in tiles to minimize memory reads/writes. Not a new attention type but an efficient implementation.',
    complexity: {
      time: 'O(n^2 * d)',
      memory: 'O(n)',
    },
    kvCacheSize: 'Same as underlying attention type',
    pros: [
      'Exact attention (no approximation)',
      '2-4x faster than standard attention',
      'Linear memory in sequence length',
      'Better utilizes GPU compute',
    ],
    cons: [
      'Requires specialized CUDA kernels',
      'Not all features supported (some masks)',
      'Platform-specific optimizations needed',
    ],
    useCases: [
      'All attention-based models',
      'Long sequence training',
      'Memory-efficient training',
    ],
    examples: ['Used in LLaMA, Mistral, most modern LLMs'],
    codePattern: `
# Using Flash Attention 2
from flash_attn import flash_attn_func

class FlashAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        # Flash attention expects (B, N, H, D)
        x = flash_attn_func(q, k, v, causal=True)
        return self.out_proj(x.reshape(B, N, C))`,
    papers: ['FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)'],
  },

  SLIDING_WINDOW: {
    name: 'Sliding Window Attention',
    abbreviation: 'SWA',
    description:
      'Each token attends only to a fixed window of surrounding tokens, reducing complexity from quadratic to linear.',
    complexity: {
      time: 'O(n * w * d)',
      memory: 'O(n * w)',
    },
    kvCacheSize: 'O(layers * window_size * head_dim)',
    pros: [
      'Linear complexity in sequence length',
      'Fixed KV cache size',
      'Efficient for very long sequences',
      'Stacks to large effective receptive field',
    ],
    cons: [
      'Limited direct long-range attention',
      'May miss global patterns',
      'Needs many layers for large receptive field',
    ],
    useCases: [
      'Very long sequences',
      'Streaming applications',
      'When local context is sufficient',
    ],
    examples: ['Mistral', 'Longformer'],
    codePattern: `
class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size):
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, C = x.shape
        # Create sliding window mask
        mask = torch.ones(N, N, dtype=torch.bool)
        for i in range(N):
            start = max(0, i - self.window_size // 2)
            end = min(N, i + self.window_size // 2 + 1)
            mask[i, start:end] = False

        qkv = self.qkv_proj(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(mask, float('-inf'))
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)`,
    papers: ['Longformer: The Long-Document Transformer (Beltagy et al., 2020)'],
  },

  LINEAR: {
    name: 'Linear Attention',
    abbreviation: 'Linear',
    description:
      'Replaces softmax attention with kernel feature maps, enabling O(n) complexity through associativity of matrix multiplication.',
    complexity: {
      time: 'O(n * d^2)',
      memory: 'O(n * d)',
    },
    kvCacheSize: 'O(layers * d^2)',
    pros: [
      'True linear complexity in sequence length',
      'Constant time per token in autoregressive',
      'Can handle very long sequences',
    ],
    cons: [
      'Often lower quality than softmax attention',
      'Requires careful kernel design',
      'May need more layers for same quality',
    ],
    useCases: [
      'Extremely long sequences',
      'When speed is critical',
      'Linear complexity required',
    ],
    examples: ['Linear Transformer', 'RWKV', 'RetNet'],
    codePattern: `
class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, feature_dim=None):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.feature_dim = feature_dim or self.head_dim
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.feature_map = nn.Linear(self.head_dim, self.feature_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Apply feature map (e.g., elu + 1)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Linear attention: (Q @ K^T) @ V = Q @ (K^T @ V)
        kv = k.transpose(-2, -1) @ v  # (B, H, D, D)
        x = (q @ kv).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)`,
    papers: ['Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (Katharopoulos et al., 2020)'],
  },
};

/**
 * Normalization pattern definition.
 */
export interface NormalizationPattern {
  name: string;
  description: string;
  formula: string;
  pros: string[];
  cons: string[];
  useCases: string[];
  examples: string[];
  codePattern: string;
}

/**
 * Comprehensive normalization patterns database.
 */
export const NORMALIZATION_PATTERNS: Record<string, NormalizationPattern> = {
  LAYER_NORM: {
    name: 'Layer Normalization',
    description:
      'Normalizes across all features for each sample. Standard for transformers.',
    formula: 'y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta',
    pros: [
      'Batch-size independent',
      'Works well for variable sequence lengths',
      'Stable training for transformers',
    ],
    cons: [
      'Slightly slower than RMSNorm',
      'Computing mean is not strictly necessary',
    ],
    useCases: ['Transformers', 'RNNs', 'Variable batch sizes'],
    examples: ['BERT', 'GPT-2', 'Original Transformer'],
    codePattern: `
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta`,
  },

  RMS_NORM: {
    name: 'Root Mean Square Normalization',
    description:
      'Normalizes using only RMS (no mean subtraction). 10-15% faster than LayerNorm.',
    formula: 'y = x / sqrt(mean(x^2) + eps) * gamma',
    pros: [
      '10-15% faster than LayerNorm',
      'Similar quality to LayerNorm',
      'Simpler computation',
    ],
    cons: [
      'No centering (no mean subtraction)',
      'Slightly different training dynamics',
    ],
    useCases: ['Modern LLMs', 'When speed matters', 'Large-scale training'],
    examples: ['LLaMA', 'Mistral', 'Qwen', 'GPT-NeoX'],
    codePattern: `
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight`,
  },

  BATCH_NORM: {
    name: 'Batch Normalization',
    description:
      'Normalizes across batch dimension. Standard for CNNs, uses running statistics at inference.',
    formula: 'y = (x - batch_mean) / sqrt(batch_var + eps) * gamma + beta',
    pros: [
      'Enables higher learning rates',
      'Regularization effect',
      'Accelerates training for CNNs',
    ],
    cons: [
      'Batch-size dependent',
      'Different behavior train vs eval',
      'Issues with small batches',
    ],
    useCases: ['CNNs', 'Large batch training', 'Image classification'],
    examples: ['ResNet', 'EfficientNet', 'Most CNNs'],
    codePattern: `
# PyTorch built-in
nn.BatchNorm2d(num_features)
nn.BatchNorm1d(num_features)`,
  },

  GROUP_NORM: {
    name: 'Group Normalization',
    description:
      'Divides channels into groups and normalizes within each group. Batch-size independent alternative to BatchNorm.',
    formula: 'y = (x - group_mean) / sqrt(group_var + eps) * gamma + beta',
    pros: [
      'Batch-size independent',
      'Good for small batch training',
      'Works with detection/segmentation',
    ],
    cons: [
      'Need to choose number of groups',
      'Slightly slower than BatchNorm',
    ],
    useCases: ['Detection', 'Segmentation', 'Small batch training'],
    examples: ['Mask R-CNN', 'DETR'],
    codePattern: `
# PyTorch built-in
nn.GroupNorm(num_groups, num_channels)`,
  },

  DEEP_NORM: {
    name: 'DeepNorm',
    description:
      'Combines residual scaling with modified initialization for training very deep models (1000+ layers).',
    formula: 'y = LayerNorm(x * alpha + sublayer(x))',
    pros: [
      'Enables training 1000+ layer models',
      'Stable gradients in very deep networks',
      'Works with standard optimizers',
    ],
    cons: [
      'Requires coordinated init and scaling',
      'Specific alpha/beta formulas',
    ],
    useCases: ['Very deep transformers', 'Scaling depth experiments'],
    examples: ['DeepNet'],
    codePattern: `
class DeepNormTransformerBlock(nn.Module):
    def __init__(self, d_model, n_layers):
        # alpha = (2 * n_layers) ^ 0.25
        # beta = (8 * n_layers) ^ -0.25 for Xavier init
        self.alpha = (2 * n_layers) ** 0.25
        self.attn = MultiHeadAttention(d_model)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x * self.alpha + self.attn(x))
        x = self.norm2(x * self.alpha + self.ffn(x))
        return x`,
  },
};

/**
 * Skip connection pattern definition.
 */
export interface SkipConnectionPattern {
  name: string;
  description: string;
  pros: string[];
  cons: string[];
  examples: string[];
  codePattern: string;
}

/**
 * Skip connection patterns database.
 */
export const SKIP_CONNECTION_PATTERNS: Record<string, SkipConnectionPattern> = {
  RESIDUAL: {
    name: 'Residual Connection',
    description: 'Adds input to output: y = x + f(x). Standard for deep networks.',
    pros: [
      'Enables training very deep networks',
      'Gradient highway',
      'Easy optimization',
    ],
    cons: [
      'May need scaling for very deep models',
      'Identity mapping may not be optimal',
    ],
    examples: ['ResNet', 'Transformer', 'Most modern architectures'],
    codePattern: `
def residual_block(x):
    return x + sublayer(x)`,
  },

  PRE_NORM: {
    name: 'Pre-Normalization',
    description: 'Normalizes before sublayer: y = x + f(norm(x)). More stable training.',
    pros: [
      'More stable gradients',
      'Better for deep models',
      'Easier to train',
    ],
    cons: [
      'May have slightly different final quality',
      'Needs final normalization after last layer',
    ],
    examples: ['GPT-2', 'LLaMA', 'Most modern LLMs'],
    codePattern: `
def pre_norm_block(x):
    return x + sublayer(norm(x))`,
  },

  POST_NORM: {
    name: 'Post-Normalization',
    description: 'Normalizes after sublayer: y = norm(x + f(x)). Original transformer style.',
    pros: [
      'Can achieve slightly better final quality',
      'Original transformer design',
    ],
    cons: [
      'Harder to train deep models',
      'Gradient issues in very deep networks',
    ],
    examples: ['Original Transformer', 'BERT'],
    codePattern: `
def post_norm_block(x):
    return norm(x + sublayer(x))`,
  },

  DENSE: {
    name: 'Dense Connection',
    description: 'Concatenates all previous outputs: y = [x_0, x_1, ..., f(x_n)].',
    pros: [
      'Feature reuse',
      'Strong gradient flow',
      'Parameter efficient',
    ],
    cons: [
      'Memory intensive',
      'Increasing channel dimensions',
    ],
    examples: ['DenseNet', 'DenseFormer'],
    codePattern: `
def dense_block(features):
    for layer in layers:
        new_features = layer(torch.cat(features, dim=1))
        features.append(new_features)
    return torch.cat(features, dim=1)`,
  },

  PARALLEL: {
    name: 'Parallel Attention + FFN',
    description: 'Computes attention and FFN in parallel: y = x + attn(x) + ffn(x).',
    pros: [
      'Better GPU utilization',
      '~15% training speedup',
      'Similar quality to sequential',
    ],
    cons: [
      'Different architecture than standard',
      'May need re-tuning hyperparameters',
    ],
    examples: ['GPT-J', 'PaLM'],
    codePattern: `
def parallel_block(x):
    x_norm = norm(x)
    return x + attn(x_norm) + ffn(x_norm)`,
  },
};

/**
 * Efficient convolution pattern definition.
 */
export interface EfficientConvPattern {
  name: string;
  description: string;
  parameterReduction: string;
  flopReduction: string;
  pros: string[];
  cons: string[];
  examples: string[];
  codePattern: string;
}

/**
 * Efficient convolution patterns database.
 */
export const EFFICIENT_CONV_PATTERNS: Record<string, EfficientConvPattern> = {
  DEPTHWISE_SEPARABLE: {
    name: 'Depthwise Separable Convolution',
    description:
      'Factorizes standard conv into depthwise (spatial) and pointwise (channel mixing).',
    parameterReduction: '~8-9x for 3x3 conv',
    flopReduction: '~8-9x for 3x3 conv',
    pros: [
      'Massive parameter reduction',
      'Much faster than standard conv',
      'Good for mobile/edge',
    ],
    cons: [
      'May have lower capacity',
      'Two operations instead of one',
    ],
    examples: ['MobileNet', 'Xception', 'EfficientNet'],
    codePattern: `
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, groups=in_ch, padding=kernel_size//2)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))`,
  },

  INVERTED_RESIDUAL: {
    name: 'Inverted Residual Block',
    description:
      'Expands channels, applies depthwise conv, then projects back. MobileNetV2 core block.',
    parameterReduction: 'Efficient for narrow->wide->narrow',
    flopReduction: 'Efficient with expansion in depthwise',
    pros: [
      'Efficient for low-dimensional inputs',
      'Good capacity/efficiency ratio',
      'Standard for mobile nets',
    ],
    cons: [
      'Needs careful expansion ratio tuning',
      'Memory for expanded activations',
    ],
    examples: ['MobileNetV2', 'MobileNetV3', 'EfficientNet'],
    codePattern: `
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=6):
        hidden = in_ch * expand_ratio
        self.expand = nn.Conv2d(in_ch, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.project = nn.Conv2d(hidden, out_ch, 1)
        self.use_residual = in_ch == out_ch

    def forward(self, x):
        out = F.relu6(self.expand(x))
        out = F.relu6(self.depthwise(out))
        out = self.project(out)
        return x + out if self.use_residual else out`,
  },

  BOTTLENECK: {
    name: 'Bottleneck Block',
    description: '1x1 reduce -> 3x3 conv -> 1x1 expand. ResNet core block for deeper models.',
    parameterReduction: '~3-4x vs standard 3x3 stack',
    flopReduction: '~3-4x vs standard 3x3 stack',
    pros: [
      'Efficient for deep networks',
      'Maintains receptive field',
      'Standard for ResNet-50+',
    ],
    cons: [
      'Information bottleneck in middle',
      'Needs careful width tuning',
    ],
    examples: ['ResNet-50/101/152', 'ResNeXt'],
    codePattern: `
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, width):
        out_ch = width * self.expansion
        self.conv1 = nn.Conv2d(in_ch, width, 1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.conv3 = nn.Conv2d(width, out_ch, 1)
        self.bn1, self.bn2, self.bn3 = [nn.BatchNorm2d(c) for c in [width, width, out_ch]]
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))`,
  },
};

/**
 * Modern transformer variant definition.
 */
export interface TransformerVariant {
  name: string;
  description: string;
  keyInnovations: string[];
  performance: string;
  useCases: string[];
  architecture: {
    attention: string;
    normalization: string;
    activation: string;
    positionEncoding: string;
  };
  papers: string[];
}

/**
 * Modern transformer variants database.
 */
export const TRANSFORMER_VARIANTS: Record<string, TransformerVariant> = {
  LLAMA: {
    name: 'LLaMA',
    description: 'Efficient open-source LLM architecture from Meta with modern design choices.',
    keyInnovations: [
      'RMSNorm instead of LayerNorm',
      'SwiGLU activation',
      'RoPE positional encoding',
      'GQA (in LLaMA-2)',
      'Pre-normalization',
    ],
    performance: 'Competitive with GPT-3 at smaller sizes',
    useCases: ['General language modeling', 'Open-source LLM base'],
    architecture: {
      attention: 'GQA (MHA in LLaMA-1)',
      normalization: 'RMSNorm (pre-norm)',
      activation: 'SwiGLU',
      positionEncoding: 'RoPE',
    },
    papers: ['LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)'],
  },

  MISTRAL: {
    name: 'Mistral',
    description: 'Efficient LLM with sliding window attention for extended context.',
    keyInnovations: [
      'Sliding window attention',
      'Rolling buffer KV cache',
      'GQA',
      'Pre-fill and chunking',
    ],
    performance: 'Outperforms LLaMA-2 13B at 7B parameters',
    useCases: ['Long context', 'Efficient inference', 'Production deployment'],
    architecture: {
      attention: 'GQA with sliding window',
      normalization: 'RMSNorm',
      activation: 'SwiGLU',
      positionEncoding: 'RoPE',
    },
    papers: ['Mistral 7B (Jiang et al., 2023)'],
  },

  GPT2: {
    name: 'GPT-2',
    description: 'Influential autoregressive transformer establishing many modern practices.',
    keyInnovations: [
      'Pre-normalization',
      'Learned position embeddings',
      'Scaled initialization',
      'GELU activation',
    ],
    performance: 'Strong zero-shot performance',
    useCases: ['Text generation', 'Foundation for many models'],
    architecture: {
      attention: 'MHA',
      normalization: 'LayerNorm (pre-norm)',
      activation: 'GELU',
      positionEncoding: 'Learned',
    },
    papers: ['Language Models are Unsupervised Multitask Learners (Radford et al., 2019)'],
  },

  MAMBA: {
    name: 'Mamba',
    description: 'State space model with selective state spaces for linear complexity.',
    keyInnovations: [
      'Selective state spaces',
      'Input-dependent parameters',
      'Linear time complexity',
      'Hardware-efficient implementation',
    ],
    performance: 'Competitive with transformers, much faster',
    useCases: ['Very long sequences', 'Linear complexity needed', 'Efficient training'],
    architecture: {
      attention: 'None (state space)',
      normalization: 'RMSNorm',
      activation: 'SiLU',
      positionEncoding: 'Implicit in SSM',
    },
    papers: ['Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)'],
  },
};

/**
 * Position encoding pattern definition.
 */
export interface PositionEncodingPattern {
  name: string;
  abbreviation: string;
  description: string;
  maxLength: string;
  extrapolation: string;
  pros: string[];
  cons: string[];
  examples: string[];
  codePattern: string;
}

/**
 * Position encoding patterns database.
 */
export const POSITION_ENCODING_PATTERNS: Record<string, PositionEncodingPattern> = {
  ROPE: {
    name: 'Rotary Position Embedding',
    abbreviation: 'RoPE',
    description:
      'Encodes positions by rotating query and key vectors. Relative position naturally emerges.',
    maxLength: 'Theoretically infinite, practical limit depends on training',
    extrapolation: 'Good with techniques like NTK-aware scaling, YaRN',
    pros: [
      'Good length extrapolation',
      'Relative position emerges naturally',
      'Efficient implementation',
      'No extra parameters',
    ],
    cons: [
      'Extrapolation needs techniques for very long contexts',
      'Fixed base frequency',
    ],
    examples: ['LLaMA', 'Mistral', 'PaLM', 'GPT-NeoX'],
    codePattern: `
def apply_rotary_emb(x, freqs):
    # x: (B, H, N, D)
    # freqs: (N, D//2)
    x_rot = x[..., :x.shape[-1]//2]
    x_pass = x[..., x.shape[-1]//2:]

    # Apply rotation
    cos = freqs.cos()
    sin = freqs.sin()
    x_rot = x_rot * cos - x_rot.flip(-1) * sin

    return torch.cat([x_rot, x_pass], dim=-1)`,
  },

  ALIBI: {
    name: 'Attention with Linear Biases',
    abbreviation: 'ALiBi',
    description:
      'Adds linear distance-based bias to attention scores. Excellent extrapolation.',
    maxLength: 'Infinite (can extrapolate to any length)',
    extrapolation: 'Excellent, designed for length generalization',
    pros: [
      'Perfect length extrapolation',
      'No extra parameters',
      'Simple to implement',
    ],
    cons: [
      'Adds bias computation',
      'May not work well for all tasks',
    ],
    examples: ['BLOOM', 'MPT'],
    codePattern: `
def get_alibi_slopes(n_heads):
    # Returns slopes for each head
    ratio = 2 ** (-8 / n_heads)
    return torch.tensor([ratio ** i for i in range(1, n_heads + 1)])

def alibi_attention(q, k, v, slopes):
    scores = q @ k.transpose(-2, -1)
    # Add distance-based bias
    positions = torch.arange(scores.size(-1))
    bias = -slopes.view(-1, 1, 1) * (positions.view(1, 1, -1) - positions.view(1, -1, 1)).abs()
    scores = scores + bias
    return F.softmax(scores, dim=-1) @ v`,
  },

  LEARNED: {
    name: 'Learned Position Embeddings',
    abbreviation: 'Learned',
    description:
      'Learnable embedding vectors for each position. Simple but limited to training length.',
    maxLength: 'Fixed to maximum trained length',
    extrapolation: 'None (cannot generalize beyond training length)',
    pros: [
      'Simple implementation',
      'Learnable patterns',
      'Well-understood',
    ],
    cons: [
      'Cannot extrapolate',
      'Extra parameters',
      'Fixed maximum length',
    ],
    examples: ['BERT', 'GPT-2', 'Original ViT'],
    codePattern: `
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1)]`,
  },

  SINUSOIDAL: {
    name: 'Sinusoidal Position Encoding',
    abbreviation: 'Sinusoidal',
    description:
      'Fixed sinusoidal patterns at different frequencies. Original transformer encoding.',
    maxLength: 'Theoretically infinite',
    extrapolation: 'Limited (not trained for extrapolation)',
    pros: [
      'No learned parameters',
      'Deterministic',
      'Can generate for any length',
    ],
    cons: [
      'Not as effective as learned alternatives',
      'Fixed patterns may be suboptimal',
    ],
    examples: ['Original Transformer', 'Some encoder models'],
    codePattern: `
def sinusoidal_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe`,
  },
};

/**
 * Activation function pattern definition.
 */
export interface ActivationPattern {
  name: string;
  formula: string;
  pros: string[];
  cons: string[];
  useCases: string[];
  computationalCost: string;
}

/**
 * Activation function patterns database.
 */
export const ACTIVATION_PATTERNS: Record<string, ActivationPattern> = {
  GELU: {
    name: 'Gaussian Error Linear Unit',
    formula: 'x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))',
    pros: [
      'Smooth approximation',
      'Non-zero gradient everywhere',
      'Standard for transformers',
    ],
    cons: ['Slower than ReLU', 'More computation'],
    useCases: ['Transformers', 'BERT-style models', 'GPT models'],
    computationalCost: 'Medium (~10 ops)',
  },

  SWIGLU: {
    name: 'SwiGLU (Swish-Gated Linear Unit)',
    formula: '(x * W1) * swish(x * W_gate)',
    pros: [
      'Better quality than GELU for LLMs',
      'Gating mechanism adds expressivity',
      'State-of-the-art for modern LLMs',
    ],
    cons: ['Requires 50% more parameters', 'More computation'],
    useCases: ['LLaMA', 'Modern LLMs', 'When quality > efficiency'],
    computationalCost: 'High (gating + swish)',
  },

  SILU: {
    name: 'SiLU/Swish',
    formula: 'x * sigmoid(x)',
    pros: [
      'Smooth',
      'Self-gated',
      'Good for many architectures',
    ],
    cons: ['Slower than ReLU'],
    useCases: ['EfficientNet', 'Transformers', 'Modern CNNs'],
    computationalCost: 'Medium (multiply + sigmoid)',
  },

  RELU: {
    name: 'Rectified Linear Unit',
    formula: 'max(0, x)',
    pros: [
      'Fastest computation',
      'Sparse activations',
      'Well-understood',
    ],
    cons: ['Dead neurons', 'Non-smooth at 0'],
    useCases: ['CNNs', 'When speed critical', 'ReLU-based architectures'],
    computationalCost: 'Low (single comparison)',
  },
};

/**
 * Helper function to get pattern by name.
 */
export function getAttentionPattern(name: string): AttentionPattern | undefined {
  return ATTENTION_PATTERNS[name.toUpperCase()];
}

/**
 * Helper function to get normalization pattern by name.
 */
export function getNormalizationPattern(name: string): NormalizationPattern | undefined {
  const key = name.toUpperCase().replace(/\s+/g, '_');
  return NORMALIZATION_PATTERNS[key];
}

/**
 * Helper function to get all pattern categories.
 */
export function getAllPatternCategories(): string[] {
  return [
    'attention',
    'normalization',
    'skip_connection',
    'efficient_conv',
    'transformer_variant',
    'position_encoding',
    'activation',
  ];
}

/**
 * Export all patterns as a combined object for easy access.
 */
export const ARCHITECTURE_PATTERNS = {
  attention: ATTENTION_PATTERNS,
  normalization: NORMALIZATION_PATTERNS,
  skipConnection: SKIP_CONNECTION_PATTERNS,
  efficientConv: EFFICIENT_CONV_PATTERNS,
  transformerVariant: TRANSFORMER_VARIANTS,
  positionEncoding: POSITION_ENCODING_PATTERNS,
  activation: ACTIVATION_PATTERNS,
};

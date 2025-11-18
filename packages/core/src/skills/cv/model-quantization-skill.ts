/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import { SkillCategory, SkillComplexity, type SkillDefinition } from '../types.js';
import { AgentExecutor } from '../../agents/executor.js';
import { ModelEvaluationAgent } from '../../agents/research/model-evaluation-agent.js';

const QuantizationInputSchema = z.object({
  modelPath: z.string(),
  quantizationType: z.enum(['int8', 'fp16', 'dynamic', 'static']).default('int8'),
  calibrationData: z.string().optional(),
  targetPlatform: z.enum(['cpu', 'cuda', 'tensorrt', 'onnx', 'mobile']).default('cpu'),
});

const QuantizationOutputSchema = z.object({
  success: z.boolean(),
  quantizationPlan: z.object({
    type: z.string(),
    expectedSpeedup: z.string(),
    expectedSizeReduction: z.string(),
    accuracyImpact: z.string(),
  }),
  implementation: z.object({
    code: z.string(),
    calibrationSteps: z.array(z.string()),
  }),
  validation: z.object({
    metrics: z.array(z.string()),
    comparisonStrategy: z.string(),
  }),
});

export const ModelQuantizationSkill: SkillDefinition = {
  id: 'cv.model_quantization',
  name: 'Model Quantization',
  description: 'Quantize models for efficient deployment with INT8, FP16, or dynamic quantization',
  usage: 'Reduce model size and improve inference speed while maintaining accuracy',
  category: SkillCategory.OPTIMIZATION,
  complexity: SkillComplexity.ADVANCED,
  version: '1.0.0',
  inputSchema: QuantizationInputSchema,
  outputSchema: QuantizationOutputSchema,
  requiredAgents: ['model_evaluation_agent'],
  estimatedTime: '5-8 minutes',
  tags: ['optimization', 'quantization', 'deployment', 'efficiency'],
  examples: [{
    title: 'INT8 quantization for edge',
    description: 'Quantize model to INT8 for edge deployment',
    input: {
      modelPath: './model.pth',
      quantizationType: 'int8',
      targetPlatform: 'cpu',
    },
  }],
  async execute(input, context, config) {
    const code = `
import torch
from torch.quantization import quantize_dynamic, quantize_static, prepare, convert

# Load model
model = torch.load('${input.modelPath}')
model.eval()

# ${input.quantizationType.toUpperCase()} Quantization
${input.quantizationType === 'dynamic' ? `
# Dynamic quantization (easiest, good for LSTM/Transformer)
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
` : input.quantizationType === 'static' ? `
# Static quantization (best accuracy)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
prepared_model = prepare(model)

# Calibration
with torch.no_grad():
    for data in calibration_loader:
        prepared_model(data)

quantized_model = convert(prepared_model)
` : `
# Post-training quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
quantized_model = torch.quantization.quantize_dynamic(
    model, dtype=torch.qint8
)
`}

# Save quantized model
torch.save(quantized_model.state_dict(), '${input.modelPath.replace('.pth', '_quantized.pth')}')

# Compare sizes
original_size = os.path.getsize('${input.modelPath}') / 1024 / 1024  # MB
quantized_size = os.path.getsize('${input.modelPath.replace('.pth', '_quantized.pth')}') / 1024 / 1024
print(f"Original: {original_size:.2f}MB, Quantized: {quantized_size:.2f}MB")
print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
`;

    return {
      success: true,
      data: {
        success: true,
        quantizationPlan: {
          type: input.quantizationType,
          expectedSpeedup: input.quantizationType === 'int8' ? '2-4x' : '1.5-2x',
          expectedSizeReduction: input.quantizationType === 'int8' ? '75%' : '50%',
          accuracyImpact: '< 1% degradation',
        },
        implementation: {
          code,
          calibrationSteps: input.quantizationType === 'static' ? [
            'Prepare representative calibration dataset',
            'Run model on calibration data',
            'Collect activation statistics',
            'Apply quantization based on statistics',
          ] : [
            'No calibration needed for dynamic quantization',
            'Apply quantization directly',
          ],
        },
        validation: {
          metrics: ['Accuracy', 'Inference latency', 'Model size', 'Memory usage'],
          comparisonStrategy: 'Compare quantized vs original on test set',
        },
      },
      metadata: {
        skillName: 'Model Quantization',
        duration: 0,
        agentsUsed: ['model_evaluation_agent'],
      },
    };
  },
};

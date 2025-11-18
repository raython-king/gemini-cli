/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import {
  SkillCategory,
  SkillComplexity,
  type SkillDefinition,
  type SkillContext,
  type SkillResult,
} from '../types.js';
import { AgentExecutor } from '../../agents/executor.js';
import { LLMResearchAgent } from '../../agents/research/llm-research-agent.js';
import { ExperimentAgent } from '../../agents/research/experiment-agent.js';

const FineTuningInputSchema = z.object({
  baseModel: z.string().describe('Base model to fine-tune (e.g., "llama-2-7b")'),
  dataset: z.string().describe('Path to fine-tuning dataset'),
  task: z.enum(['instruction_following', 'question_answering', 'summarization', 'translation', 'custom'])
    .describe('Fine-tuning task type'),
  trainingStrategy: z.enum(['full', 'lora', 'qlora', 'prefix_tuning']).default('lora')
    .describe('Training strategy'),
  hyperparameters: z.object({
    learningRate: z.number().optional(),
    batchSize: z.number().optional(),
    epochs: z.number().optional(),
    warmupSteps: z.number().optional(),
  }).optional().describe('Training hyperparameters'),
  evaluationMetric: z.string().optional().describe('Metric for evaluation'),
  computeBudget: z.string().optional().describe('Available compute budget'),
});

const FineTuningOutputSchema = z.object({
  success: z.boolean(),
  summary: z.string(),
  trainingPlan: z.object({
    strategy: z.string(),
    estimatedTime: z.string(),
    resourceRequirements: z.string(),
    steps: z.array(z.string()),
  }),
  hyperparameters: z.object({
    recommended: z.record(z.any()),
    rationale: z.string(),
  }),
  dataPreparation: z.object({
    steps: z.array(z.string()),
    formatExample: z.string(),
    validation: z.array(z.string()),
  }),
  implementationGuide: z.object({
    code: z.string(),
    dependencies: z.array(z.string()),
    tips: z.array(z.string()),
  }),
  evaluationPlan: z.object({
    metrics: z.array(z.string()),
    baselines: z.array(z.string()),
    testingStrategy: z.string(),
  }),
  artifacts: z.array(z.string()).optional(),
});

type FineTuningInput = z.infer<typeof FineTuningInputSchema>;
type FineTuningOutput = z.infer<typeof FineTuningOutputSchema>;

/**
 * LLM Fine-Tuning Skill
 *
 * Expert skill for fine-tuning large language models with various strategies.
 * Provides comprehensive guidance on data preparation, hyperparameter selection,
 * training execution, and evaluation.
 */
export const FineTuningSkill: SkillDefinition<FineTuningInput, FineTuningOutput> = {
  id: 'llm.fine_tuning',
  name: 'LLM Fine-Tuning',
  description: 'Fine-tune large language models with expert guidance on data prep, hyperparameters, and training strategies',
  usage: `
Fine-tune a language model for specific tasks:

Examples:
  - Fine-tune for instruction following
  - Adapt model to domain-specific data
  - Implement parameter-efficient fine-tuning (LoRA, QLoRA)
  - Optimize for specific evaluation metrics

Supports multiple training strategies and provides complete implementation guidance.
  `,
  category: SkillCategory.LLM,
  complexity: SkillComplexity.ADVANCED,
  version: '1.0.0',
  author: 'Gemini CLI',
  inputSchema: FineTuningInputSchema,
  outputSchema: FineTuningOutputSchema,
  requiredAgents: ['llm_research_agent', 'experiment_agent'],
  optionalAgents: ['data_flow_agent'],
  estimatedTime: '10-15 minutes',
  tags: ['llm', 'training', 'fine-tuning', 'lora', 'qlora', 'instruction-tuning'],

  examples: [
    {
      title: 'Fine-tune for instruction following',
      description: 'Fine-tune Llama 2 for instruction following using LoRA',
      input: {
        baseModel: 'llama-2-7b',
        dataset: './data/instructions.jsonl',
        task: 'instruction_following',
        trainingStrategy: 'lora',
        hyperparameters: {
          learningRate: 2e-4,
          batchSize: 4,
          epochs: 3,
        },
        evaluationMetric: 'accuracy',
      },
    },
    {
      title: 'Domain adaptation with QLoRA',
      description: 'Adapt model to medical domain using QLoRA for efficiency',
      input: {
        baseModel: 'llama-2-13b',
        dataset: './data/medical_qa.jsonl',
        task: 'question_answering',
        trainingStrategy: 'qlora',
        computeBudget: '1x A100 GPU, 24 hours',
      },
    },
  ],

  async execute(
    input: FineTuningInput,
    context: SkillContext,
    config: any,
  ): Promise<SkillResult<FineTuningOutput>> {
    const startTime = Date.now();
    const agentsUsed: string[] = [];

    try {
      // Step 1: Research fine-tuning strategy
      const llmResearchExecutor = await AgentExecutor.create(
        LLMResearchAgent,
        config,
      );
      agentsUsed.push('llm_research_agent');

      const researchResult = await llmResearchExecutor.run({
        objective: `Best practices for ${input.trainingStrategy} fine-tuning of ${input.baseModel} for ${input.task}`,
        topic: 'fine_tuning',
        depth: 'thorough',
        focus: ['training', 'optimization'],
        includeImplementation: true,
      });

      const researchData = JSON.parse(researchResult.result);

      // Step 2: Design fine-tuning experiment
      const experimentExecutor = await AgentExecutor.create(
        ExperimentAgent,
        config,
      );
      agentsUsed.push('experiment_agent');

      const experimentResult = await experimentExecutor.run({
        objective: `Design fine-tuning experiment for ${input.baseModel} on ${input.task} task`,
        algorithmType: 'llm',
        experimentType: 'baseline',
        codebase: context.workingDir,
        dataset: input.dataset,
        constraints: {
          maxComputeBudget: input.computeBudget,
        },
      });

      const experimentData = JSON.parse(experimentResult.result);

      // Step 3: Generate implementation code
      const loraCode = `
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# Load base model and tokenizer
model_name = "${input.baseModel}"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=${input.trainingStrategy === 'qlora'},
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load and prepare dataset
dataset = load_dataset("json", data_files="${input.dataset}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=${input.hyperparameters?.batchSize || 4},
    gradient_accumulation_steps=4,
    learning_rate=${input.hyperparameters?.learningRate || 2e-4},
    num_train_epochs=${input.hyperparameters?.epochs || 3},
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    warmup_steps=${input.hyperparameters?.warmupSteps || 100},
    fp16=True,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset.get("validation"),
    tokenizer=tokenizer,
    max_seq_length=2048,
)

# Train
trainer.train()

# Save model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
`;

      // Step 4: Prepare comprehensive output
      const output: FineTuningOutput = {
        success: true,
        summary: `Complete fine-tuning plan for ${input.baseModel} using ${input.trainingStrategy} strategy for ${input.task} task.`,

        trainingPlan: {
          strategy: input.trainingStrategy,
          estimatedTime: experimentData.ResourceRequirements?.estimatedTime || '4-8 hours',
          resourceRequirements: experimentData.ResourceRequirements?.computeResources || '1x A100 GPU, 40GB VRAM',
          steps: [
            'Prepare and validate dataset',
            'Configure model and LoRA parameters',
            'Set up training arguments',
            'Execute training with monitoring',
            'Evaluate on validation set',
            'Save fine-tuned model',
          ],
        },

        hyperparameters: {
          recommended: {
            learning_rate: input.hyperparameters?.learningRate || 2e-4,
            batch_size: input.hyperparameters?.batchSize || 4,
            epochs: input.hyperparameters?.epochs || 3,
            warmup_steps: input.hyperparameters?.warmupSteps || 100,
            lora_r: 16,
            lora_alpha: 32,
            lora_dropout: 0.05,
            gradient_accumulation_steps: 4,
          },
          rationale: experimentData.TrainingProcedure?.optimizer ||
            'Recommended hyperparameters based on model size and task complexity. ' +
            'LoRA rank 16 provides good balance between efficiency and performance.',
        },

        dataPreparation: {
          steps: experimentData.DataPreparation?.preprocessing || [
            'Load dataset in JSONL format',
            'Tokenize with model tokenizer',
            'Format prompts for instruction tuning',
            'Split into train/validation/test sets',
            'Verify data quality and balance',
          ],
          formatExample: `{
  "instruction": "Translate to French: Hello, how are you?",
  "input": "",
  "output": "Bonjour, comment allez-vous?"
}`,
          validation: [
            'Check for duplicate examples',
            'Verify token length distribution',
            'Ensure balanced task distribution',
            'Validate prompt formatting',
          ],
        },

        implementationGuide: {
          code: loraCode,
          dependencies: [
            'transformers>=4.35.0',
            'peft>=0.6.0',
            'trl>=0.7.0',
            'datasets>=2.14.0',
            'accelerate>=0.24.0',
            'bitsandbytes>=0.41.0',
            'torch>=2.0.0',
          ],
          tips: researchData.ImplementationInsights?.bestPractices || [
            'Monitor training loss and validation metrics closely',
            'Use gradient checkpointing for larger models',
            'Save checkpoints frequently',
            'Consider early stopping based on validation loss',
            'Test with small subset first to verify setup',
            'Use mixed precision training (fp16) for efficiency',
          ],
        },

        evaluationPlan: {
          metrics: experimentData.EvaluationMetrics?.map((m: any) => m.metric) || [
            'Perplexity',
            'Task-specific accuracy',
            'BLEU/ROUGE (if applicable)',
            'Human evaluation',
          ],
          baselines: experimentData.Baselines?.map((b: any) => b.name) || [
            'Base model (no fine-tuning)',
            'Few-shot prompting baseline',
          ],
          testingStrategy: 'Evaluate on held-out test set with multiple runs for statistical significance',
        },

        artifacts: [
          './fine_tuning_plan.md',
          './training_script.py',
          './data_preparation.py',
          './evaluation_script.py',
        ],
      };

      return {
        success: true,
        data: output,
        metadata: {
          skillName: 'LLM Fine-Tuning',
          duration: Date.now() - startTime,
          agentsUsed,
        },
        artifacts: output.artifacts,
        suggestions: [
          'Run the data preparation script first to validate your dataset',
          'Start with a small subset to test the training pipeline',
          'Consider using Weights & Biases or TensorBoard for monitoring',
          'After fine-tuning, compare with base model on your evaluation set',
        ],
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        metadata: {
          skillName: 'LLM Fine-Tuning',
          duration: Date.now() - startTime,
          agentsUsed,
        },
      };
    }
  },
};

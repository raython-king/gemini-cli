<!-- markdownlint-disable MD013 MD033 -->
# Expert Skills System

## Overview

Gemini CLI now features a **Claude Code-like Skills System** that provides reusable, expert-level capabilities for LLM and CV development. Skills are pre-packaged workflows that combine multiple agents to solve specific tasks efficiently.

## 🎯 What are Skills?

Skills are **expert-level, reusable capabilities** that:
- Combine multiple agents intelligently
- Provide complete, end-to-end solutions
- Include best practices and implementation code
- Are categorized by domain (LLM, CV, Optimization, etc.)
- Can be chained together for complex workflows

Think of skills as **expert consultants** you can invoke instantly.

## 🚀 Quick Start

### List All Skills

```bash
gemini skill list
```

### Get Skill Details

```bash
gemini skill info llm.fine_tuning
```

### Run a Skill

```bash
# Interactive mode
gemini skill run llm.fine_tuning --interactive

# With JSON input
gemini skill run llm.fine_tuning --input '{"baseModel":"llama-2-7b","dataset":"./data.jsonl","task":"instruction_following"}'

# From file
gemini skill run llm.fine_tuning --input-file ./config.json
```

### Search Skills

```bash
# By category
gemini skill search --category llm

# By tag
gemini skill search --tag optimization

# Text search
gemini skill search --query "quantization"
```

## 📚 Available Skills

### LLM Skills

#### 1. **LLM Fine-Tuning** (`llm.fine_tuning`)

**What it does**: Complete fine-tuning setup for LLMs with LoRA, QLoRA, or full fine-tuning.

**Complexity**: Advanced

**Outputs**:
- Training plan with resource requirements
- Optimized hyperparameters with rationale
- Data preparation pipeline
- Complete implementation code (Python)
- Evaluation strategy
- Monitoring setup

**Example**:
```bash
gemini skill run llm.fine_tuning --input '{
  "baseModel": "llama-2-7b",
  "dataset": "./instructions.jsonl",
  "task": "instruction_following",
  "trainingStrategy": "lora",
  "hyperparameters": {
    "learningRate": 2e-4,
    "batchSize": 4,
    "epochs": 3
  }
}'
```

**Use Cases**:
- Instruction tuning
- Domain adaptation (medical, legal, technical)
- Task-specific optimization
- Few-shot to fine-tuned conversion

#### 2. **Prompt Optimization** (`llm.prompt_optimization`)

**What it does**: Optimize prompts using expert techniques like chain-of-thought, few-shot, and role-based prompting.

**Complexity**: Intermediate

**Outputs**:
- Multiple optimized prompt variations
- Technique explanations
- Expected performance improvements
- Testing strategy

**Example**:
```bash
gemini skill run llm.prompt_optimization --input '{
  "task": "Code generation",
  "currentPrompt": "Write a sorting function",
  "model": "gpt-4",
  "examples": [
    {"input": "Sort integers", "expectedOutput": "def sort(arr): ..."}
  ]
}'
```

**Techniques Applied**:
- Role definition
- Chain-of-thought prompting
- Few-shot examples
- Output formatting
- Constraint specification

### CV Skills

#### 3. **Object Detection Setup** (`cv.object_detection`)

**What it does**: Complete object detection pipeline setup with YOLOv8, Faster R-CNN, or DETR.

**Complexity**: Advanced

**Outputs**:
- Architecture configuration
- Data preparation pipeline with augmentations
- Training script
- Deployment code (edge/cloud)
- Evaluation metrics

**Example**:
```bash
gemini skill run cv.object_detection --input '{
  "dataset": "./custom_data",
  "classes": ["person", "car", "bicycle"],
  "architecture": "yolov8",
  "imageSize": 640,
  "deploymentTarget": "edge"
}'
```

**Supported Architectures**:
- YOLOv8 (fastest, best for real-time)
- Faster R-CNN (high accuracy)
- DETR (transformer-based)
- RetinaNet (balanced)

#### 4. **Model Quantization** (`cv.model_quantization`)

**What it does**: Quantize models for efficient deployment (INT8, FP16, dynamic).

**Complexity**: Advanced

**Outputs**:
- Quantization strategy
- Expected speedup and size reduction
- Implementation code
- Calibration procedure
- Validation metrics

**Example**:
```bash
gemini skill run cv.model_quantization --input '{
  "modelPath": "./model.pth",
  "quantizationType": "int8",
  "targetPlatform": "cpu",
  "calibrationData": "./calib_data"
}'
```

**Quantization Types**:
- **INT8**: 75% size reduction, 2-4x speedup
- **FP16**: 50% size reduction, 1.5-2x speedup
- **Dynamic**: No calibration needed
- **Static**: Best accuracy, needs calibration

## 🔧 Skill System Architecture

### Skill Definition

```typescript
interface SkillDefinition {
  id: string;                    // Unique identifier
  name: string;                  // Display name
  description: string;           // What it does
  category: SkillCategory;       // llm, cv, optimization, etc.
  complexity: SkillComplexity;   // basic, intermediate, advanced, expert
  version: string;               // Version
  inputSchema: ZodSchema;        // Input validation
  outputSchema: ZodSchema;       // Output structure
  requiredAgents: string[];      // Agents it uses
  estimatedTime: string;         // How long it takes
  tags: string[];                // For searchability
  examples: Example[];           // Usage examples
  execute: (input, context, config) => Promise<SkillResult>;
}
```

### Skill Categories

- **LLM**: Language model skills
- **CV**: Computer vision skills
- **EXPERIMENT**: Experiment design and execution
- **OPTIMIZATION**: Performance optimization
- **ANALYSIS**: Data and model analysis
- **DEPLOYMENT**: Production deployment
- **GENERAL**: General-purpose utilities

### Complexity Levels

- **Basic**: Simple, single-agent tasks (5 min)
- **Intermediate**: Multi-step workflows (5-10 min)
- **Advanced**: Complex, multi-agent coordination (10-20 min)
- **Expert**: Comprehensive, research-grade workflows (20+ min)

## 💡 Using Skills Programmatically

### Basic Usage

```typescript
import { SkillManager, BUILT_IN_SKILLS } from '@google/gemini-cli-core';

// Initialize
const skillManager = new SkillManager();
skillManager.registerSkills(BUILT_IN_SKILLS);

// Execute a skill
const result = await skillManager.executeSkill(
  'llm.fine_tuning',
  {
    baseModel: 'llama-2-7b',
    dataset: './data.jsonl',
    task: 'instruction_following',
    trainingStrategy: 'lora',
  },
  context,
  config,
  {
    verbose: true,
    onProgress: (progress) => {
      console.log(`${progress.stage}: ${progress.message}`);
    },
  },
);

if (result.success) {
  console.log('Fine-tuning plan:', result.data);
  console.log('Artifacts:', result.artifacts);
  console.log('Suggestions:', result.suggestions);
}
```

### Skill Chaining

```typescript
// Define a skill chain
const chain = {
  id: 'llm_deployment_chain',
  name: 'LLM Deployment Pipeline',
  description: 'Fine-tune, optimize, and deploy an LLM',
  skills: [
    {
      skillId: 'llm.fine_tuning',
      inputs: { /* ... */ },
      outputMapping: {
        'trainingPlan': 'plan',
      },
    },
    {
      skillId: 'cv.model_quantization',
      inputs: { /* ... */ },
      outputMapping: {
        'quantizationPlan': 'deployment',
      },
    },
  ],
};

// Register and execute chain
skillManager.registerChain(chain);
const results = await skillManager.executeChain(
  'llm_deployment_chain',
  context,
  config,
);
```

### Custom Skills

```typescript
import { SkillCategory, SkillComplexity } from '@google/gemini-cli-core';
import { z } from 'zod';

const MyCustomSkill = {
  id: 'custom.my_skill',
  name: 'My Custom Skill',
  description: 'Does something amazing',
  category: SkillCategory.GENERAL,
  complexity: SkillComplexity.INTERMEDIATE,
  version: '1.0.0',
  inputSchema: z.object({
    param1: z.string(),
    param2: z.number().optional(),
  }),
  outputSchema: z.object({
    result: z.string(),
  }),
  requiredAgents: ['explore_agent'],
  estimatedTime: '5 minutes',
  tags: ['custom', 'utility'],
  examples: [
    {
      title: 'Example usage',
      description: 'How to use this skill',
      input: { param1: 'value' },
    },
  ],
  async execute(input, context, config) {
    // Your implementation
    return {
      success: true,
      data: { result: 'done' },
      metadata: {
        skillName: 'My Custom Skill',
        duration: 1000,
        agentsUsed: ['explore_agent'],
      },
    };
  },
};

// Register custom skill
skillManager.registerSkill(MyCustomSkill);
```

## 🎓 Advanced Features

### Progress Monitoring

```typescript
const result = await skillManager.executeSkill(
  'llm.fine_tuning',
  input,
  context,
  config,
  {
    verbose: true,
    onProgress: (progress) => {
      console.log(`[${progress.stage}] ${progress.message} - ${progress.progress}%`);
      if (progress.currentAgent) {
        console.log(`  Using agent: ${progress.currentAgent}`);
      }
    },
  },
);
```

### Dry Run

```typescript
// Validate inputs without executing
const result = await skillManager.executeSkill(
  'llm.fine_tuning',
  input,
  context,
  config,
  { dryRun: true },
);

console.log('Validation:', result.success);
console.log('Would use agents:', result.metadata.agentsUsed);
```

### Timeouts

```typescript
// Set execution timeout
const result = await skillManager.executeSkill(
  'llm.fine_tuning',
  input,
  context,
  config,
  { timeout: 300000 }, // 5 minutes
);
```

### Search and Discovery

```typescript
// Search by category
const llmSkills = skillManager.searchSkills({
  category: SkillCategory.LLM,
});

// Search by complexity
const expertSkills = skillManager.searchSkills({
  complexity: SkillComplexity.EXPERT,
});

// Search by tags
const optimizationSkills = skillManager.searchSkills({
  tags: ['optimization', 'efficiency'],
});

// Text search
const results = skillManager.searchSkills({
  query: 'quantization',
});

// Combined filters
const filtered = skillManager.searchSkills({
  category: SkillCategory.CV,
  complexity: SkillComplexity.ADVANCED,
  tags: ['deployment'],
});
```

### Statistics

```typescript
const stats = skillManager.getStats();

console.log('Total skills:', stats.totalSkills);
console.log('Total chains:', stats.totalChains);
console.log('By category:', stats.byCategory);
console.log('By complexity:', stats.byComplexity);
```

## 📖 Real-World Examples

### Example 1: Fine-Tune Llama 2 for Medical QA

```bash
gemini skill run llm.fine_tuning --input '{
  "baseModel": "llama-2-7b",
  "dataset": "./medical_qa.jsonl",
  "task": "question_answering",
  "trainingStrategy": "qlora",
  "hyperparameters": {
    "learningRate": 2e-4,
    "batchSize": 4,
    "epochs": 3
  },
  "evaluationMetric": "f1_score",
  "computeBudget": "1x A100, 24 hours"
}'
```

**Output**:
- Complete QLoRA training setup
- Medical domain-specific data preprocessing
- Evaluation on medical benchmarks
- Inference optimization for clinical use

### Example 2: Optimize YOLOv8 for Raspberry Pi

```bash
gemini skill run cv.object_detection --input '{
  "dataset": "./custom_objects",
  "classes": ["product_a", "product_b", "defect"],
  "architecture": "yolov8",
  "imageSize": 416,
  "deploymentTarget": "edge"
}'
```

Then quantize:

```bash
gemini skill run cv.model_quantization --input '{
  "modelPath": "./runs/detect/weights/best.pt",
  "quantizationType": "int8",
  "targetPlatform": "cpu"
}'
```

**Result**:
- Edge-optimized YOLOv8 model
- 4x smaller, 3x faster
- < 1% accuracy loss
- Ready for Raspberry Pi deployment

### Example 3: Prompt Engineering Chain

```bash
# Step 1: Optimize prompt
gemini skill run llm.prompt_optimization --input '{
  "task": "SQL generation from natural language",
  "currentPrompt": "Generate SQL query",
  "model": "gpt-4"
}'

# Step 2: Use optimized prompt in fine-tuning
# (Use output from step 1 as training data)
```

## 🔍 Skill Details Reference

### Fine-Tuning Skill

**Input Parameters**:
```typescript
{
  baseModel: string;                    // Model to fine-tune
  dataset: string;                      // Path to training data
  task: 'instruction_following' | 'question_answering' | 'summarization' | 'translation' | 'custom';
  trainingStrategy: 'full' | 'lora' | 'qlora' | 'prefix_tuning';
  hyperparameters?: {
    learningRate?: number;
    batchSize?: number;
    epochs?: number;
    warmupSteps?: number;
  };
  evaluationMetric?: string;
  computeBudget?: string;
}
```

**Output Structure**:
```typescript
{
  success: boolean;
  summary: string;
  trainingPlan: {
    strategy: string;
    estimatedTime: string;
    resourceRequirements: string;
    steps: string[];
  };
  hyperparameters: {
    recommended: Record<string, any>;
    rationale: string;
  };
  dataPreparation: {
    steps: string[];
    formatExample: string;
    validation: string[];
  };
  implementationGuide: {
    code: string;
    dependencies: string[];
    tips: string[];
  };
  evaluationPlan: {
    metrics: string[];
    baselines: string[];
    testingStrategy: string;
  };
}
```

### Object Detection Skill

**Input Parameters**:
```typescript
{
  dataset: string;
  classes: string[];
  architecture: 'yolov8' | 'faster_rcnn' | 'detr' | 'retinanet';
  imageSize?: number;  // default: 640
  deploymentTarget?: 'cloud' | 'edge' | 'mobile';
}
```

**Output Structure**:
```typescript
{
  success: boolean;
  setupGuide: {
    architecture: string;
    configuration: Record<string, any>;
    trainingCommand: string;
  };
  dataPreparation: {
    format: string;
    augmentations: string[];
    splits: { train: number; val: number; test: number };
  };
  implementation: {
    code: string;
    dependencies: string[];
  };
}
```

## 🚧 Upcoming Skills

Planned skills for future releases:

### LLM Skills
- [ ] `llm.rlhf_training` - RLHF/DPO training setup
- [ ] `llm.model_merging` - Merge multiple fine-tuned models
- [ ] `llm.inference_optimization` - Optimize for serving
- [ ] `llm.prompt_injection_defense` - Security hardening
- [ ] `llm.hallucination_detection` - Detect and mitigate hallucinations

### CV Skills
- [ ] `cv.image_segmentation` - Semantic/instance segmentation
- [ ] `cv.transfer_learning` - Transfer learning setup
- [ ] `cv.data_augmentation` - Advanced augmentation pipeline
- [ ] `cv.model_distillation` - Knowledge distillation
- [ ] `cv.anomaly_detection` - Anomaly detection setup

### General Skills
- [ ] `experiment.ablation_study` - Automated ablation studies
- [ ] `deployment.api_server` - Create inference API
- [ ] `deployment.monitoring` - Setup monitoring and logging
- [ ] `optimization.hyperparameter_tuning` - Auto hyperparameter search
- [ ] `analysis.model_comparison` - Compare multiple models

## 💼 Best Practices

### 1. Use Skills for Repetitive Tasks

Instead of manually setting up fine-tuning every time, use the skill:

```bash
# ❌ Manual setup (hours of work)
# - Research best practices
# - Configure LoRA
# - Set hyperparameters
# - Write training code
# - Setup evaluation

# ✅ Use skill (minutes)
gemini skill run llm.fine_tuning --interactive
```

### 2. Chain Skills for Complex Workflows

```bash
# Research → Implement → Optimize → Deploy
gemini skill run llm.fine_tuning ...
# Use output for next step
gemini skill run cv.model_quantization ...
```

### 3. Customize and Extend

Start with built-in skills, then create custom ones for your specific needs:

```typescript
// Extend existing skill
const MyFineTuningSkill = {
  ...FineTuningSkill,
  id: 'custom.my_fine_tuning',
  execute: async (input, context, config) => {
    // Add custom logic
    const baseResult = await FineTuningSkill.execute(input, context, config);
    // Enhance result
    return enhancedResult;
  },
};
```

### 4. Use Dry Run for Validation

```bash
# Test inputs before execution
gemini skill run llm.fine_tuning --input-file ./config.json --dry-run
```

### 5. Save and Share Configurations

```bash
# Save successful configuration
echo '{
  "baseModel": "llama-2-7b",
  "dataset": "./data.jsonl",
  "trainingStrategy": "lora"
}' > llm-config.json

# Reuse later
gemini skill run llm.fine_tuning --input-file llm-config.json
```

## 📊 Performance

Skill execution times (approximate):

| Skill | Complexity | Time | Agents Used |
|-------|-----------|------|-------------|
| Prompt Optimization | Intermediate | 3-5 min | 1 |
| Fine-Tuning | Advanced | 10-15 min | 2 |
| Object Detection | Advanced | 10-15 min | 2 |
| Model Quantization | Advanced | 5-8 min | 1 |

## 🤝 Contributing

To add a new skill:

1. Create skill definition file
2. Implement input/output schemas
3. Write execute function
4. Add examples and documentation
5. Register in `BUILT_IN_SKILLS`
6. Add tests

See existing skills for examples.

## License

Copyright 2025 Google LLC. Licensed under Apache-2.0.

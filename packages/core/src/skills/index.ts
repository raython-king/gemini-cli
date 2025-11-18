/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Export types
export * from './types.js';

// Export skill manager
export { SkillManager } from './skill-manager.js';

// Export LLM skills
export { FineTuningSkill } from './llm/fine-tuning-skill.js';
export { PromptOptimizationSkill } from './llm/prompt-optimization-skill.js';

// Export CV skills
export { ObjectDetectionSkill } from './cv/object-detection-skill.js';
export { ModelQuantizationSkill } from './cv/model-quantization-skill.js';

// Built-in skills registry
import { FineTuningSkill } from './llm/fine-tuning-skill.js';
import { PromptOptimizationSkill } from './llm/prompt-optimization-skill.js';
import { ObjectDetectionSkill } from './cv/object-detection-skill.js';
import { ModelQuantizationSkill } from './cv/model-quantization-skill.js';

export const BUILT_IN_SKILLS = [
  FineTuningSkill,
  PromptOptimizationSkill,
  ObjectDetectionSkill,
  ModelQuantizationSkill,
];

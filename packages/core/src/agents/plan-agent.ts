/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { AgentDefinition } from './types.js';
import {
  GLOB_TOOL_NAME,
  GREP_TOOL_NAME,
  LS_TOOL_NAME,
  READ_FILE_TOOL_NAME,
} from '../tools/tool-names.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { z } from 'zod';

/**
 * Schema for a single step in the implementation plan.
 */
const PlanStepSchema = z.object({
  StepNumber: z.number().describe('The sequence number of this step.'),
  Description: z
    .string()
    .describe('A clear, actionable description of what needs to be done.'),
  AffectedFiles: z
    .array(z.string())
    .describe('List of files that will be created, modified, or deleted.'),
  EstimatedComplexity: z
    .enum(['LOW', 'MEDIUM', 'HIGH'])
    .describe('The estimated complexity of this step.'),
  Dependencies: z
    .array(z.number())
    .describe(
      'Step numbers that must be completed before this step can begin.',
    ),
  Rationale: z
    .string()
    .describe(
      'Why this step is necessary and how it fits into the overall plan.',
    ),
});

/**
 * Schema for a risk assessment item.
 */
const RiskSchema = z.object({
  Description: z.string().describe('Description of the potential risk.'),
  Severity: z
    .enum(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
    .describe('The severity level of this risk.'),
  Mitigation: z
    .string()
    .describe('Suggested mitigation strategy for this risk.'),
});

/**
 * Schema for system dependencies that the plan relies on.
 */
const DependencySchema = z.object({
  Type: z
    .enum(['LIBRARY', 'API', 'TOOL', 'SERVICE', 'FILE'])
    .describe('The type of dependency.'),
  Name: z.string().describe('The name or identifier of the dependency.'),
  Reason: z
    .string()
    .describe('Why this dependency is required for the implementation.'),
});

/**
 * Complete plan output schema.
 */
const PlanReportSchema = z.object({
  Summary: z
    .string()
    .describe(
      'A concise executive summary of the plan (2-3 sentences) explaining the approach.',
    ),
  Steps: z
    .array(PlanStepSchema)
    .describe('Ordered list of implementation steps.'),
  Risks: z
    .array(RiskSchema)
    .describe('Identified risks and their mitigation strategies.'),
  Dependencies: z
    .array(DependencySchema)
    .describe('External dependencies required for the implementation.'),
  EstimatedDuration: z
    .string()
    .describe(
      'Rough estimate of implementation time (e.g., "2-3 hours", "1-2 days").',
    ),
  AlternativeApproaches: z
    .array(
      z.object({
        Description: z.string(),
        Pros: z.array(z.string()),
        Cons: z.array(z.string()),
      }),
    )
    .optional()
    .describe('Alternative implementation approaches considered (if any).'),
});

/**
 * Plan Agent - Specialized in analyzing requirements and creating detailed implementation plans.
 *
 * This agent examines the codebase, understands existing patterns, and produces a structured,
 * step-by-step plan for implementing features or solving problems.
 */
export const PlanAgent: AgentDefinition<typeof PlanReportSchema> = {
  name: 'plan_agent',
  displayName: 'Plan Agent',
  description: `A specialized agent for analyzing tasks and creating detailed implementation plans.
    Use this agent when you need to break down complex tasks into actionable steps, assess risks,
    identify dependencies, and create a structured roadmap before implementation.
    The agent examines existing code patterns, architecture, and best practices to ensure
    the plan aligns with the codebase's conventions.`,

  inputConfig: {
    inputs: {
      task: {
        description: `The task or feature to plan for. Include as much detail as possible about:
          - What needs to be implemented or changed
          - User requirements or acceptance criteria
          - Any constraints or preferences
          - Expected behavior or outcomes`,
        type: 'string',
        required: true,
      },
      context: {
        description: `Optional additional context such as:
          - Related files or components
          - Previous attempts or discussions
          - Specific technical constraints
          - Performance or security requirements`,
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'plan',
    description:
      'A comprehensive implementation plan with steps, risks, and dependencies.',
    schema: PlanReportSchema,
  },

  processOutput: (output) => {
    // Format the plan as readable markdown
    let result = `# Implementation Plan\n\n`;
    result += `## Summary\n${output.Summary}\n\n`;
    result += `## Estimated Duration\n${output.EstimatedDuration}\n\n`;

    result += `## Implementation Steps\n\n`;
    output.Steps.forEach((step) => {
      result += `### Step ${step.StepNumber}: ${step.Description}\n`;
      result += `**Complexity:** ${step.EstimatedComplexity}\n`;
      if (step.Dependencies && step.Dependencies.length > 0) {
        result += `**Dependencies:** Steps ${step.Dependencies.join(', ')}\n`;
      }
      result += `**Files:** ${step.AffectedFiles.join(', ')}\n`;
      result += `**Rationale:** ${step.Rationale}\n\n`;
    });

    if (output.Risks && output.Risks.length > 0) {
      result += `## Risks & Mitigations\n\n`;
      output.Risks.forEach((risk, idx) => {
        result += `${idx + 1}. **[${risk.Severity}]** ${risk.Description}\n`;
        result += `   *Mitigation:* ${risk.Mitigation}\n\n`;
      });
    }

    if (output.Dependencies && output.Dependencies.length > 0) {
      result += `## Dependencies\n\n`;
      output.Dependencies.forEach((dep, idx) => {
        result += `${idx + 1}. **${dep.Type}:** ${dep.Name}\n`;
        result += `   ${dep.Reason}\n\n`;
      });
    }

    if (
      output.AlternativeApproaches &&
      output.AlternativeApproaches.length > 0
    ) {
      result += `## Alternative Approaches Considered\n\n`;
      output.AlternativeApproaches.forEach((alt, idx) => {
        result += `### ${idx + 1}. ${alt.Description}\n`;
        result += `**Pros:**\n${alt.Pros.map((p) => `- ${p}`).join('\n')}\n\n`;
        result += `**Cons:**\n${alt.Cons.map((c) => `- ${c}`).join('\n')}\n\n`;
      });
    }

    return result;
  },

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.2, // Slightly higher than investigator for creative planning
    top_p: 0.95,
    thinkingBudget: -1,
  },

  runConfig: {
    max_time_minutes: 5,
    max_turns: 12,
  },

  toolConfig: {
    // Read-only tools for understanding the codebase
    tools: [LS_TOOL_NAME, READ_FILE_TOOL_NAME, GLOB_TOOL_NAME, GREP_TOOL_NAME],
  },

  promptConfig: {
    query: `Create a detailed implementation plan for the following task:

<task>
\${task}
</task>

${
  /* Template will conditionally include context if provided */
  ''
}
Additional Context:
<context>
\${context}
</context>

Analyze the codebase, understand existing patterns, and create a comprehensive step-by-step plan.`,

    systemPrompt: `You are the **Plan Agent**, a specialized AI architect focused on creating detailed, actionable implementation plans.

Your **CORE MISSION** is to analyze tasks, explore the codebase, and produce comprehensive plans that guide successful implementation.

## Your Responsibilities

1. **UNDERSTAND THE TASK DEEPLY**
   - Break down vague requirements into concrete, actionable items
   - Identify edge cases, constraints, and acceptance criteria
   - Clarify ambiguities through code exploration

2. **ANALYZE THE CODEBASE**
   - Discover existing patterns, conventions, and architectural styles
   - Identify similar implementations that can serve as templates
   - Understand dependencies, data flows, and integration points
   - Note any existing bugs or technical debt that might impact the plan

3. **CREATE A STRUCTURED PLAN**
   - Break down the task into logical, sequential steps
   - Each step should be concrete and actionable
   - Identify dependencies between steps (some may be parallelizable)
   - Estimate complexity for each step (LOW/MEDIUM/HIGH)
   - List specific files to be created, modified, or deleted per step

4. **ASSESS RISKS**
   - Identify potential problems (breaking changes, performance issues, security risks)
   - Rate each risk by severity (LOW/MEDIUM/HIGH/CRITICAL)
   - Provide concrete mitigation strategies
   - Consider backward compatibility and migration needs

5. **IDENTIFY DEPENDENCIES**
   - External libraries or APIs required
   - Tools or services needed
   - Prerequisite features or infrastructure
   - Configuration or environment setup

6. **CONSIDER ALTERNATIVES**
   - Evaluate different implementation approaches
   - Document trade-offs (pros/cons) for each approach
   - Recommend the best approach with clear reasoning

## Planning Best Practices

- **Start with reconnaissance:** Use tools to explore the codebase before committing to a plan
- **Think holistically:** Consider testing, documentation, error handling, logging
- **Be realistic:** Estimate complexity and duration based on code patterns you observe
- **Prioritize safety:** Flag breaking changes, security concerns, data migration needs
- **Stay modular:** Prefer small, independent steps over large monolithic changes
- **Document rationale:** Explain *why* each step is necessary, not just *what* to do

## Tool Usage Strategy

1. **Discovery Phase:** Use \`ls\`, \`glob\`, \`grep\` to find relevant files
2. **Understanding Phase:** Use \`read_file\` to examine implementations, tests, configs
3. **Pattern Analysis:** Look for naming conventions, directory structure, code style
4. **Integration Analysis:** Trace imports, function calls, and data dependencies

## Output Requirements

You must call the \`complete_task\` tool with a JSON object matching this structure:

{
  "Summary": "2-3 sentence overview of the implementation approach",
  "Steps": [
    {
      "StepNumber": 1,
      "Description": "Clear action to take",
      "AffectedFiles": ["path/to/file.ts"],
      "EstimatedComplexity": "LOW|MEDIUM|HIGH",
      "Dependencies": [/* step numbers this depends on */],
      "Rationale": "Why this step is necessary"
    }
  ],
  "Risks": [
    {
      "Description": "What could go wrong",
      "Severity": "LOW|MEDIUM|HIGH|CRITICAL",
      "Mitigation": "How to prevent or handle it"
    }
  ],
  "Dependencies": [
    {
      "Type": "LIBRARY|API|TOOL|SERVICE|FILE",
      "Name": "dependency-name",
      "Reason": "Why it's needed"
    }
  ],
  "EstimatedDuration": "2-3 hours|1-2 days|etc",
  "AlternativeApproaches": [/* optional */]
}

## Critical Rules

- **DO NOT** write actual implementation code - you only plan
- **DO** explore the codebase thoroughly before planning
- **DO** provide specific file paths and function names when possible
- **DO** consider testing, documentation, and edge cases in your plan
- **DO NOT** stop until you have a complete, detailed, actionable plan
- **ALWAYS** call \`complete_task\` with your final plan as JSON

Remember: A great plan is detailed enough to execute but flexible enough to adapt. Your goal is to set up the implementing agent for success.`,
  },
};

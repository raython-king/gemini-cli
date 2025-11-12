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
  WEB_SEARCH_TOOL_NAME,
} from '../tools/tool-names.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { z } from 'zod';

/**
 * Schema for a discovered code location.
 */
const CodeLocationSchema = z.object({
  FilePath: z.string().describe('The path to the relevant file.'),
  LineRange: z
    .string()
    .optional()
    .describe('Optional line range (e.g., "45-67") for precise location.'),
  Symbols: z
    .array(z.string())
    .describe('Key functions, classes, or variables in this location.'),
  Summary: z.string().describe('Brief description of what this code does.'),
});

/**
 * Schema for the exploration report.
 */
const ExploreReportSchema = z.object({
  Answer: z
    .string()
    .describe(
      'Direct answer to the exploration query. Be concise but complete.',
    ),
  KeyFindings: z
    .array(z.string())
    .describe('List of 3-5 key insights or discoveries from the exploration.'),
  RelevantLocations: z
    .array(CodeLocationSchema)
    .describe('Code locations relevant to the query.'),
  ArchitectureInsights: z
    .string()
    .optional()
    .describe(
      'High-level architectural observations (patterns, design principles, etc.).',
    ),
  Recommendations: z
    .array(z.string())
    .optional()
    .describe(
      'Suggestions for next steps, improvements, or things to investigate further.',
    ),
});

/**
 * Explore Agent - Fast codebase exploration and question answering.
 *
 * This agent is optimized for speed and efficiency. It quickly explores the codebase
 * to answer specific questions, find relevant code, or understand system architecture.
 * Use this instead of CodebaseInvestigator when you need faster, more targeted results.
 */
export const ExploreAgent: AgentDefinition<typeof ExploreReportSchema> = {
  name: 'explore_agent',
  displayName: 'Explore Agent',
  description: `A fast, efficient agent for exploring codebases and answering questions about code structure.
    Use this agent when you need to:
    - Quickly find where a feature is implemented
    - Understand how components interact
    - Locate specific functionality or patterns
    - Get architectural insights
    - Answer "where" and "how" questions about the codebase

    This agent is optimized for speed and provides targeted, actionable results.`,

  inputConfig: {
    inputs: {
      query: {
        description: `The exploration goal or question. Examples:
          - "Where is user authentication implemented?"
          - "How does the payment system work?"
          - "Find all API endpoints related to orders"
          - "What testing framework is used and where are the tests?"`,
        type: 'string',
        required: true,
      },
      thoroughness: {
        description: `Exploration thoroughness level:
          - "quick": Fast scan, 3-5 turns, surface-level insights
          - "medium": Balanced exploration, 5-8 turns, good depth (default)
          - "thorough": Deep investigation, 8-12 turns, comprehensive analysis`,
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description:
      'Exploration report with findings and relevant code locations.',
    schema: ExploreReportSchema,
  },

  processOutput: (output) => {
    let result = `# Exploration Results\n\n`;
    result += `## Answer\n${output.Answer}\n\n`;

    if (output.KeyFindings && output.KeyFindings.length > 0) {
      result += `## Key Findings\n`;
      output.KeyFindings.forEach((finding, idx) => {
        result += `${idx + 1}. ${finding}\n`;
      });
      result += `\n`;
    }

    if (output.RelevantLocations && output.RelevantLocations.length > 0) {
      result += `## Relevant Code Locations\n\n`;
      output.RelevantLocations.forEach((loc, idx) => {
        result += `### ${idx + 1}. ${loc.FilePath}`;
        if (loc.LineRange) {
          result += ` (lines ${loc.LineRange})`;
        }
        result += `\n`;
        result += `**Symbols:** ${loc.Symbols.join(', ')}\n`;
        result += `${loc.Summary}\n\n`;
      });
    }

    if (output.ArchitectureInsights) {
      result += `## Architecture Insights\n${output.ArchitectureInsights}\n\n`;
    }

    if (output.Recommendations && output.Recommendations.length > 0) {
      result += `## Recommendations\n`;
      output.Recommendations.forEach((rec, idx) => {
        result += `${idx + 1}. ${rec}\n`;
      });
      result += `\n`;
    }

    return result;
  },

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.1, // Low temperature for focused exploration
    top_p: 0.95,
    thinkingBudget: -1,
  },

  runConfig: {
    max_time_minutes: 3, // Faster than CodebaseInvestigator
    max_turns: 10, // Adjustable based on thoroughness
  },

  toolConfig: {
    // Read-only tools + web search for documentation lookup
    tools: [
      LS_TOOL_NAME,
      READ_FILE_TOOL_NAME,
      GLOB_TOOL_NAME,
      GREP_TOOL_NAME,
      WEB_SEARCH_TOOL_NAME,
    ],
  },

  promptConfig: {
    query: `Explore the codebase to answer the following query:

<query>
\${query}
</query>

Thoroughness level: \${thoroughness || "medium"}

Provide a clear, concise answer with relevant code locations.`,

    systemPrompt: `You are the **Explore Agent**, a specialized AI for rapid codebase exploration and answering questions.

Your **CORE MISSION** is to quickly find relevant code, understand system architecture, and provide clear, actionable answers.

## Your Strengths

- **Speed:** Prioritize efficiency - get to the answer quickly
- **Precision:** Focus on what's relevant to the query
- **Clarity:** Provide clear, easy-to-understand explanations
- **Practicality:** Give actionable insights developers can use immediately

## Exploration Strategy

### 1. Quick Thoroughness (3-5 turns)
   - Start with targeted searches (grep for keywords)
   - Read 1-2 most relevant files
   - Provide surface-level answer with key locations
   - Best for simple "where is X" questions

### 2. Medium Thoroughness (5-8 turns) - DEFAULT
   - Search for relevant files and patterns
   - Read 3-5 key files to understand implementation
   - Trace dependencies and interactions
   - Provide detailed answer with architectural context
   - Best for "how does X work" questions

### 3. Thorough Exploration (8-12 turns)
   - Comprehensive search across codebase
   - Read multiple files, follow call chains
   - Understand full subsystem architecture
   - Identify patterns, conventions, best practices
   - Best for "explain the entire X system" questions

## Tool Usage Guidelines

**Phase 1: Discovery (1-3 turns)**
- Use \`grep\` to search for keywords, function names, or patterns
- Use \`glob\` to find files by name or extension patterns
- Use \`ls\` to understand directory structure
- Prioritize files that seem most central to the query

**Phase 2: Understanding (2-5 turns)**
- Use \`read_file\` to examine relevant implementations
- Focus on entry points, main functions, and key classes
- Follow imports and function calls to understand data flow
- Note architectural patterns and design decisions

**Phase 3: Synthesis (1-2 turns)**
- Connect the dots between different code locations
- Identify how components interact
- Formulate clear, concise answer
- List relevant locations with context

**Optional: Documentation Lookup**
- Use \`web_search\` if you encounter unfamiliar libraries or concepts
- Search for official documentation, not general tutorials

## Output Requirements

You must call \`complete_task\` with a JSON object:

{
  "Answer": "Clear, direct answer to the query (2-5 sentences)",
  "KeyFindings": [
    "Important insight #1",
    "Important insight #2",
    "etc."
  ],
  "RelevantLocations": [
    {
      "FilePath": "path/to/file.ts",
      "LineRange": "45-67" /* optional */,
      "Symbols": ["functionName", "ClassName"],
      "Summary": "What this code does and why it's relevant"
    }
  ],
  "ArchitectureInsights": "High-level observations about design (optional)",
  "Recommendations": ["Next steps or suggestions (optional)"]
}

## Best Practices

✅ **DO:**
- Start with the most promising search terms
- Read files selectively (not entire codebases)
- Provide line numbers when possible for precision
- Explain *why* locations are relevant, not just *where* they are
- Adapt exploration depth to the thoroughness level
- Use web_search to understand unfamiliar libraries

❌ **DON'T:**
- Read every file you find (be selective!)
- Provide vague answers like "it's in the src directory"
- List locations without explaining their relevance
- Spend too many turns on tangential details
- Continue exploring after you have a good answer

## Termination Criteria

Call \`complete_task\` when:
- You have a clear, well-supported answer to the query
- You've identified 3-5 relevant code locations
- You've reached the turn limit for the thoroughness level
- Further exploration would have diminishing returns

Remember: Speed and relevance matter. Provide useful insights quickly, not perfect insights slowly.`,
  },
};

/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { AgentDefinition } from '../types.js';
import { z } from 'zod';
import { DEFAULT_GEMINI_MODEL } from '../../config/models.js';
import {
  READ_FILE_TOOL_NAME,
  GLOB_TOOL_NAME,
  GREP_TOOL_NAME,
} from '../../tools/tool-names.js';

/**
 * Schema for a key point in the summary.
 */
const KeyPointSchema = z.object({
  point: z.string().describe('The key point or insight'),
  category: z.string().describe('Category or theme this point belongs to'),
  importance: z
    .enum(['critical', 'high', 'medium', 'low'])
    .describe('Importance level'),
  details: z.string().optional().describe('Additional details or context'),
  sourceReference: z
    .string()
    .optional()
    .describe('Reference to source (file, section, etc.)'),
});

/**
 * Schema for summary sections.
 */
const SummarySectionSchema = z.object({
  title: z.string().describe('Section title'),
  content: z.string().describe('Section content'),
  keyPoints: z.array(z.string()).describe('Key points in this section'),
});

/**
 * Schema for action items or recommendations.
 */
const ActionItemSchema = z.object({
  action: z.string().describe('Recommended action'),
  priority: z.enum(['high', 'medium', 'low']).describe('Priority level'),
  rationale: z.string().describe('Why this action is recommended'),
  dependencies: z
    .array(z.string())
    .optional()
    .describe('Dependencies or prerequisites'),
});

/**
 * Schema for summarization report.
 */
const SummarizerReportSchema = z.object({
  executiveSummary: z
    .string()
    .describe(
      'High-level executive summary (100-150 words) suitable for quick overview',
    ),
  detailedSummary: z
    .string()
    .describe('Detailed summary covering all major points (300-500 words)'),
  keyPoints: z
    .array(KeyPointSchema)
    .describe('Most important points extracted from the content'),
  sections: z
    .array(SummarySectionSchema)
    .optional()
    .describe('Organized sections for structured content'),
  themes: z.array(z.string()).describe('Main themes or topics identified'),
  statistics: z
    .object({
      sourceCount: z.number().describe('Number of sources processed'),
      totalWords: z
        .number()
        .optional()
        .describe('Approximate total word count'),
      compressionRatio: z
        .number()
        .optional()
        .describe('Compression ratio (summary length / original length)'),
    })
    .describe('Summary statistics'),
  actionItems: z
    .array(ActionItemSchema)
    .optional()
    .describe('Actionable recommendations or next steps'),
  gaps: z
    .array(z.string())
    .optional()
    .describe('Information gaps or missing pieces'),
  relatedTopics: z
    .array(z.string())
    .optional()
    .describe('Related topics worth exploring'),
});

/**
 * Summarizer Agent - Intelligent multi-document summarization and synthesis.
 *
 * This agent specializes in:
 * - Generating concise and comprehensive summaries
 * - Extracting key points and insights
 * - Synthesizing information from multiple sources
 * - Creating structured, hierarchical summaries
 * - Identifying themes and patterns
 * - Generating executive summaries
 * - Providing actionable recommendations
 *
 * The agent uses context engineering to maintain coherence across
 * multiple summarization tasks and build comprehensive understanding.
 */
export const SummarizerAgent: AgentDefinition<typeof SummarizerReportSchema> = {
  name: 'summarizer_agent',
  displayName: 'Summarizer Agent',
  description: `An intelligent agent for multi-document summarization and synthesis.

  Use this agent when you need to:
  - Generate concise summaries of long documents
  - Synthesize information from multiple sources
  - Extract key points and insights
  - Create executive summaries for quick overview
  - Organize complex information hierarchically
  - Identify main themes and patterns
  - Generate actionable recommendations
  - Create structured reports from unstructured data
  - Compress information while preserving meaning

  The agent uses context engineering to maintain consistency across
  summarization tasks and build on previous analyses.`,

  inputConfig: {
    inputs: {
      sources: {
        description:
          'Paths to documents or descriptions of content to summarize. Can be file paths, directory paths, or text descriptions.',
        type: 'string[]',
        required: true,
      },
      style: {
        description: `Summarization style to use:
          - "executive": Focus on high-level insights and implications (default)
          - "technical": Include specific details and methodologies
          - "academic": Emphasize research findings and evidence
          - "narrative": Create engaging, story-like flow`,
        type: 'string',
        required: false,
      },
      length: {
        description: `Desired summary length:
          - "brief": Maximum compression, 50-250 words total
          - "moderate": Balanced approach, 300-650 words total (default)
          - "comprehensive": Minimal information loss, 500-1000 words total`,
        type: 'string',
        required: false,
      },
      focus: {
        description:
          'Specific topics or aspects to focus on during summarization. Leave empty for general summary.',
        type: 'string[]',
        required: false,
      },
      includeActionItems: {
        description:
          'Whether to include actionable recommendations in the summary. Default: true',
        type: 'boolean',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description:
      'Comprehensive summarization report with executive summary, detailed summary, key points, and recommendations.',
    schema: SummarizerReportSchema,
  },

  processOutput: (output) => {
    let result = `# Summarization Report\n\n`;

    // Executive Summary
    result += `## Executive Summary\n\n${output.executiveSummary}\n\n`;

    // Themes
    if (output.themes && output.themes.length > 0) {
      result += `## Main Themes\n\n`;
      output.themes.forEach((theme, idx) => {
        result += `${idx + 1}. ${theme}\n`;
      });
      result += `\n`;
    }

    // Key Points
    if (output.keyPoints && output.keyPoints.length > 0) {
      result += `## Key Points\n\n`;
      output.keyPoints.forEach((kp, idx) => {
        const importance = kp.importance.toUpperCase();
        result += `### ${idx + 1}. [${importance}] ${kp.point}\n\n`;
        result += `**Category:** ${kp.category}\n\n`;
        if (kp.details) {
          result += `${kp.details}\n\n`;
        }
        if (kp.sourceReference) {
          result += `*Source: ${kp.sourceReference}*\n\n`;
        }
      });
    }

    // Detailed Summary
    result += `## Detailed Summary\n\n${output.detailedSummary}\n\n`;

    // Sections (if present)
    if (output.sections && output.sections.length > 0) {
      result += `## Detailed Sections\n\n`;
      output.sections.forEach((section, idx) => {
        result += `### ${idx + 1}. ${section.title}\n\n`;
        result += `${section.content}\n\n`;
        if (section.keyPoints.length > 0) {
          result += `**Key Points:**\n`;
          section.keyPoints.forEach((point) => {
            result += `- ${point}\n`;
          });
          result += `\n`;
        }
      });
    }

    // Action Items
    if (output.actionItems && output.actionItems.length > 0) {
      result += `## Recommended Actions\n\n`;
      output.actionItems.forEach((item, idx) => {
        const priority = item.priority.toUpperCase();
        result += `### ${idx + 1}. [${priority}] ${item.action}\n\n`;
        result += `**Rationale:** ${item.rationale}\n\n`;
        if (item.dependencies && item.dependencies.length > 0) {
          result += `**Dependencies:**\n`;
          item.dependencies.forEach((dep) => {
            result += `- ${dep}\n`;
          });
          result += `\n`;
        }
      });
    }

    // Gaps
    if (output.gaps && output.gaps.length > 0) {
      result += `## Information Gaps\n\n`;
      output.gaps.forEach((gap, idx) => {
        result += `${idx + 1}. ${gap}\n`;
      });
      result += `\n`;
    }

    // Related Topics
    if (output.relatedTopics && output.relatedTopics.length > 0) {
      result += `## Related Topics for Further Exploration\n\n`;
      output.relatedTopics.forEach((topic, idx) => {
        result += `${idx + 1}. ${topic}\n`;
      });
      result += `\n`;
    }

    // Statistics
    result += `## Statistics\n\n`;
    result += `- **Sources Processed:** ${output.statistics.sourceCount}\n`;
    if (output.statistics.totalWords) {
      result += `- **Total Words:** ~${output.statistics.totalWords}\n`;
    }
    if (output.statistics.compressionRatio) {
      result += `- **Compression Ratio:** ${(output.statistics.compressionRatio * 100).toFixed(1)}%\n`;
    }
    result += `\n`;

    return result;
  },

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.4, // Moderate temperature for balanced creativity and accuracy
    top_p: 0.95,
    thinkingBudget: -1,
  },

  runConfig: {
    max_time_minutes: 5,
    max_turns: 25,
  },

  toolConfig: {
    tools: [READ_FILE_TOOL_NAME, GLOB_TOOL_NAME, GREP_TOOL_NAME],
  },

  promptConfig: {
    query: `Summarize the following sources:

<sources>
\${sources.join('\\n')}
</sources>

Style: \${style || "executive"}
Length: \${length || "moderate"}
Focus areas: \${focus && focus.length > 0 ? focus.join(', ') : 'General summary'}
Include action items: \${includeActionItems !== false ? 'Yes' : 'No'}

Please provide a comprehensive summary following the specified parameters.`,

    systemPrompt: `You are a Summarizer Agent specialized in creating concise, comprehensive summaries from complex information.

# Your Capabilities

1. **Information Extraction**
   - Identify main ideas and supporting details
   - Extract key facts, figures, and quotes
   - Recognize important concepts and definitions
   - Filter noise and redundancy

2. **Synthesis**
   - Combine information from multiple sources
   - Identify common themes and patterns
   - Resolve contradictions or differences
   - Build coherent narrative

3. **Structuring**
   - Organize information hierarchically
   - Create logical flow and transitions
   - Group related concepts
   - Maintain clear sections

4. **Prioritization**
   - Determine importance of information
   - Focus on critical insights
   - Balance breadth and depth
   - Preserve essential context

5. **Abstraction**
   - Generalize from specific examples
   - Identify higher-level patterns
   - Create conceptual frameworks
   - Distill complex ideas

# Your Process

1. **Information Gathering**
   - Read all source materials thoroughly
   - Identify document types and purposes
   - Note overall structure and organization
   - Track sources for reference

2. **Analysis**
   - Extract key points from each source
   - Identify main themes and topics
   - Note supporting evidence and examples
   - Recognize patterns and connections

3. **Synthesis**
   - Combine related information
   - Resolve redundancy
   - Build coherent narrative
   - Organize hierarchically

4. **Summary Creation**
   - Write executive summary (high-level overview)
   - Create detailed summary (comprehensive coverage)
   - Extract and categorize key points
   - Identify themes and patterns

5. **Enhancement**
   - Add actionable recommendations
   - Note information gaps
   - Suggest related topics
   - Provide statistics and metrics

# Guidelines for Different Styles

**Executive Style:**
- Focus on high-level insights and implications
- Emphasize business value and decisions
- Keep language clear and non-technical
- Highlight actionable recommendations

**Technical Style:**
- Include specific details and methodologies
- Use precise technical terminology
- Explain technical concepts
- Focus on implementation details

**Academic Style:**
- Emphasize research findings and evidence
- Include methodology and limitations
- Use formal academic language
- Reference sources and citations

**Narrative Style:**
- Create engaging, story-like flow
- Use concrete examples and anecdotes
- Make information accessible
- Build logical progression

# Summary Length Guidelines

**Brief:**
- Executive summary: 50-100 words
- Detailed summary: 150-250 words
- Focus on absolute essentials
- Maximum compression

**Moderate:**
- Executive summary: 100-150 words
- Detailed summary: 300-500 words
- Balance breadth and depth
- Standard approach

**Comprehensive:**
- Executive summary: 150-200 words
- Detailed summary: 500-800 words
- Include more details and context
- Minimal information loss

# Guidelines

- Maintain objectivity and accuracy
- Preserve key information and context
- Use clear, concise language
- Avoid redundancy and wordiness
- Group related information logically
- Highlight what's most important
- Provide actionable insights
- Note limitations and gaps

# Context Awareness

You have access to shared context from previous analyses. Use this to:
- Build on previous summaries
- Maintain consistent terminology
- Identify evolving themes
- Avoid repeating information
- Connect related topics
- Build comprehensive understanding

When summarizing, focus on:
- What are the main points?
- Why are they important?
- What are the implications?
- What should be done next?
- What's missing or unclear?

When you complete your task, provide a comprehensive summary by calling the complete_task function with a JSON object following the SummarizerReport schema.`,
  },
};

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
  WEB_SEARCH_TOOL_NAME,
} from '../../tools/tool-names.js';

/**
 * Schema for citation/reference information.
 */
const CitationSchema = z.object({
  authors: z.array(z.string()).describe('List of authors'),
  title: z.string().describe('Title of the work'),
  year: z.number().optional().describe('Publication year'),
  source: z
    .string()
    .optional()
    .describe('Publication source (journal, conference, etc.)'),
  url: z.string().optional().describe('URL or DOI'),
});

/**
 * Schema for key concept extraction.
 */
const ConceptSchema = z.object({
  concept: z.string().describe('Main concept or topic'),
  definition: z.string().describe('Definition or explanation'),
  relevance: z
    .enum(['high', 'medium', 'low'])
    .describe('Relevance to the main topic'),
  relatedConcepts: z
    .array(z.string())
    .optional()
    .describe('Related concepts mentioned'),
});

/**
 * Schema for methodology analysis.
 */
const MethodologySchema = z.object({
  approach: z.string().describe('Research or analysis approach used'),
  methods: z.array(z.string()).describe('Specific methods or techniques'),
  dataSource: z.string().optional().describe('Data sources used'),
  limitations: z
    .array(z.string())
    .optional()
    .describe('Identified limitations'),
});

/**
 * Schema for findings/results.
 */
const FindingSchema = z.object({
  finding: z.string().describe('Key finding or result'),
  evidence: z.string().describe('Supporting evidence'),
  significance: z
    .enum(['critical', 'important', 'moderate', 'minor'])
    .describe('Significance level'),
  implications: z
    .array(z.string())
    .optional()
    .describe('Implications of this finding'),
});

/**
 * Schema for literature analysis report.
 */
const LiteratureAnalysisReportSchema = z.object({
  summary: z
    .string()
    .describe('Comprehensive summary of the literature (200-300 words)'),
  citation: CitationSchema.optional().describe(
    'Citation information if available',
  ),
  mainThemes: z.array(z.string()).describe('3-5 main themes or topics covered'),
  keyConcepts: z
    .array(ConceptSchema)
    .describe('Important concepts and definitions'),
  methodology: MethodologySchema.optional().describe(
    'Methodology used (if applicable)',
  ),
  keyFindings: z
    .array(FindingSchema)
    .describe('Most important findings or conclusions'),
  criticalAnalysis: z
    .object({
      strengths: z.array(z.string()).describe('Strengths of the work'),
      weaknesses: z.array(z.string()).describe('Weaknesses or gaps'),
      contributions: z
        .array(z.string())
        .describe('Key contributions to the field'),
    })
    .describe('Critical analysis of the literature'),
  relatedWork: z
    .array(CitationSchema)
    .optional()
    .describe('Related works cited or referenced'),
  futureDirections: z
    .array(z.string())
    .optional()
    .describe('Suggested future research directions'),
  practicalApplications: z
    .array(z.string())
    .optional()
    .describe('Practical applications or use cases'),
});

/**
 * Literature Analyzer Agent - Deep analysis of academic and technical literature.
 *
 * This agent specializes in:
 * - Analyzing research papers, technical documents, and academic literature
 * - Extracting key concepts, methodologies, and findings
 * - Identifying themes and patterns across multiple documents
 * - Providing critical analysis and synthesis
 * - Extracting citations and building reference networks
 * - Identifying research gaps and future directions
 *
 * The agent leverages context engineering to build comprehensive understanding
 * across multiple documents and maintain consistency in analysis.
 */
export const LiteratureAnalyzerAgent: AgentDefinition<
  typeof LiteratureAnalysisReportSchema
> = {
  name: 'literature_analyzer_agent',
  displayName: 'Literature Analyzer Agent',
  description: `An intelligent agent for deep analysis of academic and technical literature.

Use this agent when you need to:
- Analyze research papers and technical documents
- Extract key concepts, methodologies, and findings
- Understand complex academic or technical content
- Identify themes across multiple documents
- Synthesize information from various sources
- Evaluate research quality and contributions
- Extract and organize citations
- Identify research gaps and future directions
- Generate critical analysis and insights

The agent uses context engineering to maintain comprehensive understanding
across multiple documents and build connections between related works.`,

  inputConfig: {
    inputs: {
      document: {
        description: `Path to the document or description of the literature to analyze. Examples:
          - "docs/research-paper.pdf"
          - "papers/ml-architecture-2024.md"
          - "Analyze the technical documentation in the docs/ folder"`,
        type: 'string',
        required: true,
      },
      focus: {
        description: `Specific aspect to focus on during analysis. Examples:
          - "methodology" - Focus on research methods and approaches
          - "findings" - Emphasize key results and conclusions
          - "concepts" - Extract and analyze key concepts and definitions
          - "critical analysis" - Deep evaluation of strengths/weaknesses
          - "comparison" - Compare with other works in the field`,
        type: 'string',
        required: false,
      },
      depth: {
        description: `Depth of analysis required:
          - "overview": High-level summary and main points (quick, 5-10 turns)
          - "detailed": Thorough analysis with key insights (balanced, 15-20 turns, default)
          - "comprehensive": Deep dive with extensive cross-referencing (thorough, 25-30 turns)`,
        type: 'string',
        required: false,
      },
      compareTo: {
        description: `Array of paths to other documents to compare with. Enables comparative analysis to identify:
          - Similarities and differences in methodology
          - Contrasting findings or conclusions
          - Evolution of concepts across works
          - Complementary or conflicting perspectives`,
        type: 'string[]',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description:
      'Comprehensive literature analysis report with key concepts, findings, and critical evaluation.',
    schema: LiteratureAnalysisReportSchema,
  },

  processOutput: (output) => {
    let result = `# Literature Analysis Report\n\n`;

    // Summary
    result += `## Summary\n${output.summary}\n\n`;

    // Citation
    if (output.citation) {
      result += `## Citation\n`;
      result += `**Title:** ${output.citation.title}\n`;
      result += `**Authors:** ${output.citation.authors.join(', ')}\n`;
      if (output.citation.year) {
        result += `**Year:** ${output.citation.year}\n`;
      }
      if (output.citation.source) {
        result += `**Source:** ${output.citation.source}\n`;
      }
      if (output.citation.url) {
        result += `**URL:** ${output.citation.url}\n`;
      }
      result += `\n`;
    }

    // Main Themes
    if (output.mainThemes && output.mainThemes.length > 0) {
      result += `## Main Themes\n`;
      output.mainThemes.forEach((theme, idx) => {
        result += `${idx + 1}. ${theme}\n`;
      });
      result += `\n`;
    }

    // Key Concepts
    if (output.keyConcepts && output.keyConcepts.length > 0) {
      result += `## Key Concepts\n\n`;
      output.keyConcepts.forEach((concept, idx) => {
        result += `### ${idx + 1}. ${concept.concept} (Relevance: ${concept.relevance})\n`;
        result += `${concept.definition}\n`;
        if (concept.relatedConcepts && concept.relatedConcepts.length > 0) {
          result += `**Related Concepts:** ${concept.relatedConcepts.join(', ')}\n`;
        }
        result += `\n`;
      });
    }

    // Methodology
    if (output.methodology) {
      result += `## Methodology\n`;
      result += `**Approach:** ${output.methodology.approach}\n\n`;
      result += `**Methods:**\n`;
      output.methodology.methods.forEach((method) => {
        result += `- ${method}\n`;
      });
      if (output.methodology.dataSource) {
        result += `\n**Data Source:** ${output.methodology.dataSource}\n`;
      }
      if (output.methodology.limitations && output.methodology.limitations.length > 0) {
        result += `\n**Limitations:**\n`;
        output.methodology.limitations.forEach((limitation) => {
          result += `- ${limitation}\n`;
        });
      }
      result += `\n`;
    }

    // Key Findings
    if (output.keyFindings && output.keyFindings.length > 0) {
      result += `## Key Findings\n\n`;
      output.keyFindings.forEach((finding, idx) => {
        result += `### ${idx + 1}. ${finding.finding} (${finding.significance})\n`;
        result += `**Evidence:** ${finding.evidence}\n`;
        if (finding.implications && finding.implications.length > 0) {
          result += `**Implications:**\n`;
          finding.implications.forEach((implication) => {
            result += `- ${implication}\n`;
          });
        }
        result += `\n`;
      });
    }

    // Critical Analysis
    if (output.criticalAnalysis) {
      result += `## Critical Analysis\n\n`;
      result += `### Strengths\n`;
      output.criticalAnalysis.strengths.forEach((strength) => {
        result += `- ${strength}\n`;
      });
      result += `\n### Weaknesses\n`;
      output.criticalAnalysis.weaknesses.forEach((weakness) => {
        result += `- ${weakness}\n`;
      });
      result += `\n### Key Contributions\n`;
      output.criticalAnalysis.contributions.forEach((contribution) => {
        result += `- ${contribution}\n`;
      });
      result += `\n`;
    }

    // Related Work
    if (output.relatedWork && output.relatedWork.length > 0) {
      result += `## Related Work\n\n`;
      output.relatedWork.forEach((work, idx) => {
        result += `${idx + 1}. ${work.authors.join(', ')}. "${work.title}"`;
        if (work.year) {
          result += ` (${work.year})`;
        }
        if (work.source) {
          result += `. ${work.source}`;
        }
        if (work.url) {
          result += ` [Link](${work.url})`;
        }
        result += `\n`;
      });
      result += `\n`;
    }

    // Future Directions
    if (output.futureDirections && output.futureDirections.length > 0) {
      result += `## Future Research Directions\n`;
      output.futureDirections.forEach((direction, idx) => {
        result += `${idx + 1}. ${direction}\n`;
      });
      result += `\n`;
    }

    // Practical Applications
    if (output.practicalApplications && output.practicalApplications.length > 0) {
      result += `## Practical Applications\n`;
      output.practicalApplications.forEach((application, idx) => {
        result += `${idx + 1}. ${application}\n`;
      });
      result += `\n`;
    }

    return result;
  },

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.3, // Moderate temperature for balanced analysis
    top_p: 0.95,
    thinkingBudget: -1, // No extended thinking budget limit
  },

  runConfig: {
    max_time_minutes: 15, // Allow sufficient time for thorough analysis
    max_turns: 30, // Adjust based on depth parameter
  },

  toolConfig: {
    tools: [
      READ_FILE_TOOL_NAME,
      GLOB_TOOL_NAME,
      GREP_TOOL_NAME,
      WEB_SEARCH_TOOL_NAME,
    ],
  },

  promptConfig: {
    query: `Analyze the following literature:

<document>
\${document}
</document>

${`\${focus ? \`<focus>Focus on: \${focus}</focus>\` : ''}`}

${`\${depth ? \`<depth>Analysis depth: \${depth}</depth>\` : '<depth>Analysis depth: detailed</depth>'}`}

${`\${compareTo && compareTo.length > 0 ? \`<comparison>Compare with: \${compareTo.join(', ')}</comparison>\` : ''}`}

Provide a comprehensive analysis following the LiteratureAnalysisReport schema.`,

    systemPrompt: `You are a Literature Analyzer Agent specialized in deep analysis of academic and technical literature.

# Your Capabilities

1. **Content Analysis**
   - Extract and organize main themes and concepts
   - Identify key arguments and claims
   - Analyze structure and organization
   - Understand technical terminology

2. **Methodology Analysis**
   - Identify research methods and approaches
   - Evaluate methodology appropriateness
   - Note limitations and constraints
   - Assess data sources and quality

3. **Finding Extraction**
   - Identify key findings and results
   - Evaluate evidence quality
   - Assess significance of contributions
   - Extract quantitative and qualitative results

4. **Critical Evaluation**
   - Assess strengths and weaknesses
   - Identify contributions to the field
   - Evaluate logical consistency
   - Note potential biases or limitations

5. **Synthesis**
   - Connect concepts across documents
   - Identify patterns and trends
   - Build comprehensive understanding
   - Generate insights and implications

6. **Citation Analysis**
   - Extract citation information
   - Build reference networks
   - Identify influential works
   - Track research lineage

# Your Process

1. **Initial Reading**
   - Scan the document structure
   - Identify main sections and organization
   - Note document type and purpose

2. **Deep Analysis**
   - Extract key concepts and definitions
   - Analyze methodology (if applicable)
   - Identify main findings and claims
   - Note supporting evidence

3. **Critical Evaluation**
   - Assess quality and rigor
   - Identify strengths and contributions
   - Note weaknesses or gaps
   - Evaluate implications

4. **Synthesis**
   - Connect to related works in context
   - Identify broader themes
   - Generate insights
   - Suggest future directions

5. **Documentation**
   - Organize findings systematically
   - Provide clear summaries
   - Include relevant citations
   - Highlight key takeaways

# Guidelines

- Be thorough and systematic in analysis
- Maintain objectivity in evaluation
- Support claims with evidence from the text
- Consider context and broader implications
- Use precise academic language
- Identify both explicit and implicit concepts
- Note uncertainty or ambiguity
- Connect to previous analysis in context

# Context Awareness

You have access to shared context from previous analyses. Use this to:
- Build comprehensive understanding across documents
- Identify connections between works
- Maintain consistent terminology
- Track evolving concepts
- Avoid redundant analysis
- Build on previous insights

When analyzing literature, pay special attention to:
- Main research questions or objectives
- Methodology and data sources
- Key findings and contributions
- Limitations and future work
- Practical applications

# Depth-Based Strategy

**Overview Depth (5-10 turns):**
- Focus on high-level summary and main themes
- Identify key findings and contributions
- Provide basic critical analysis
- Suitable for quick understanding

**Detailed Depth (15-20 turns) - DEFAULT:**
- Thorough extraction of concepts and methodology
- Comprehensive finding analysis
- In-depth critical evaluation
- Identify patterns and connections
- Suitable for most analysis needs

**Comprehensive Depth (25-30 turns):**
- Exhaustive concept extraction and definition
- Deep methodology evaluation
- Extensive cross-referencing
- Detailed comparative analysis
- Track all citations and references
- Suitable for research synthesis

# Output Requirements

When you complete your analysis, call \`complete_task\` with a JSON object following the LiteratureAnalysisReport schema. Ensure all required fields are populated with high-quality, evidence-based content.

Remember: Your goal is to provide insightful, academically rigorous analysis that helps researchers and practitioners understand and build upon existing literature.`,
  },
};

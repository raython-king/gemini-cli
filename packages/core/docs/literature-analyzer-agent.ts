/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { AgentDefinition } from './types.js';
import { z } from 'zod';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import {
  READ_FILE_TOOL_NAME,
  GLOB_TOOL_NAME,
  GREP_TOOL_NAME,
  WEB_SEARCH_TOOL_NAME,
} from '../tools/tool-names.js';

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

  taskDefinitionSchema: z.object({
    document: z
      .string()
      .describe(
        'Path to the document or description of the literature to analyze',
      ),
    focus: z
      .string()
      .optional()
      .describe(
        'Specific aspect to focus on (e.g., "methodology", "findings", "concepts")',
      ),
    depth: z
      .enum(['overview', 'detailed', 'comprehensive'])
      .optional()
      .default('detailed')
      .describe('Depth of analysis required'),
    compareTo: z
      .array(z.string())
      .optional()
      .describe('Other documents to compare with'),
  }),

  reportSchema: LiteratureAnalysisReportSchema,

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

When you complete your task, provide a comprehensive analysis following the LiteratureAnalysisReport schema.`,

  model: DEFAULT_GEMINI_MODEL,

  tools: [
    READ_FILE_TOOL_NAME,
    GLOB_TOOL_NAME,
    GREP_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
  ],

  options: {
    maxTurns: 30,
    temperature: 0.3, // Moderate temperature for balanced analysis
    topP: 0.95,
  },
};

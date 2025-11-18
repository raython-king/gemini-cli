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
} from '../tools/tool-names.js';

/**
 * Schema for data transformation step.
 */
const DataTransformSchema = z.object({
  step: z.string().describe('Transformation step name'),
  operation: z
    .enum(['extract', 'filter', 'transform', 'aggregate', 'validate'])
    .describe('Type of operation'),
  input: z.string().describe('Input data description'),
  output: z.string().describe('Output data description'),
  logic: z.string().describe('Transformation logic applied'),
});

/**
 * Schema for data quality metrics.
 */
const DataQualitySchema = z.object({
  completeness: z.number().min(0).max(1).describe('Completeness score (0-1)'),
  accuracy: z.number().min(0).max(1).describe('Accuracy score (0-1)'),
  consistency: z.number().min(0).max(1).describe('Consistency score (0-1)'),
  issues: z
    .array(z.string())
    .describe('List of data quality issues identified'),
  recommendations: z
    .array(z.string())
    .describe('Recommendations for data quality improvement'),
});

/**
 * Schema for data flow processing report.
 */
const DataFlowReportSchema = z.object({
  summary: z
    .string()
    .describe('High-level summary of data flow processing results'),
  dataSource: z
    .object({
      type: z.string().describe('Type of data source'),
      location: z.string().describe('Location of data source'),
      format: z.string().describe('Data format'),
      recordCount: z.number().describe('Number of records processed'),
    })
    .describe('Information about the data source'),
  transformations: z
    .array(DataTransformSchema)
    .describe('List of transformations applied to the data'),
  dataQuality: DataQualitySchema.describe('Data quality assessment'),
  outputSchema: z
    .record(z.string(), z.string())
    .describe('Schema of the output data (field name -> type)'),
  insights: z
    .array(z.string())
    .describe('Key insights discovered during data processing'),
  nextSteps: z
    .array(z.string())
    .optional()
    .describe('Recommended next steps for data processing pipeline'),
});

/**
 * DataFlow Agent - Automated data flow processing and transformation.
 *
 * This agent specializes in:
 * - Extracting data from various sources (files, APIs, databases)
 * - Transforming and cleaning data
 * - Validating data quality
 * - Creating data processing pipelines
 * - Identifying patterns and anomalies in data
 *
 * The agent uses context engineering to learn from previous data processing
 * tasks and apply learned patterns to new data.
 */
export const DataFlowAgent: AgentDefinition<typeof DataFlowReportSchema> = {
  name: 'data_flow_agent',
  displayName: 'Data Flow Agent',
  description: `An intelligent agent for automated data flow processing and transformation.

  Use this agent when you need to:
  - Process and transform data from various sources
  - Extract structured information from unstructured data
  - Validate data quality and identify issues
  - Create data processing pipelines
  - Aggregate and analyze data patterns
  - Clean and normalize data
  - Convert between data formats

  The agent learns from previous data processing tasks and can apply
  learned transformations to similar data structures.`,

  taskDefinitionSchema: z.object({
    dataSource: z.string().describe('Path or description of the data source'),
    task: z
      .string()
      .describe(
        'Description of the data processing task (e.g., "extract user information", "validate data quality")',
      ),
    outputFormat: z
      .string()
      .optional()
      .describe('Desired output format (e.g., JSON, CSV, structured)'),
    validationRules: z
      .array(z.string())
      .optional()
      .describe('Data validation rules to apply'),
  }),

  reportSchema: DataFlowReportSchema,

  systemPrompt: `You are a Data Flow Processing Agent specialized in automated data extraction, transformation, and validation.

# Your Capabilities

1. **Data Extraction**
   - Read data from files (JSON, CSV, XML, plain text, code files)
   - Parse structured and semi-structured data
   - Extract specific fields or patterns from data

2. **Data Transformation**
   - Clean and normalize data
   - Apply transformations (mapping, filtering, aggregation)
   - Convert between data formats
   - Handle missing or malformed data

3. **Data Validation**
   - Assess data quality metrics (completeness, accuracy, consistency)
   - Identify data quality issues
   - Validate against schemas or rules
   - Detect anomalies and outliers

4. **Pipeline Creation**
   - Design multi-step data processing pipelines
   - Document transformation logic
   - Recommend optimization opportunities

# Your Process

1. **Understand the Task**
   - Identify the data source and format
   - Clarify the processing requirements
   - Review any validation rules

2. **Explore the Data**
   - Read sample data to understand structure
   - Identify data types and patterns
   - Note any data quality issues

3. **Process the Data**
   - Apply required transformations step-by-step
   - Track each transformation for documentation
   - Handle errors and edge cases gracefully

4. **Validate Results**
   - Assess data quality metrics
   - Verify transformation correctness
   - Identify remaining issues

5. **Document Findings**
   - Summarize what was done
   - Report data quality assessment
   - Provide actionable insights and recommendations

# Guidelines

- Be thorough in data exploration before processing
- Handle missing or malformed data gracefully
- Provide clear documentation of transformations
- Focus on data quality and accuracy
- Suggest improvements to data processing pipelines
- Learn from previous data processing patterns in context

# Context Awareness

You have access to shared context from previous agent executions. Use this to:
- Apply learned data transformation patterns
- Avoid repeating known issues
- Build upon previous data insights
- Maintain consistency in data processing approaches

When you complete your task, provide a comprehensive report following the DataFlowReport schema.`,

  model: DEFAULT_GEMINI_MODEL,

  tools: [READ_FILE_TOOL_NAME, GLOB_TOOL_NAME, GREP_TOOL_NAME],

  options: {
    maxTurns: 25,
    temperature: 0.2, // Lower temperature for consistent data processing
    topP: 0.9,
  },
};

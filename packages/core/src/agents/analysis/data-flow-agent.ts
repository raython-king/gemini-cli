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

  inputConfig: {
    inputs: {
      dataSource: {
        description: `Path or description of the data source. Examples:
          - "data/users.json" (file path)
          - "src/config/*.yaml" (file pattern)
          - "API endpoint at https://api.example.com/data"
          - "Database table: users"`,
        type: 'string',
        required: true,
      },
      task: {
        description: `Description of the data processing task. Examples:
          - "Extract user information and validate email formats"
          - "Transform CSV data to JSON with normalized fields"
          - "Validate data quality and identify missing values"
          - "Aggregate sales data by region and calculate totals"`,
        type: 'string',
        required: true,
      },
      outputFormat: {
        description: `Desired output format (e.g., "JSON", "CSV", "structured object", "markdown table"). If not specified, agent will choose the most appropriate format.`,
        type: 'string',
        required: false,
      },
      validationRules: {
        description: `Comma-separated list of validation rules to apply. Examples:
          - "email must be valid, age > 0, name required"
          - "no null values, unique IDs, dates in ISO format"
          - "price > 0, quantity is integer"
        Leave empty if no specific validation rules are needed.`,
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description:
      'Comprehensive data flow processing report with transformations, quality metrics, and insights.',
    schema: DataFlowReportSchema,
  },

  processOutput: (output) => {
    let result = `# Data Flow Processing Report\n\n`;

    // Summary
    result += `## Summary\n${output.summary}\n\n`;

    // Data Source Information
    result += `## Data Source\n`;
    result += `- **Type:** ${output.dataSource.type}\n`;
    result += `- **Location:** ${output.dataSource.location}\n`;
    result += `- **Format:** ${output.dataSource.format}\n`;
    result += `- **Records Processed:** ${output.dataSource.recordCount}\n\n`;

    // Transformations
    if (output.transformations && output.transformations.length > 0) {
      result += `## Transformations\n\n`;
      output.transformations.forEach((transform, idx) => {
        result += `### ${idx + 1}. ${transform.step}\n`;
        result += `- **Operation:** ${transform.operation}\n`;
        result += `- **Input:** ${transform.input}\n`;
        result += `- **Output:** ${transform.output}\n`;
        result += `- **Logic:** ${transform.logic}\n\n`;
      });
    }

    // Data Quality
    result += `## Data Quality Assessment\n`;
    result += `- **Completeness:** ${(output.dataQuality.completeness * 100).toFixed(1)}%\n`;
    result += `- **Accuracy:** ${(output.dataQuality.accuracy * 100).toFixed(1)}%\n`;
    result += `- **Consistency:** ${(output.dataQuality.consistency * 100).toFixed(1)}%\n\n`;

    if (output.dataQuality.issues && output.dataQuality.issues.length > 0) {
      result += `### Issues Identified\n`;
      output.dataQuality.issues.forEach((issue, idx) => {
        result += `${idx + 1}. ${issue}\n`;
      });
      result += `\n`;
    }

    if (
      output.dataQuality.recommendations &&
      output.dataQuality.recommendations.length > 0
    ) {
      result += `### Recommendations\n`;
      output.dataQuality.recommendations.forEach((rec, idx) => {
        result += `${idx + 1}. ${rec}\n`;
      });
      result += `\n`;
    }

    // Output Schema
    if (output.outputSchema && Object.keys(output.outputSchema).length > 0) {
      result += `## Output Schema\n`;
      Object.entries(output.outputSchema).forEach(([field, type]) => {
        result += `- **${field}:** ${type}\n`;
      });
      result += `\n`;
    }

    // Insights
    if (output.insights && output.insights.length > 0) {
      result += `## Key Insights\n`;
      output.insights.forEach((insight, idx) => {
        result += `${idx + 1}. ${insight}\n`;
      });
      result += `\n`;
    }

    // Next Steps
    if (output.nextSteps && output.nextSteps.length > 0) {
      result += `## Next Steps\n`;
      output.nextSteps.forEach((step, idx) => {
        result += `${idx + 1}. ${step}\n`;
      });
      result += `\n`;
    }

    return result;
  },

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.2, // Lower temperature for consistent data processing
    top_p: 0.9,
    thinkingBudget: -1,
  },

  runConfig: {
    max_time_minutes: 5, // Allow sufficient time for data processing
    max_turns: 25,
  },

  toolConfig: {
    tools: [READ_FILE_TOOL_NAME, GLOB_TOOL_NAME, GREP_TOOL_NAME],
  },

  promptConfig: {
    query: `Process the following data flow task:

<data_source>
\${dataSource}
</data_source>

<task>
\${task}
</task>

Output format: \${outputFormat || "agent will choose appropriate format"}

Validation rules: \${validationRules || "none specified"}

Complete the data processing task and provide a comprehensive report following the DataFlowReport schema.`,

    systemPrompt: `You are a Data Flow Processing Agent specialized in automated data extraction, transformation, and validation.

# Your Capabilities

1. **Data Extraction**
   - Read data from files (JSON, CSV, XML, plain text, code files)
   - Parse structured and semi-structured data
   - Extract specific fields or patterns from data
   - Handle various data sources and formats

2. **Data Transformation**
   - Clean and normalize data
   - Apply transformations (mapping, filtering, aggregation)
   - Convert between data formats
   - Handle missing or malformed data
   - Apply business logic and rules

3. **Data Validation**
   - Assess data quality metrics (completeness, accuracy, consistency)
   - Identify data quality issues
   - Validate against schemas or rules
   - Detect anomalies and outliers
   - Verify data integrity

4. **Pipeline Creation**
   - Design multi-step data processing pipelines
   - Document transformation logic clearly
   - Recommend optimization opportunities
   - Ensure reproducibility

# Your Process

1. **Understand the Task**
   - Identify the data source and format
   - Clarify the processing requirements
   - Review any validation rules provided
   - Determine the desired output format

2. **Explore the Data**
   - Use \`glob\` to find data files if patterns are provided
   - Use \`grep\` to search for specific patterns in data
   - Use \`read_file\` to examine sample data and understand structure
   - Identify data types, patterns, and schema
   - Note any immediate data quality issues

3. **Process the Data**
   - Apply required transformations step-by-step
   - Track each transformation for documentation
   - Handle errors and edge cases gracefully
   - Maintain data integrity throughout processing
   - Apply validation rules as specified

4. **Validate Results**
   - Assess data quality metrics:
     * Completeness: % of required fields present
     * Accuracy: % of data conforming to expected formats/ranges
     * Consistency: % of data following consistent patterns
   - Verify transformation correctness
   - Identify remaining issues or anomalies
   - Document any data quality concerns

5. **Document Findings**
   - Provide clear summary of processing results
   - List all transformations applied with details
   - Report comprehensive data quality assessment
   - Provide actionable insights and patterns discovered
   - Suggest next steps for data pipeline improvement

# Tool Usage Guidelines

**Discovery Phase:**
- Use \`glob\` to find files matching patterns (e.g., "*.json", "data/*.csv")
- Use \`grep\` to search for specific values or patterns in data files
- Use \`ls\` if you need to understand directory structure

**Processing Phase:**
- Use \`read_file\` to load and examine data files
- Read files selectively - sample first if files are large
- Parse data carefully, handling different formats appropriately

# Output Requirements

You must call \`complete_task\` with a JSON object following the DataFlowReport schema:

{
  "summary": "Clear summary of what was processed and key results",
  "dataSource": {
    "type": "File/API/Database/etc",
    "location": "Specific path or identifier",
    "format": "JSON/CSV/XML/etc",
    "recordCount": 42
  },
  "transformations": [
    {
      "step": "Step name",
      "operation": "extract|filter|transform|aggregate|validate",
      "input": "Description of input data",
      "output": "Description of output data",
      "logic": "Explanation of transformation logic"
    }
  ],
  "dataQuality": {
    "completeness": 0.95,  // Score from 0 to 1
    "accuracy": 0.98,      // Score from 0 to 1
    "consistency": 0.92,   // Score from 0 to 1
    "issues": ["List of issues found"],
    "recommendations": ["How to improve quality"]
  },
  "outputSchema": {
    "fieldName": "type description"
  },
  "insights": [
    "Key patterns or findings from the data"
  ],
  "nextSteps": [
    "Optional suggestions for next actions"
  ]
}

# Best Practices

✅ **DO:**
- Handle errors gracefully and report them clearly
- Validate data thoroughly before and after transformations
- Document each transformation step with clear logic
- Provide realistic quality scores based on actual analysis
- Give specific, actionable recommendations
- Consider edge cases and malformed data
- Maintain data type integrity during transformations

❌ **DON'T:**
- Skip validation steps
- Assume data is well-formed without checking
- Provide vague quality assessments
- Ignore missing or malformed data
- Apply transformations without documenting them
- Make assumptions about data structure without verifying

# Data Quality Scoring Guidelines

**Completeness (0-1):**
- 1.0: All required fields present in all records
- 0.8-0.99: Minor missing values in optional fields
- 0.5-0.79: Some missing required fields
- <0.5: Significant missing data

**Accuracy (0-1):**
- 1.0: All data conforms to expected format/range
- 0.8-0.99: Minor format inconsistencies
- 0.5-0.79: Some invalid values or out-of-range data
- <0.5: Significant data validation failures

**Consistency (0-1):**
- 1.0: Perfect consistency in patterns, formats, conventions
- 0.8-0.99: Minor inconsistencies in formatting
- 0.5-0.79: Multiple inconsistent patterns
- <0.5: Highly inconsistent data

# Context Awareness

You have access to shared context from previous agent executions. Use this to:
- Apply learned data transformation patterns
- Avoid repeating known issues
- Build upon previous data insights
- Maintain consistency in data processing approaches
- Reference similar data structures seen before

# Termination Criteria

Call \`complete_task\` when:
- You have successfully processed the data according to the task
- All required transformations have been applied
- Data quality has been assessed comprehensively
- You have documented all findings and insights
- You have reached the turn limit

Remember: Thoroughness and accuracy in data processing are paramount. Take time to validate your work and provide actionable insights.`,
  },
};

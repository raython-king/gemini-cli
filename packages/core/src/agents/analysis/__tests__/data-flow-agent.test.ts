/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { DataFlowAgent } from '../data-flow-agent.js';
import { z } from 'zod';

describe('DataFlowAgent', () => {
  describe('Agent Definition', () => {
    it('should have correct agent metadata', () => {
      expect(DataFlowAgent.name).toBe('data_flow_agent');
      expect(DataFlowAgent.displayName).toBe('Data Flow Agent');
      expect(DataFlowAgent.description).toContain('data flow processing');
    });

    it('should define required input fields', () => {
      expect(DataFlowAgent.inputConfig.inputs.dataSource).toBeDefined();
      expect(DataFlowAgent.inputConfig.inputs.dataSource.required).toBe(true);
      expect(DataFlowAgent.inputConfig.inputs.dataSource.type).toBe('string');

      expect(DataFlowAgent.inputConfig.inputs.task).toBeDefined();
      expect(DataFlowAgent.inputConfig.inputs.task.required).toBe(true);
      expect(DataFlowAgent.inputConfig.inputs.task.type).toBe('string');
    });

    it('should define optional input fields', () => {
      expect(DataFlowAgent.inputConfig.inputs.outputFormat).toBeDefined();
      expect(DataFlowAgent.inputConfig.inputs.outputFormat.required).toBe(false);

      expect(DataFlowAgent.inputConfig.inputs.validationRules).toBeDefined();
      expect(DataFlowAgent.inputConfig.inputs.validationRules.required).toBe(
        false,
      );
    });

    it('should have appropriate model configuration', () => {
      expect(DataFlowAgent.modelConfig.temp).toBe(0.2);
      expect(DataFlowAgent.modelConfig.top_p).toBe(0.9);
      expect(DataFlowAgent.modelConfig.thinkingBudget).toBe(-1);
    });

    it('should have appropriate run configuration', () => {
      expect(DataFlowAgent.runConfig.max_time_minutes).toBe(5);
      expect(DataFlowAgent.runConfig.max_turns).toBe(25);
    });

    it('should include required tools', () => {
      expect(DataFlowAgent.toolConfig?.tools).toContain('read_file');
      expect(DataFlowAgent.toolConfig?.tools).toContain('glob');
      expect(DataFlowAgent.toolConfig?.tools).toContain('grep');
    });
  });

  describe('Output Schema', () => {
    it('should define output configuration', () => {
      expect(DataFlowAgent.outputConfig).toBeDefined();
      expect(DataFlowAgent.outputConfig?.outputName).toBe('report');
      expect(DataFlowAgent.outputConfig?.schema).toBeDefined();
    });

    it('should validate correct output structure', () => {
      const validOutput = {
        summary: 'Processed 100 records successfully',
        dataSource: {
          type: 'JSON',
          location: 'data/users.json',
          format: 'JSON',
          recordCount: 100,
        },
        transformations: [
          {
            step: 'Extract user data',
            operation: 'extract' as const,
            input: 'Raw JSON',
            output: 'User objects',
            logic: 'Parsed JSON and extracted user fields',
          },
        ],
        dataQuality: {
          completeness: 0.95,
          accuracy: 0.98,
          consistency: 0.92,
          issues: ['Some missing email addresses'],
          recommendations: ['Add email validation'],
        },
        outputSchema: {
          id: 'number',
          name: 'string',
          email: 'string',
        },
        insights: ['Most users are from US', 'Peak registration in Q4'],
      };

      const result = DataFlowAgent.outputConfig?.schema.safeParse(validOutput);
      expect(result?.success).toBe(true);
    });

    it('should reject output with invalid data quality scores', () => {
      const invalidOutput = {
        summary: 'Test',
        dataSource: {
          type: 'JSON',
          location: 'test.json',
          format: 'JSON',
          recordCount: 10,
        },
        transformations: [],
        dataQuality: {
          completeness: 1.5, // Invalid: > 1
          accuracy: 0.98,
          consistency: 0.92,
          issues: [],
          recommendations: [],
        },
        outputSchema: {},
        insights: [],
      };

      const result = DataFlowAgent.outputConfig?.schema.safeParse(invalidOutput);
      expect(result?.success).toBe(false);
    });

    it('should require all mandatory fields in output', () => {
      const incompleteOutput = {
        summary: 'Test',
        dataSource: {
          type: 'JSON',
          location: 'test.json',
          format: 'JSON',
          recordCount: 10,
        },
        // Missing transformations, dataQuality, outputSchema, insights
      };

      const result =
        DataFlowAgent.outputConfig?.schema.safeParse(incompleteOutput);
      expect(result?.success).toBe(false);
    });

    it('should validate transformation operation types', () => {
      const validOperations = [
        'extract',
        'filter',
        'transform',
        'aggregate',
        'validate',
      ];

      validOperations.forEach((operation) => {
        const output = {
          summary: 'Test',
          dataSource: {
            type: 'JSON',
            location: 'test.json',
            format: 'JSON',
            recordCount: 10,
          },
          transformations: [
            {
              step: 'Test step',
              operation,
              input: 'Input',
              output: 'Output',
              logic: 'Logic',
            },
          ],
          dataQuality: {
            completeness: 0.9,
            accuracy: 0.9,
            consistency: 0.9,
            issues: [],
            recommendations: [],
          },
          outputSchema: {},
          insights: [],
        };

        const result = DataFlowAgent.outputConfig?.schema.safeParse(output);
        expect(result?.success).toBe(true);
      });
    });
  });

  describe('processOutput', () => {
    it('should format output as markdown report', () => {
      const output = {
        summary: 'Successfully processed user data',
        dataSource: {
          type: 'JSON',
          location: 'data/users.json',
          format: 'JSON',
          recordCount: 100,
        },
        transformations: [
          {
            step: 'Data extraction',
            operation: 'extract' as const,
            input: 'Raw JSON file',
            output: 'User objects',
            logic: 'Parsed JSON and extracted user fields',
          },
        ],
        dataQuality: {
          completeness: 0.95,
          accuracy: 0.98,
          consistency: 0.92,
          issues: ['Missing email for 5 users'],
          recommendations: ['Add email validation'],
        },
        outputSchema: {
          id: 'number',
          name: 'string',
          email: 'string',
        },
        insights: ['User base is growing', 'Email completion rate is high'],
        nextSteps: ['Validate email addresses', 'Add phone numbers'],
      };

      const result = DataFlowAgent.processOutput?.(output);

      expect(result).toContain('# Data Flow Processing Report');
      expect(result).toContain('## Summary');
      expect(result).toContain('Successfully processed user data');
      expect(result).toContain('## Data Source');
      expect(result).toContain('**Type:** JSON');
      expect(result).toContain('**Location:** data/users.json');
      expect(result).toContain('**Records Processed:** 100');
      expect(result).toContain('## Transformations');
      expect(result).toContain('1. Data extraction');
      expect(result).toContain('**Operation:** extract');
      expect(result).toContain('## Data Quality Assessment');
      expect(result).toContain('**Completeness:** 95.0%');
      expect(result).toContain('**Accuracy:** 98.0%');
      expect(result).toContain('**Consistency:** 92.0%');
      expect(result).toContain('### Issues Identified');
      expect(result).toContain('Missing email for 5 users');
      expect(result).toContain('### Recommendations');
      expect(result).toContain('Add email validation');
      expect(result).toContain('## Output Schema');
      expect(result).toContain('**id:** number');
      expect(result).toContain('**name:** string');
      expect(result).toContain('## Key Insights');
      expect(result).toContain('User base is growing');
      expect(result).toContain('## Next Steps');
      expect(result).toContain('Validate email addresses');
    });

    it('should handle output with minimal data', () => {
      const minimalOutput = {
        summary: 'Basic processing',
        dataSource: {
          type: 'CSV',
          location: 'data.csv',
          format: 'CSV',
          recordCount: 10,
        },
        transformations: [],
        dataQuality: {
          completeness: 1.0,
          accuracy: 1.0,
          consistency: 1.0,
          issues: [],
          recommendations: [],
        },
        outputSchema: {},
        insights: [],
      };

      const result = DataFlowAgent.processOutput?.(minimalOutput);

      expect(result).toContain('# Data Flow Processing Report');
      expect(result).toContain('Basic processing');
      expect(result).toContain('**Completeness:** 100.0%');
    });

    it('should format multiple transformations correctly', () => {
      const output = {
        summary: 'Multi-step processing',
        dataSource: {
          type: 'JSON',
          location: 'data.json',
          format: 'JSON',
          recordCount: 50,
        },
        transformations: [
          {
            step: 'Step 1',
            operation: 'extract' as const,
            input: 'Raw data',
            output: 'Structured data',
            logic: 'Extract fields',
          },
          {
            step: 'Step 2',
            operation: 'filter' as const,
            input: 'Structured data',
            output: 'Filtered data',
            logic: 'Remove invalid records',
          },
          {
            step: 'Step 3',
            operation: 'validate' as const,
            input: 'Filtered data',
            output: 'Valid data',
            logic: 'Apply validation rules',
          },
        ],
        dataQuality: {
          completeness: 0.9,
          accuracy: 0.9,
          consistency: 0.9,
          issues: [],
          recommendations: [],
        },
        outputSchema: {},
        insights: [],
      };

      const result = DataFlowAgent.processOutput?.(output);

      expect(result).toContain('1. Step 1');
      expect(result).toContain('2. Step 2');
      expect(result).toContain('3. Step 3');
      expect(result).toContain('**Operation:** extract');
      expect(result).toContain('**Operation:** filter');
      expect(result).toContain('**Operation:** validate');
    });
  });

  describe('Prompt Configuration', () => {
    it('should include system prompt', () => {
      expect(DataFlowAgent.promptConfig.systemPrompt).toBeDefined();
      expect(DataFlowAgent.promptConfig.systemPrompt).toContain(
        'Data Flow Processing Agent',
      );
      expect(DataFlowAgent.promptConfig.systemPrompt).toContain(
        'Data Extraction',
      );
      expect(DataFlowAgent.promptConfig.systemPrompt).toContain(
        'Data Transformation',
      );
      expect(DataFlowAgent.promptConfig.systemPrompt).toContain(
        'Data Validation',
      );
    });

    it('should include query template', () => {
      expect(DataFlowAgent.promptConfig.query).toBeDefined();
      expect(DataFlowAgent.promptConfig.query).toContain('data_source');
      expect(DataFlowAgent.promptConfig.query).toContain('task');
      expect(DataFlowAgent.promptConfig.query).toContain('outputFormat');
      expect(DataFlowAgent.promptConfig.query).toContain('validationRules');
    });

    it('should provide clear guidelines in system prompt', () => {
      const systemPrompt = DataFlowAgent.promptConfig.systemPrompt;

      expect(systemPrompt).toContain('complete_task');
      expect(systemPrompt).toContain('completeness');
      expect(systemPrompt).toContain('accuracy');
      expect(systemPrompt).toContain('consistency');
    });
  });
});

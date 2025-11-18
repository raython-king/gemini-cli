/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { SummarizerAgent } from '../summarizer-agent.js';

describe('SummarizerAgent', () => {
  describe('Agent Definition', () => {
    it('should have correct agent metadata', () => {
      expect(SummarizerAgent.name).toBe('summarizer_agent');
      expect(SummarizerAgent.displayName).toBe('Summarizer Agent');
      expect(SummarizerAgent.description).toContain(
        'multi-document summarization',
      );
    });

    it('should define required input fields', () => {
      expect(SummarizerAgent.inputConfig.inputs.sources).toBeDefined();
      expect(SummarizerAgent.inputConfig.inputs.sources.required).toBe(true);
      expect(SummarizerAgent.inputConfig.inputs.sources.type).toBe('string[]');
    });

    it('should define optional input fields', () => {
      expect(SummarizerAgent.inputConfig.inputs.style).toBeDefined();
      expect(SummarizerAgent.inputConfig.inputs.style.required).toBe(false);

      expect(SummarizerAgent.inputConfig.inputs.length).toBeDefined();
      expect(SummarizerAgent.inputConfig.inputs.length.required).toBe(false);

      expect(SummarizerAgent.inputConfig.inputs.focus).toBeDefined();
      expect(SummarizerAgent.inputConfig.inputs.focus.required).toBe(false);

      expect(
        SummarizerAgent.inputConfig.inputs.includeActionItems,
      ).toBeDefined();
      expect(SummarizerAgent.inputConfig.inputs.includeActionItems.required).toBe(
        false,
      );
    });

    it('should have appropriate model configuration', () => {
      expect(SummarizerAgent.modelConfig.temp).toBe(0.4);
      expect(SummarizerAgent.modelConfig.top_p).toBe(0.95);
      expect(SummarizerAgent.modelConfig.thinkingBudget).toBe(-1);
    });

    it('should have appropriate run configuration', () => {
      expect(SummarizerAgent.runConfig.max_time_minutes).toBe(5);
      expect(SummarizerAgent.runConfig.max_turns).toBe(25);
    });

    it('should include required tools', () => {
      expect(SummarizerAgent.toolConfig?.tools).toContain('read_file');
      expect(SummarizerAgent.toolConfig?.tools).toContain('glob');
      expect(SummarizerAgent.toolConfig?.tools).toContain('grep');
    });
  });

  describe('Output Schema', () => {
    it('should define output configuration', () => {
      expect(SummarizerAgent.outputConfig).toBeDefined();
      expect(SummarizerAgent.outputConfig?.outputName).toBe('report');
      expect(SummarizerAgent.outputConfig?.schema).toBeDefined();
    });

    it('should validate correct output structure', () => {
      const validOutput = {
        executiveSummary:
          'This report summarizes key findings from three research papers on machine learning.',
        detailedSummary:
          'The papers collectively explore various aspects of machine learning optimization, with particular focus on gradient descent methods and adaptive learning rates. Key themes include convergence analysis, computational efficiency, and practical applications.',
        keyPoints: [
          {
            point: 'Adaptive learning rates improve convergence speed',
            category: 'Optimization',
            importance: 'critical' as const,
            details: 'Multiple studies confirm faster convergence',
            sourceReference: 'Paper 1, Section 3',
          },
        ],
        sections: [
          {
            title: 'Introduction',
            content: 'Overview of the research area',
            keyPoints: ['Background on ML optimization'],
          },
        ],
        themes: ['Optimization', 'Machine Learning', 'Convergence'],
        statistics: {
          sourceCount: 3,
          totalWords: 15000,
          compressionRatio: 0.05,
        },
        actionItems: [
          {
            action: 'Implement adaptive learning rates',
            priority: 'high' as const,
            rationale: 'Demonstrated performance improvements',
            dependencies: ['Framework selection'],
          },
        ],
        gaps: ['Limited evaluation on large-scale datasets'],
        relatedTopics: ['Neural architecture search', 'Transfer learning'],
      };

      const result = SummarizerAgent.outputConfig?.schema.safeParse(validOutput);
      expect(result?.success).toBe(true);
    });

    it('should require mandatory fields', () => {
      const minimalOutput = {
        executiveSummary: 'Executive summary text',
        detailedSummary: 'Detailed summary text',
        keyPoints: [],
        themes: ['Theme 1'],
        statistics: {
          sourceCount: 1,
        },
      };

      const result =
        SummarizerAgent.outputConfig?.schema.safeParse(minimalOutput);
      expect(result?.success).toBe(true);
    });

    it('should validate key point importance levels', () => {
      const validImportance = ['critical', 'high', 'medium', 'low'];

      validImportance.forEach((importance) => {
        const output = {
          executiveSummary: 'Summary',
          detailedSummary: 'Details',
          keyPoints: [
            {
              point: 'Test point',
              category: 'Test category',
              importance,
            },
          ],
          themes: ['Theme'],
          statistics: { sourceCount: 1 },
        };

        const result = SummarizerAgent.outputConfig?.schema.safeParse(output);
        expect(result?.success).toBe(true);
      });
    });

    it('should validate action item priority levels', () => {
      const validPriority = ['high', 'medium', 'low'];

      validPriority.forEach((priority) => {
        const output = {
          executiveSummary: 'Summary',
          detailedSummary: 'Details',
          keyPoints: [],
          themes: ['Theme'],
          statistics: { sourceCount: 1 },
          actionItems: [
            {
              action: 'Test action',
              priority,
              rationale: 'Test rationale',
            },
          ],
        };

        const result = SummarizerAgent.outputConfig?.schema.safeParse(output);
        expect(result?.success).toBe(true);
      });
    });

    it('should reject invalid importance level', () => {
      const output = {
        executiveSummary: 'Summary',
        detailedSummary: 'Details',
        keyPoints: [
          {
            point: 'Test point',
            category: 'Test category',
            importance: 'invalid' as any,
          },
        ],
        themes: ['Theme'],
        statistics: { sourceCount: 1 },
      };

      const result = SummarizerAgent.outputConfig?.schema.safeParse(output);
      expect(result?.success).toBe(false);
    });
  });

  describe('processOutput', () => {
    it('should format output as markdown report', () => {
      const output = {
        executiveSummary:
          'This report summarizes key findings from multiple research papers.',
        detailedSummary:
          'The papers explore machine learning optimization techniques with focus on practical applications.',
        keyPoints: [
          {
            point: 'Adaptive learning rates are effective',
            category: 'Optimization',
            importance: 'critical' as const,
            details: 'Multiple studies confirm this',
            sourceReference: 'Paper 1',
          },
        ],
        sections: [
          {
            title: 'Introduction',
            content: 'Background information',
            keyPoints: ['Context setting', 'Problem definition'],
          },
        ],
        themes: ['Optimization', 'Machine Learning'],
        statistics: {
          sourceCount: 3,
          totalWords: 15000,
          compressionRatio: 0.05,
        },
        actionItems: [
          {
            action: 'Implement new optimizer',
            priority: 'high' as const,
            rationale: 'Performance improvements demonstrated',
            dependencies: ['Framework update'],
          },
        ],
        gaps: ['Limited large-scale evaluation'],
        relatedTopics: ['Neural architecture search'],
      };

      const result = SummarizerAgent.processOutput?.(output);

      expect(result).toContain('# Summarization Report');
      expect(result).toContain('## Executive Summary');
      expect(result).toContain('key findings from multiple research papers');
      expect(result).toContain('## Main Themes');
      expect(result).toContain('1. Optimization');
      expect(result).toContain('2. Machine Learning');
      expect(result).toContain('## Key Points');
      expect(result).toContain('[CRITICAL] Adaptive learning rates are effective');
      expect(result).toContain('**Category:** Optimization');
      expect(result).toContain('Multiple studies confirm this');
      expect(result).toContain('*Source: Paper 1*');
      expect(result).toContain('## Detailed Summary');
      expect(result).toContain('machine learning optimization techniques');
      expect(result).toContain('## Detailed Sections');
      expect(result).toContain('1. Introduction');
      expect(result).toContain('Background information');
      expect(result).toContain('**Key Points:**');
      expect(result).toContain('- Context setting');
      expect(result).toContain('## Recommended Actions');
      expect(result).toContain('[HIGH] Implement new optimizer');
      expect(result).toContain('**Rationale:** Performance improvements demonstrated');
      expect(result).toContain('**Dependencies:**');
      expect(result).toContain('- Framework update');
      expect(result).toContain('## Information Gaps');
      expect(result).toContain('Limited large-scale evaluation');
      expect(result).toContain('## Related Topics for Further Exploration');
      expect(result).toContain('Neural architecture search');
      expect(result).toContain('## Statistics');
      expect(result).toContain('**Sources Processed:** 3');
      expect(result).toContain('**Total Words:** ~15000');
      expect(result).toContain('**Compression Ratio:** 5.0%');
    });

    it('should handle output without optional fields', () => {
      const minimalOutput = {
        executiveSummary: 'Brief executive summary',
        detailedSummary: 'Brief detailed summary',
        keyPoints: [],
        themes: ['Theme 1'],
        statistics: {
          sourceCount: 1,
        },
      };

      const result = SummarizerAgent.processOutput?.(minimalOutput);

      expect(result).toContain('# Summarization Report');
      expect(result).toContain('Brief executive summary');
      expect(result).toContain('Brief detailed summary');
      expect(result).toContain('**Sources Processed:** 1');
      expect(result).not.toContain('## Detailed Sections');
      expect(result).not.toContain('## Recommended Actions');
      expect(result).not.toContain('## Information Gaps');
    });

    it('should format multiple key points correctly', () => {
      const output = {
        executiveSummary: 'Summary',
        detailedSummary: 'Details',
        keyPoints: [
          {
            point: 'Point 1',
            category: 'Category A',
            importance: 'critical' as const,
          },
          {
            point: 'Point 2',
            category: 'Category B',
            importance: 'high' as const,
            details: 'Additional details for point 2',
          },
          {
            point: 'Point 3',
            category: 'Category A',
            importance: 'medium' as const,
            sourceReference: 'Source X',
          },
        ],
        themes: ['Theme'],
        statistics: { sourceCount: 1 },
      };

      const result = SummarizerAgent.processOutput?.(output);

      expect(result).toContain('1. [CRITICAL] Point 1');
      expect(result).toContain('**Category:** Category A');
      expect(result).toContain('2. [HIGH] Point 2');
      expect(result).toContain('**Category:** Category B');
      expect(result).toContain('Additional details for point 2');
      expect(result).toContain('3. [MEDIUM] Point 3');
      expect(result).toContain('*Source: Source X*');
    });

    it('should handle statistics with missing optional fields', () => {
      const output = {
        executiveSummary: 'Summary',
        detailedSummary: 'Details',
        keyPoints: [],
        themes: ['Theme'],
        statistics: {
          sourceCount: 5,
        },
      };

      const result = SummarizerAgent.processOutput?.(output);

      expect(result).toContain('**Sources Processed:** 5');
      expect(result).not.toContain('**Total Words:**');
      expect(result).not.toContain('**Compression Ratio:**');
    });
  });

  describe('Prompt Configuration', () => {
    it('should include system prompt', () => {
      expect(SummarizerAgent.promptConfig.systemPrompt).toBeDefined();
      expect(SummarizerAgent.promptConfig.systemPrompt).toContain(
        'Summarizer Agent',
      );
      expect(SummarizerAgent.promptConfig.systemPrompt).toContain(
        'Information Extraction',
      );
      expect(SummarizerAgent.promptConfig.systemPrompt).toContain('Synthesis');
      expect(SummarizerAgent.promptConfig.systemPrompt).toContain('Structuring');
    });

    it('should include query template', () => {
      expect(SummarizerAgent.promptConfig.query).toBeDefined();
      expect(SummarizerAgent.promptConfig.query).toContain('sources');
      expect(SummarizerAgent.promptConfig.query).toContain('style');
      expect(SummarizerAgent.promptConfig.query).toContain('length');
      expect(SummarizerAgent.promptConfig.query).toContain('focus');
      expect(SummarizerAgent.promptConfig.query).toContain('includeActionItems');
    });

    it('should provide style guidelines', () => {
      const systemPrompt = SummarizerAgent.promptConfig.systemPrompt;

      expect(systemPrompt).toContain('Executive Style');
      expect(systemPrompt).toContain('Technical Style');
      expect(systemPrompt).toContain('Academic Style');
      expect(systemPrompt).toContain('Narrative Style');
    });

    it('should provide length guidelines', () => {
      const systemPrompt = SummarizerAgent.promptConfig.systemPrompt;

      expect(systemPrompt).toContain('Brief:');
      expect(systemPrompt).toContain('Moderate:');
      expect(systemPrompt).toContain('Comprehensive:');
      expect(systemPrompt).toContain('50-100 words');
      expect(systemPrompt).toContain('100-150 words');
      expect(systemPrompt).toContain('150-200 words');
    });

    it('should mention SummarizerReport schema', () => {
      const systemPrompt = SummarizerAgent.promptConfig.systemPrompt;

      expect(systemPrompt).toContain('SummarizerReport');
      expect(systemPrompt).toContain('complete_task');
    });
  });
});

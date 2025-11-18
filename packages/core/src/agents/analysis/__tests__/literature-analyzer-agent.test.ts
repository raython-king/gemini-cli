/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { LiteratureAnalyzerAgent } from '../literature-analyzer-agent.js';

describe('LiteratureAnalyzerAgent', () => {
  describe('Agent Definition', () => {
    it('should have correct agent metadata', () => {
      expect(LiteratureAnalyzerAgent.name).toBe('literature_analyzer_agent');
      expect(LiteratureAnalyzerAgent.displayName).toBe(
        'Literature Analyzer Agent',
      );
      expect(LiteratureAnalyzerAgent.description).toContain(
        'academic and technical literature',
      );
    });

    it('should define required input fields', () => {
      expect(LiteratureAnalyzerAgent.inputConfig.inputs.document).toBeDefined();
      expect(
        LiteratureAnalyzerAgent.inputConfig.inputs.document.required,
      ).toBe(true);
      expect(LiteratureAnalyzerAgent.inputConfig.inputs.document.type).toBe(
        'string',
      );
    });

    it('should define optional input fields', () => {
      expect(LiteratureAnalyzerAgent.inputConfig.inputs.focus).toBeDefined();
      expect(LiteratureAnalyzerAgent.inputConfig.inputs.focus.required).toBe(
        false,
      );

      expect(LiteratureAnalyzerAgent.inputConfig.inputs.depth).toBeDefined();
      expect(LiteratureAnalyzerAgent.inputConfig.inputs.depth.required).toBe(
        false,
      );

      expect(
        LiteratureAnalyzerAgent.inputConfig.inputs.compareTo,
      ).toBeDefined();
      expect(
        LiteratureAnalyzerAgent.inputConfig.inputs.compareTo.required,
      ).toBe(false);
    });

    it('should have appropriate model configuration', () => {
      expect(LiteratureAnalyzerAgent.modelConfig.temp).toBe(0.3);
      expect(LiteratureAnalyzerAgent.modelConfig.top_p).toBe(0.95);
      expect(LiteratureAnalyzerAgent.modelConfig.thinkingBudget).toBe(-1);
    });

    it('should have appropriate run configuration', () => {
      expect(LiteratureAnalyzerAgent.runConfig.max_time_minutes).toBe(15);
      expect(LiteratureAnalyzerAgent.runConfig.max_turns).toBe(30);
    });

    it('should include required tools', () => {
      expect(LiteratureAnalyzerAgent.toolConfig?.tools).toContain('read_file');
      expect(LiteratureAnalyzerAgent.toolConfig?.tools).toContain('glob');
      expect(LiteratureAnalyzerAgent.toolConfig?.tools).toContain('grep');
      expect(LiteratureAnalyzerAgent.toolConfig?.tools).toContain('web_search');
    });
  });

  describe('Output Schema', () => {
    it('should define output configuration', () => {
      expect(LiteratureAnalyzerAgent.outputConfig).toBeDefined();
      expect(LiteratureAnalyzerAgent.outputConfig?.outputName).toBe('report');
      expect(LiteratureAnalyzerAgent.outputConfig?.schema).toBeDefined();
    });

    it('should validate correct output structure', () => {
      const validOutput = {
        summary:
          'This paper explores machine learning optimization techniques...',
        citation: {
          authors: ['John Doe', 'Jane Smith'],
          title: 'Advances in ML Optimization',
          year: 2024,
          source: 'Journal of AI Research',
          url: 'https://example.com/paper',
        },
        mainThemes: [
          'Gradient descent optimization',
          'Learning rate scheduling',
          'Convergence analysis',
        ],
        keyConcepts: [
          {
            concept: 'Adaptive learning rates',
            definition: 'Learning rates that adjust during training',
            relevance: 'high' as const,
            relatedConcepts: ['Adam', 'RMSprop'],
          },
        ],
        methodology: {
          approach: 'Empirical evaluation',
          methods: ['Benchmark testing', 'Ablation studies'],
          dataSource: 'ImageNet dataset',
          limitations: ['Limited to vision tasks', 'Computational cost'],
        },
        keyFindings: [
          {
            finding: 'Adaptive methods converge faster',
            evidence: 'Experiments showed 2x speedup',
            significance: 'critical' as const,
            implications: ['Reduced training time', 'Lower compute costs'],
          },
        ],
        criticalAnalysis: {
          strengths: ['Rigorous methodology', 'Comprehensive experiments'],
          weaknesses: ['Limited to specific domains', 'Small sample size'],
          contributions: [
            'New optimization algorithm',
            'Theoretical analysis',
          ],
        },
        relatedWork: [
          {
            authors: ['Alice Brown'],
            title: 'Learning Rate Schedules',
            year: 2023,
          },
        ],
        futureDirections: [
          'Extend to other domains',
          'Theoretical convergence proofs',
        ],
        practicalApplications: ['Computer vision', 'NLP tasks'],
      };

      const result =
        LiteratureAnalyzerAgent.outputConfig?.schema.safeParse(validOutput);
      expect(result?.success).toBe(true);
    });

    it('should require mandatory fields', () => {
      const incompleteOutput = {
        summary: 'Test summary',
        mainThemes: ['Theme 1'],
        keyConcepts: [],
        keyFindings: [],
        criticalAnalysis: {
          strengths: [],
          weaknesses: [],
          contributions: [],
        },
        // Missing citation (optional) is fine
      };

      const result =
        LiteratureAnalyzerAgent.outputConfig?.schema.safeParse(
          incompleteOutput,
        );
      expect(result?.success).toBe(true);
    });

    it('should validate concept relevance levels', () => {
      const validRelevance = ['high', 'medium', 'low'];

      validRelevance.forEach((relevance) => {
        const output = {
          summary: 'Test',
          mainThemes: ['Theme'],
          keyConcepts: [
            {
              concept: 'Test Concept',
              definition: 'Definition',
              relevance,
            },
          ],
          keyFindings: [],
          criticalAnalysis: {
            strengths: [],
            weaknesses: [],
            contributions: [],
          },
        };

        const result =
          LiteratureAnalyzerAgent.outputConfig?.schema.safeParse(output);
        expect(result?.success).toBe(true);
      });
    });

    it('should validate finding significance levels', () => {
      const validSignificance = ['critical', 'important', 'moderate', 'minor'];

      validSignificance.forEach((significance) => {
        const output = {
          summary: 'Test',
          mainThemes: ['Theme'],
          keyConcepts: [],
          keyFindings: [
            {
              finding: 'Test finding',
              evidence: 'Evidence',
              significance,
            },
          ],
          criticalAnalysis: {
            strengths: [],
            weaknesses: [],
            contributions: [],
          },
        };

        const result =
          LiteratureAnalyzerAgent.outputConfig?.schema.safeParse(output);
        expect(result?.success).toBe(true);
      });
    });

    it('should reject invalid significance level', () => {
      const output = {
        summary: 'Test',
        mainThemes: ['Theme'],
        keyConcepts: [],
        keyFindings: [
          {
            finding: 'Test finding',
            evidence: 'Evidence',
            significance: 'invalid' as any,
          },
        ],
        criticalAnalysis: {
          strengths: [],
          weaknesses: [],
          contributions: [],
        },
      };

      const result =
        LiteratureAnalyzerAgent.outputConfig?.schema.safeParse(output);
      expect(result?.success).toBe(false);
    });
  });

  describe('processOutput', () => {
    it('should format output as markdown report', () => {
      const output = {
        summary:
          'This paper presents novel approaches to machine learning optimization.',
        citation: {
          authors: ['John Doe', 'Jane Smith'],
          title: 'Advances in ML Optimization',
          year: 2024,
          source: 'Journal of AI Research',
          url: 'https://example.com/paper',
        },
        mainThemes: [
          'Gradient descent optimization',
          'Learning rate scheduling',
        ],
        keyConcepts: [
          {
            concept: 'Adaptive learning rates',
            definition: 'Learning rates that adjust during training',
            relevance: 'high' as const,
            relatedConcepts: ['Adam', 'RMSprop'],
          },
        ],
        keyFindings: [
          {
            finding: 'Adaptive methods converge faster',
            evidence: 'Experiments showed 2x speedup',
            significance: 'critical' as const,
            implications: ['Reduced training time'],
          },
        ],
        criticalAnalysis: {
          strengths: ['Rigorous methodology'],
          weaknesses: ['Limited domain'],
          contributions: ['New optimization algorithm'],
        },
        futureDirections: ['Extend to other domains'],
        practicalApplications: ['Computer vision'],
      };

      const result = LiteratureAnalyzerAgent.processOutput?.(output);

      expect(result).toContain('# Literature Analysis Report');
      expect(result).toContain('## Summary');
      expect(result).toContain('novel approaches to machine learning');
      expect(result).toContain('## Citation');
      expect(result).toContain('**Title:** Advances in ML Optimization');
      expect(result).toContain('**Authors:** John Doe, Jane Smith');
      expect(result).toContain('**Year:** 2024');
      expect(result).toContain('## Main Themes');
      expect(result).toContain('1. Gradient descent optimization');
      expect(result).toContain('## Key Concepts');
      expect(result).toContain('Adaptive learning rates');
      expect(result).toContain('Relevance: high');
      expect(result).toContain('**Related Concepts:** Adam, RMSprop');
      expect(result).toContain('## Key Findings');
      expect(result).toContain('Adaptive methods converge faster');
      expect(result).toContain('**Evidence:** Experiments showed 2x speedup');
      expect(result).toContain('## Critical Analysis');
      expect(result).toContain('### Strengths');
      expect(result).toContain('Rigorous methodology');
      expect(result).toContain('### Weaknesses');
      expect(result).toContain('Limited domain');
      expect(result).toContain('## Future Research Directions');
      expect(result).toContain('Extend to other domains');
      expect(result).toContain('## Practical Applications');
      expect(result).toContain('Computer vision');
    });

    it('should handle output without optional fields', () => {
      const minimalOutput = {
        summary: 'Brief summary of the paper',
        mainThemes: ['Theme 1', 'Theme 2'],
        keyConcepts: [],
        keyFindings: [],
        criticalAnalysis: {
          strengths: ['Strength 1'],
          weaknesses: ['Weakness 1'],
          contributions: ['Contribution 1'],
        },
      };

      const result = LiteratureAnalyzerAgent.processOutput?.(minimalOutput);

      expect(result).toContain('# Literature Analysis Report');
      expect(result).toContain('Brief summary of the paper');
      expect(result).not.toContain('## Citation');
      expect(result).not.toContain('## Key Concepts');
      expect(result).toContain('## Critical Analysis');
    });

    it('should format related work citations correctly', () => {
      const output = {
        summary: 'Test',
        mainThemes: ['Theme'],
        keyConcepts: [],
        keyFindings: [],
        criticalAnalysis: {
          strengths: [],
          weaknesses: [],
          contributions: [],
        },
        relatedWork: [
          {
            authors: ['Alice Brown', 'Bob Green'],
            title: 'Related Paper 1',
            year: 2023,
            source: 'Conference on AI',
            url: 'https://example.com/related1',
          },
          {
            authors: ['Charlie White'],
            title: 'Related Paper 2',
          },
        ],
      };

      const result = LiteratureAnalyzerAgent.processOutput?.(output);

      expect(result).toContain('## Related Work');
      expect(result).toContain(
        '1. Alice Brown, Bob Green. "Related Paper 1" (2023). Conference on AI [Link](https://example.com/related1)',
      );
      expect(result).toContain('2. Charlie White. "Related Paper 2"');
    });

    it('should format multiple key concepts with details', () => {
      const output = {
        summary: 'Test',
        mainThemes: ['Theme'],
        keyConcepts: [
          {
            concept: 'Concept A',
            definition: 'Definition of A',
            relevance: 'high' as const,
            relatedConcepts: ['Related 1', 'Related 2'],
          },
          {
            concept: 'Concept B',
            definition: 'Definition of B',
            relevance: 'medium' as const,
          },
        ],
        keyFindings: [],
        criticalAnalysis: {
          strengths: [],
          weaknesses: [],
          contributions: [],
        },
      };

      const result = LiteratureAnalyzerAgent.processOutput?.(output);

      expect(result).toContain('1. Concept A (Relevance: high)');
      expect(result).toContain('Definition of A');
      expect(result).toContain('**Related Concepts:** Related 1, Related 2');
      expect(result).toContain('2. Concept B (Relevance: medium)');
      expect(result).toContain('Definition of B');
      expect(result).not.toContain('**Related Concepts:**\n\n### 2.');
    });
  });

  describe('Prompt Configuration', () => {
    it('should include system prompt', () => {
      expect(LiteratureAnalyzerAgent.promptConfig.systemPrompt).toBeDefined();
      expect(
        LiteratureAnalyzerAgent.promptConfig.systemPrompt,
      ).toContain('Literature Analyzer Agent');
      expect(
        LiteratureAnalyzerAgent.promptConfig.systemPrompt,
      ).toContain('Content Analysis');
      expect(
        LiteratureAnalyzerAgent.promptConfig.systemPrompt,
      ).toContain('Methodology Analysis');
      expect(LiteratureAnalyzerAgent.promptConfig.systemPrompt).toContain(
        'Critical Evaluation',
      );
    });

    it('should include query template with dynamic sections', () => {
      expect(LiteratureAnalyzerAgent.promptConfig.query).toBeDefined();
      expect(LiteratureAnalyzerAgent.promptConfig.query).toContain('document');
      expect(LiteratureAnalyzerAgent.promptConfig.query).toContain('focus');
      expect(LiteratureAnalyzerAgent.promptConfig.query).toContain('depth');
      expect(LiteratureAnalyzerAgent.promptConfig.query).toContain('compareTo');
    });

    it('should provide depth-based strategy guidelines', () => {
      const systemPrompt = LiteratureAnalyzerAgent.promptConfig.systemPrompt;

      expect(systemPrompt).toContain('Overview Depth');
      expect(systemPrompt).toContain('Detailed Depth');
      expect(systemPrompt).toContain('Comprehensive Depth');
      expect(systemPrompt).toContain('5-10 turns');
      expect(systemPrompt).toContain('15-20 turns');
      expect(systemPrompt).toContain('25-30 turns');
    });

    it('should mention LiteratureAnalysisReport schema', () => {
      const systemPrompt = LiteratureAnalyzerAgent.promptConfig.systemPrompt;

      expect(systemPrompt).toContain('LiteratureAnalysisReport');
      expect(systemPrompt).toContain('complete_task');
    });
  });
});

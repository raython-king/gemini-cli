# Data Flow & Literature Analysis System Guide

This guide explains how to use the advanced multi-agent analysis system for
automated data processing and literature analysis in the Gemini CLI.

## Table of Contents

1. [Overview](#overview)
2. [Core Agents](#core-agents)
3. [Quick Start](#quick-start)
4. [Data Pipeline System](#data-pipeline-system)
5. [Analysis Workflows](#analysis-workflows)
6. [Context Engineering Integration](#context-engineering-integration)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

## Overview

The Analysis System provides intelligent agents for:

- **Automated Data Processing**: Extract, transform, validate, and analyze data
- **Literature Analysis**: Deep analysis of academic papers and technical
  documents
- **Multi-Document Summarization**: Synthesize information from multiple sources
- **Context-Aware Orchestration**: Maintain continuity across analysis tasks

### Architecture

```
┌──────────────────────────────────────────────────────┐
│          Analysis System Architecture                │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Specialized Agents:                                 │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ DataFlow   │  │ Literature   │  │ Summarizer  │ │
│  │ Agent      │  │ Analyzer     │  │ Agent       │ │
│  └─────┬──────┘  └──────┬───────┘  └──────┬──────┘ │
│        │                │                  │         │
│        └────────────────┴──────────────────┘         │
│                         │                            │
│  Orchestration:  ┌──────▼──────────┐                │
│                  │ Analysis        │                │
│                  │ Orchestrator    │                │
│                  └─────────────────┘                │
│                         │                            │
│  Infrastructure: ┌──────▼──────────┐                │
│                  │ Context         │                │
│                  │ Engineering     │                │
│                  │ (Shared Context,│                │
│                  │  Agent Memory)  │                │
│                  └─────────────────┘                │
└──────────────────────────────────────────────────────┘
```

## Core Agents

### 1. DataFlowAgent

Specializes in automated data flow processing:

**Capabilities:**

- Extract data from various sources (JSON, CSV, logs, code files)
- Transform and clean data
- Validate data quality
- Create processing pipelines
- Identify patterns and anomalies

**When to use:**

- Processing structured or semi-structured data
- ETL (Extract, Transform, Load) operations
- Data quality assessment
- Automated data cleaning
- Pattern recognition in datasets

**Example:**

```typescript
import { DataFlowAgent } from '@google/gemini-cli-core';

// Use with agent executor
const result = await executeAgent(DataFlowAgent, {
  dataSource: 'logs/user-activity.json',
  task: 'Extract user engagement metrics and identify patterns',
  outputFormat: 'structured',
  validationRules: ['completeness', 'consistency'],
});
```

### 2. LiteratureAnalyzerAgent

Specializes in deep analysis of academic and technical literature:

**Capabilities:**

- Analyze research papers and technical documents
- Extract key concepts, methodologies, and findings
- Identify themes and patterns
- Provide critical analysis
- Extract citations and build reference networks

**When to use:**

- Analyzing research papers
- Understanding complex technical documentation
- Conducting literature reviews
- Extracting knowledge from academic sources
- Synthesizing research findings

**Example:**

```typescript
import { LiteratureAnalyzerAgent } from '@google/gemini-cli-core';

const result = await executeAgent(LiteratureAnalyzerAgent, {
  document: 'papers/ml-optimization.pdf',
  focus: 'methodology',
  depth: 'comprehensive',
});
```

### 3. SummarizerAgent

Specializes in intelligent multi-document summarization:

**Capabilities:**

- Generate executive and detailed summaries
- Extract key points and insights
- Synthesize information from multiple sources
- Create structured, hierarchical summaries
- Provide actionable recommendations

**When to use:**

- Creating concise summaries of long documents
- Synthesizing information from multiple sources
- Generating executive summaries
- Organizing complex information
- Extracting key insights

**Example:**

```typescript
import { SummarizerAgent } from '@google/gemini-cli-core';

const result = await executeAgent(SummarizerAgent, {
  sources: ['docs/architecture.md', 'docs/api-design.md', 'docs/deployment.md'],
  style: 'executive',
  length: 'moderate',
  includeActionItems: true,
});
```

## Quick Start

### Basic Data Processing

```typescript
import {
  DataFlowAgent,
  executeAgent,
  DataPipeline,
  Transforms,
  Validators,
} from '@google/gemini-cli-core';

// 1. Simple data processing with agent
const result = await executeAgent(DataFlowAgent, {
  dataSource: 'data/users.json',
  task: 'Extract active users and calculate statistics',
});

console.log('Summary:', result.summary);
console.log('Data Quality:', result.dataQuality);
console.log('Insights:', result.insights);

// 2. Using data pipeline for custom processing
const pipeline = new DataPipeline()
  .addStep({
    name: 'parse',
    description: 'Parse JSON data',
    transform: Transforms.parseJSON,
  })
  .addStep({
    name: 'filter',
    description: 'Filter active users',
    transform: Transforms.filter((user: any) => user.active),
    validate: Validators.nonEmpty,
  })
  .addStep({
    name: 'extract',
    description: 'Extract user metrics',
    transform: Transforms.extractFields(['id', 'name', 'loginCount']),
  });

const pipelineResult = await pipeline.execute(rawData);
console.log('Processed data:', pipelineResult.data);
```

### Basic Literature Analysis

```typescript
import {
  LiteratureAnalyzerAgent,
  SummarizerAgent,
  executeAgent,
} from '@google/gemini-cli-core';

// 1. Analyze a single paper
const analysis = await executeAgent(LiteratureAnalyzerAgent, {
  document: 'research/paper1.pdf',
  depth: 'detailed',
});

console.log('Summary:', analysis.summary);
console.log('Key Concepts:', analysis.keyConcepts);
console.log('Findings:', analysis.keyFindings);

// 2. Synthesize multiple papers
const synthesis = await executeAgent(SummarizerAgent, {
  sources: [
    'research/paper1.pdf',
    'research/paper2.pdf',
    'research/paper3.pdf',
  ],
  style: 'academic',
  focus: ['methodology', 'results'],
});

console.log('Executive Summary:', synthesis.executiveSummary);
console.log('Key Points:', synthesis.keyPoints);
```

## Data Pipeline System

The DataPipeline provides a composable, type-safe way to process data:

### Creating Pipelines

```typescript
import { DataPipeline, Transforms, Validators } from '@google/gemini-cli-core';

const pipeline = new DataPipeline()
  .addStep({
    name: 'load',
    transform: async (filePath: string) => {
      const fs = await import('fs/promises');
      return await fs.readFile(filePath, 'utf-8');
    },
  })
  .addStep({
    name: 'parse',
    transform: Transforms.parseJSON,
  })
  .addStep({
    name: 'validate',
    transform: (data) => data,
    validate: Validators.requiredFields(['id', 'timestamp', 'event']),
  })
  .addStep({
    name: 'group',
    transform: Transforms.groupBy((item: any) => item.event),
  });

const result = await pipeline.execute('logs/events.json');

if (result.success) {
  console.log('Processed data:', result.data);
  console.log('Execution time:', result.metadata.executionTime);
} else {
  console.error('Pipeline errors:', result.errors);
}
```

### Built-in Transforms

```typescript
// Data extraction
const extractedData = Transforms.extractFields(['name', 'email', 'status'])(
  users,
);

// Filtering
const activeUsers = Transforms.filter((u) => u.active)(users);

// Mapping
const userNames = Transforms.map((u) => u.name)(users);

// Grouping
const byRole = Transforms.groupBy((u) => u.role)(users);

// Deduplication
const unique = Transforms.deduplicate((u) => u.email)(users);

// Sorting
const sorted = Transforms.sort((a, b) => a.timestamp - b.timestamp)(events);
```

### Built-in Validators

```typescript
// Non-empty validation
const validateNotEmpty = Validators.nonEmpty;

// Required fields
const validateUser = Validators.requiredFields(['id', 'email', 'name']);

// Type validation
const validateArray = Validators.type('array', Array.isArray);

// Custom validation
const validateEmail = Validators.custom(
  (user: any) => user.email.includes('@'),
  'Invalid email format',
);

// Combine validators
const validateAll = Validators.all(
  validateUser,
  validateEmail,
  validateNotEmpty,
);
```

## Analysis Workflows

The AnalysisOrchestrator provides pre-built workflows for common analysis tasks:

### Data Processing Workflow

```typescript
import {
  AnalysisOrchestrator,
  AnalysisWorkflowType,
} from '@google/gemini-cli-core';

const orchestrator = new AnalysisOrchestrator();

const result = await orchestrator.executeAnalysis({
  type: AnalysisWorkflowType.DATA_PROCESSING,
  description: 'Process user activity logs and generate insights',
  sources: ['logs/user-activity-2024.json'],
  options: {
    dataValidation: ['completeness', 'consistency', 'accuracy'],
  },
});

console.log('Summary:', result.summary);
console.log('Key Findings:', result.keyFindings);
console.log('Recommendations:', result.recommendations);
console.log('Agents Used:', result.metadata.agentsUsed);
```

### Literature Review Workflow

```typescript
const litReview = await orchestrator.executeAnalysis({
  type: AnalysisWorkflowType.LITERATURE_REVIEW,
  description: 'Analyze recent ML optimization research',
  sources: [
    'papers/adam-optimizer.pdf',
    'papers/sgd-variants.pdf',
    'papers/learning-rate-schedules.pdf',
  ],
  options: {
    depth: 'comprehensive',
    focus: ['methodology', 'results', 'future work'],
  },
});

console.log('Summary:', litReview.summary);
console.log('Key Findings:', litReview.keyFindings);
```

### Comprehensive Analysis Workflow

Combines data processing and literature review:

```typescript
const comprehensive = await orchestrator.executeAnalysis({
  type: AnalysisWorkflowType.COMPREHENSIVE_ANALYSIS,
  description: 'Analyze user behavior data and relevant research',
  sources: [
    'data/user-behavior.json',
    'research/user-engagement-patterns.pdf',
    'research/retention-strategies.pdf',
  ],
  options: {
    depth: 'comprehensive',
    includeActionItems: true,
  },
});
```

### Multi-Document Summary Workflow

```typescript
const summary = await orchestrator.executeAnalysis({
  type: AnalysisWorkflowType.MULTI_DOCUMENT_SUMMARY,
  description: 'Summarize product documentation',
  sources: [
    'docs/getting-started.md',
    'docs/api-reference.md',
    'docs/best-practices.md',
    'docs/troubleshooting.md',
  ],
  options: {
    depth: 'moderate',
    includeActionItems: true,
  },
});
```

## Context Engineering Integration

The analysis system fully integrates with context engineering for enhanced
capabilities:

### Automatic Context Sharing

Agents automatically share insights through SharedContext:

```typescript
import {
  AnalysisOrchestrator,
  AnalysisWorkflowType,
} from '@google/gemini-cli-core';

const orchestrator = new AnalysisOrchestrator();

// First analysis - insights stored in context
await orchestrator.executeAnalysis({
  type: AnalysisWorkflowType.DATA_PROCESSING,
  description: 'Analyze Q1 sales data',
  sources: ['data/q1-sales.json'],
});

// Second analysis - benefits from Q1 insights
await orchestrator.executeAnalysis({
  type: AnalysisWorkflowType.DATA_PROCESSING,
  description: 'Compare Q2 sales to Q1 patterns',
  sources: ['data/q2-sales.json'],
  // Automatically accesses Q1 patterns from shared context
});
```

### Long-term Memory

Important findings persist across sessions:

```typescript
// Analyze multiple documents
for (const paper of researchPapers) {
  await orchestrator.executeAnalysis({
    type: AnalysisWorkflowType.LITERATURE_REVIEW,
    description: `Analyze ${paper}`,
    sources: [paper],
  });
}

// Consolidate important findings to long-term memory
orchestrator.consolidateMemories(0.7);

// Later, these memories inform new analyses
const synthesis = await orchestrator.executeAnalysis({
  type: AnalysisWorkflowType.MULTI_DOCUMENT_SUMMARY,
  description: 'Synthesize all research findings',
  sources: researchPapers,
  // Benefits from memories of individual paper analyses
});
```

### Accessing Analysis History

```typescript
// Get all previous analyses
const allHistory = orchestrator.getAnalysisHistory();

// Get specific workflow type history
const dataHistory = orchestrator.getAnalysisHistory(
  AnalysisWorkflowType.DATA_PROCESSING,
);

console.log('Previous data processing tasks:', dataHistory);
```

### Exporting Analysis State

```typescript
const state = orchestrator.exportAnalysisState();

console.log('Workflows executed:', state.workflows);
console.log('Insights discovered:', state.insights);
console.log('Patterns identified:', state.patterns);
console.log('Context & memory stats:', state.state.stats);
```

## Best Practices

### 1. Choose the Right Agent

- **DataFlowAgent**: Use for structured/semi-structured data processing
- **LiteratureAnalyzerAgent**: Use for academic/technical document analysis
- **SummarizerAgent**: Use for creating concise summaries
- **AnalysisOrchestrator**: Use for complex multi-agent workflows

### 2. Data Processing Guidelines

```typescript
// Good: Clear, specific task
await executeAgent(DataFlowAgent, {
  dataSource: 'logs/errors.json',
  task: 'Extract error patterns, group by type, and identify trends',
  validationRules: ['completeness', 'timestamp_validity'],
});

// Better: Use pipelines for reusable logic
const errorAnalysisPipeline = new DataPipeline()
  .addStep({ name: 'parse', transform: Transforms.parseJSON })
  .addStep({
    name: 'filter',
    transform: Transforms.filter((e) => e.severity === 'ERROR'),
  })
  .addStep({
    name: 'group',
    transform: Transforms.groupBy((e) => e.type),
  });
```

### 3. Literature Analysis Tips

```typescript
// Start with overview, then detailed analysis
const overview = await executeAgent(LiteratureAnalyzerAgent, {
  document: 'paper.pdf',
  depth: 'overview',
});

// Deep dive on specific aspects
const detailed = await executeAgent(LiteratureAnalyzerAgent, {
  document: 'paper.pdf',
  depth: 'comprehensive',
  focus: 'methodology',
});
```

### 4. Leverage Context

```typescript
// Let agents learn from each other
const orchestrator = new AnalysisOrchestrator();

// Process related tasks in sequence
await orchestrator.executeAnalysis({
  /* task 1 */
});
await orchestrator.executeAnalysis({
  /* task 2 - benefits from task 1 */
});
await orchestrator.executeAnalysis({
  /* task 3 - benefits from tasks 1 & 2 */
});

// Periodically consolidate memories
orchestrator.consolidateMemories(0.7);

// Clean up old context
orchestrator.cleanup({
  maxContextAge: 3600000, // 1 hour
  minMemoryImportance: 0.3,
});
```

### 5. Error Handling

```typescript
try {
  const result = await pipeline.execute(data);

  if (!result.success) {
    console.error('Pipeline failed:');
    result.errors.forEach((err) => {
      console.error(`  Step ${err.step}: ${err.message}`);
    });

    // Check validation results
    Object.entries(result.metadata.validationResults).forEach(
      ([step, validation]) => {
        if (!validation.valid) {
          console.error(`  Validation failed at ${step}:`);
          validation.errors.forEach((e) =>
            console.error(`    - ${e.field}: ${e.message}`),
          );
        }
      },
    );
  }
} catch (error) {
  console.error('Unexpected error:', error);
}
```

## Examples

### Example 1: User Activity Analysis

```typescript
import {
  DataFlowAgent,
  SummarizerAgent,
  AnalysisOrchestrator,
  AnalysisWorkflowType,
} from '@google/gemini-cli-core';

const orchestrator = new AnalysisOrchestrator();

const result = await orchestrator.executeAnalysis({
  type: AnalysisWorkflowType.DATA_PROCESSING,
  description: 'Analyze user activity and identify engagement patterns',
  sources: ['data/user-activity-2024.json'],
  options: {
    dataValidation: ['completeness', 'consistency'],
  },
});

console.log('=== User Activity Analysis ===');
console.log('Summary:', result.summary);
console.log('\nKey Findings:');
result.keyFindings.forEach((finding) => console.log(`  - ${finding}`));
console.log('\nRecommendations:');
result.recommendations.forEach((rec) => console.log(`  - ${rec}`));
```

### Example 2: Research Paper Synthesis

```typescript
const papers = [
  'research/transformer-architecture.pdf',
  'research/attention-mechanisms.pdf',
  'research/bert-pretraining.pdf',
];

const orchestrator = new AnalysisOrchestrator();

const result = await orchestrator.executeAnalysis({
  type: AnalysisWorkflowType.LITERATURE_REVIEW,
  description: 'Synthesize transformer research findings',
  sources: papers,
  options: {
    depth: 'comprehensive',
    focus: ['architecture', 'training', 'performance'],
  },
});

console.log('=== Research Synthesis ===');
console.log('Executive Summary:', result.summary);
console.log('\nKey Contributions:');
result.keyFindings.forEach((finding) => console.log(`  - ${finding}`));
```

### Example 3: Custom Data Pipeline

```typescript
import { DataPipeline, Transforms, Validators } from '@google/gemini-cli-core';

// Define a reusable pipeline for log analysis
const logAnalysisPipeline = new DataPipeline()
  .addStep({
    name: 'parse',
    description: 'Parse JSON log entries',
    transform: Transforms.parseJSON,
  })
  .addStep({
    name: 'filter-errors',
    description: 'Filter error-level logs',
    transform: Transforms.filter(
      (log: any) => log.level === 'ERROR' || log.level === 'CRITICAL',
    ),
    validate: Validators.nonEmpty,
  })
  .addStep({
    name: 'extract-fields',
    description: 'Extract relevant fields',
    transform: Transforms.extractFields([
      'timestamp',
      'level',
      'message',
      'stack',
    ]),
  })
  .addStep({
    name: 'group-by-type',
    description: 'Group errors by type',
    transform: Transforms.groupBy((log: any) => log.message.split(':')[0]),
  });

// Process logs
const result = await logAnalysisPipeline.execute(rawLogData);

console.log('=== Log Analysis ===');
console.log(`Processed in ${result.metadata.executionTime}ms`);
console.log('Error types:', Object.keys(result.data || {}));
```

### Example 4: Continuous Analysis

```typescript
const orchestrator = new AnalysisOrchestrator();

// Analyze daily reports
const dailyReports = [
  'reports/monday.md',
  'reports/tuesday.md',
  'reports/wednesday.md',
  'reports/thursday.md',
  'reports/friday.md',
];

for (const report of dailyReports) {
  await orchestrator.executeAnalysis({
    type: AnalysisWorkflowType.MULTI_DOCUMENT_SUMMARY,
    description: `Analyze ${report}`,
    sources: [report],
  });
}

// Generate weekly summary (benefits from all daily analyses)
const weekly = await orchestrator.executeAnalysis({
  type: AnalysisWorkflowType.MULTI_DOCUMENT_SUMMARY,
  description: 'Generate weekly summary from daily reports',
  sources: dailyReports,
  options: {
    depth: 'comprehensive',
    includeActionItems: true,
  },
});

// Consolidate important findings
orchestrator.consolidateMemories(0.8);

// Export state for persistence
const state = orchestrator.exportAnalysisState();
console.log('Weekly insights:', state.insights);
```

## Conclusion

The Analysis System provides powerful tools for automated data processing and
literature analysis:

1. **Specialized Agents**: DataFlowAgent, LiteratureAnalyzerAgent,
   SummarizerAgent
2. **Composable Pipelines**: Type-safe data transformations
3. **Pre-built Workflows**: Common analysis patterns
4. **Context Engineering**: Maintain continuity and build knowledge

By combining these components, you can build sophisticated analysis workflows
that learn from experience and maintain comprehensive understanding across
multiple tasks.

For more information, see:

- [Context Engineering Guide](./CONTEXT_ENGINEERING_GUIDE.md)
- [Multi-Agent Guide](./MULTI_AGENT_GUIDE.md)
- Analysis examples in `docs/analysis-examples.ts`

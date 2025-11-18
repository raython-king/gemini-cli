# Analysis System

This directory contains the advanced analysis system for automated data flow
processing and literature analysis.

## Status

**Current Status**: ✅ Production Ready - All agents are fully integrated and
registered

The analysis system includes:

### ✅ Completed Components

1. **DataPipeline** (`data-pipeline.ts`) - Production Ready
   - Composable data processing pipelines
   - Built-in transforms (filter, map, group, sort, etc.)
   - Built-in validators (required fields, type checking, etc.)
   - Type-safe transformations
   - Validation with error handling

2. **DataFlowAgent** (`data-flow-agent.ts`) - Production Ready
   - Automated data extraction, transformation, validation
   - Data quality assessment with completeness, accuracy, and consistency metrics
   - Pattern recognition and insight generation
   - Multi-step transformation pipeline tracking
   - Comprehensive data processing reports
   - **Registered in AgentRegistry as `data_flow_agent`**

3. **LiteratureAnalyzerAgent** (`literature-analyzer-agent.ts`) - Production
   Ready
   - Deep analysis of academic and technical literature
   - Concept extraction with relevance scoring
   - Methodology and findings analysis
   - Critical evaluation (strengths, weaknesses, contributions)
   - Citation extraction and reference network building
   - Configurable depth levels (overview, detailed, comprehensive)
   - **Registered in AgentRegistry as `literature_analyzer_agent`**

4. **SummarizerAgent** (`summarizer-agent.ts`) - Production Ready
   - Multi-document summarization with executive and detailed summaries
   - Key point extraction with importance levels and categories
   - Theme and pattern identification
   - Actionable recommendations with priorities
   - Flexible summarization styles (executive, technical, academic, narrative)
   - Configurable length (brief, moderate, comprehensive)
   - Structured output with statistics and metrics
   - **Registered in AgentRegistry as `summarizer_agent`**

5. **AnalysisOrchestrator** (`analysis-orchestrator.ts`) - Production Ready
   - Context-aware coordination of analysis agents
   - Pre-built workflows (data processing, literature review, comprehensive
     analysis, multi-document summary)
   - Integration with context engineering system
   - Parallel agent execution support
   - Memory consolidation and state export

## Using the Analysis System

### DataPipeline

```typescript
import { DataPipeline, Transforms, Validators } from '@google/gemini-cli-core';

const pipeline = new DataPipeline()
  .addStep({
    name: 'parse',
    transform: Transforms.parseJSON,
  })
  .addStep({
    name: 'filter',
    transform: Transforms.filter((item: any) => item.active),
    validate: Validators.nonEmpty,
  })
  .addStep({
    name: 'sort',
    transform: Transforms.sort((a: any, b: any) => b.score - a.score),
  });

const result = await pipeline.execute(data);
```

### Analysis Agents

```typescript
import {
  DataFlowAgent,
  LiteratureAnalyzerAgent,
  SummarizerAgent,
  AgentExecutor,
} from '@google/gemini-cli-core';

// Data processing
const dataExecutor = await AgentExecutor.create(DataFlowAgent, config);
const dataResult = await dataExecutor.run(
  {
    dataSource: 'data/users.json',
    task: 'Extract and validate user data',
    validationRules: 'email must be valid',
  },
  signal,
);

// Literature analysis
const litExecutor = await AgentExecutor.create(LiteratureAnalyzerAgent, config);
const litResult = await litExecutor.run(
  {
    document: 'papers/research.pdf',
    depth: 'detailed',
    focus: 'methodology',
  },
  signal,
);

// Multi-document summarization
const summaryExecutor = await AgentExecutor.create(SummarizerAgent, config);
const summaryResult = await summaryExecutor.run(
  {
    sources: ['doc1.md', 'doc2.md'],
    style: 'executive',
    includeActionItems: true,
  },
  signal,
);
```

### AnalysisOrchestrator

```typescript
import {
  AnalysisOrchestrator,
  AnalysisWorkflowType,
} from '@google/gemini-cli-core';

const orchestrator = new AnalysisOrchestrator(config);

const result = await orchestrator.executeAnalysis({
  type: AnalysisWorkflowType.LITERATURE_REVIEW,
  description: 'Analyze ML optimization papers',
  sources: ['paper1.pdf', 'paper2.pdf'],
  options: {
    depth: 'comprehensive',
    focus: ['methodology', 'results'],
  },
});
```

## Documentation

- See `/MULTI_AGENT_GUIDE.md` for comprehensive documentation on all agents
- All agents are automatically registered and available via AgentRegistry

## Contributing

When adapting these agents, refer to existing agents like `ExploreAgent` for the
correct structure.

# Analysis System (In Development)

This directory contains the advanced analysis system for automated data flow
processing and literature analysis.

## Status

**Current Status**: Development / Prototype

The analysis system includes:

### ✅ Completed Components

1. **DataPipeline** (`data-pipeline.ts`) - Fully functional
   - Composable data processing pipelines
   - Built-in transforms (filter, map, group, sort, etc.)
   - Built-in validators (required fields, type checking, etc.)
   - Type-safe transformations
   - Validation with error handling

### 🚧 In Development

The following agents are designed and prototyped but need to be adapted to the
latest `AgentDefinition` structure:

1. **DataFlowAgent** (`data-flow-agent.ts`)
   - Automated data extraction, transformation, validation
   - Data quality assessment
   - Pattern recognition

2. **LiteratureAnalyzerAgent** (`literature-analyzer-agent.ts`)
   - Deep analysis of academic and technical literature
   - Concept extraction
   - Methodology analysis
   - Critical evaluation

3. **SummarizerAgent** (`summarizer-agent.ts`)
   - Multi-document summarization
   - Key point extraction
   - Actionable recommendations

4. **AnalysisOrchestrator** (`analysis-orchestrator.ts`)
   - Context-aware coordination of analysis agents
   - Pre-built workflows (data processing, literature review, etc.)
   - Integration with context engineering

## Using the DataPipeline

The DataPipeline is ready to use today:

\`\`\`typescript import { DataPipeline, Transforms, Validators } from
'@google/gemini-cli-core';

const pipeline = new DataPipeline() .addStep({ name: 'parse', transform:
Transforms.parseJSON, }) .addStep({ name: 'filter', transform:
Transforms.filter((item: any) => item.active), validate: Validators.nonEmpty, })
.addStep({ name: 'sort', transform: Transforms.sort((a: any, b: any) =>
b.score - a.score), });

const result = await pipeline.execute(data); \`\`\`

## Documentation

- See `../../../docs/ANALYSIS_SYSTEM_GUIDE.md` for comprehensive documentation
- See `../../../docs/analysis-examples.ts` for usage examples

## Future Work

To complete the analysis agents:

1. Adapt agent definitions to use:
   - `inputConfig` instead of `taskDefinitionSchema`
   - `outputConfig` instead of `reportSchema`
   - `modelConfig`, `runConfig`, `toolConfig`, `promptConfig`

2. Implement actual agent execution (currently simulated)

3. Add comprehensive tests

4. Integrate with agent registry

## Contributing

When adapting these agents, refer to existing agents like `ExploreAgent` for the
correct structure.

# Multi-Agent System Guide

## Overview

Gemini CLI now features a powerful multi-agent system inspired by Claude Code,
enabling specialized agents to collaborate on complex coding tasks. This system
includes 9 specialized agents that can work independently or in parallel to
handle different aspects of software development.

## Available Agents

### 1. **Plan Agent** (`plan_agent`)

**Purpose**: Analyzes requirements and creates detailed implementation plans
before coding.

**Use Cases**:

- Breaking down complex features into actionable steps
- Risk assessment before implementation
- Identifying dependencies and affected files
- Creating roadmaps for large refactorings

**Example Usage**:

```typescript
import { PlanAgent, AgentExecutor } from '@google/gemini-cli-core';

const executor = await AgentExecutor.create(
  PlanAgent,
  runtimeContext,
  activityCallback,
);

const result = await executor.run(
  {
    task: 'Implement user authentication with JWT tokens',
    context: 'Using Express.js backend and React frontend',
  },
  abortSignal,
);

console.log(result.result); // Detailed implementation plan
```

**Output Structure**:

- Summary: Executive overview of the approach
- Steps: Ordered list with complexity, dependencies, and affected files
- Risks: Potential issues with severity levels and mitigation strategies
- Dependencies: External libraries, APIs, or services needed
- Estimated Duration: Time estimate for implementation

---

### 2. **Explore Agent** (`explore_agent`)

**Purpose**: Fast codebase exploration and answering questions about code
structure.

**Use Cases**:

- Finding where a feature is implemented
- Understanding how components interact
- Locating specific functionality
- Getting architectural insights quickly

**Thoroughness Levels**:

- `quick`: 3-5 turns, surface-level insights
- `medium`: 5-8 turns, balanced exploration (default)
- `thorough`: 8-12 turns, comprehensive analysis

**Example Usage**:

```typescript
import { ExploreAgent, AgentExecutor } from '@google/gemini-cli-core';

const executor = await AgentExecutor.create(ExploreAgent, runtimeContext);

const result = await executor.run(
  {
    query: 'Where is user authentication implemented?',
    thoroughness: 'medium',
  },
  abortSignal,
);
```

**Output Structure**:

- Answer: Direct answer to the query
- KeyFindings: 3-5 important insights
- RelevantLocations: Code locations with file paths, line ranges, and symbols
- ArchitectureInsights: High-level design observations
- Recommendations: Next steps or things to investigate

---

### 3. **Code Reviewer Agent** (`code_reviewer`)

**Purpose**: Automated code review and quality assessment.

**Use Cases**:

- Reviewing code before commits
- Identifying bugs, security vulnerabilities, performance issues
- Ensuring code quality and maintainability
- Enforcing best practices and coding standards

**Review Dimensions**:

- Correctness & Bugs
- Security (OWASP Top 10)
- Performance
- Maintainability
- Testing
- Documentation
- Best Practices

**Example Usage**:

```typescript
import { CodeReviewerAgent, AgentExecutor } from '@google/gemini-cli-core';

const executor = await AgentExecutor.create(CodeReviewerAgent, runtimeContext);

// Review specific files
const result = await executor.run(
  {
    files: 'src/auth.ts, src/utils/validation.ts',
    focus: 'security', // or "performance", "testing", "all"
  },
  abortSignal,
);

// Or review a git diff
const result2 = await executor.run(
  {
    diff: gitDiffOutput,
    focus: 'all',
  },
  abortSignal,
);
```

**Output Structure**:

- Summary: Overview of review findings
- QualityMetrics: Overall score (0-10), strengths, areas for improvement
- Issues: Categorized by severity (CRITICAL, HIGH, MEDIUM, LOW, INFO)
  - Each issue includes: file, line, category, description, suggestion
- SecurityConcerns: Specific security vulnerabilities
- TestingRecommendations: Testing suggestions
- Approved: Boolean indicating if code is ready for merging

---

### 4. **Test Runner Agent** (`test_runner`)

**Purpose**: Automated test execution and fixing.

**Use Cases**:

- Running test suites and reporting results
- Analyzing test failures and identifying root causes
- Automatically fixing failing tests
- Ensuring code changes don't break existing functionality

**Example Usage**:

```typescript
import { TestRunnerAgent, AgentExecutor } from '@google/gemini-cli-core';

const executor = await AgentExecutor.create(TestRunnerAgent, runtimeContext);

const result = await executor.run(
  {
    testCommand: 'npm test',
    maxAttempts: 3, // Max fix attempts per test
    targetFiles: 'tests/auth/*.test.ts', // Optional: specific test files
  },
  abortSignal,
);
```

**Output Structure**:

- Summary: Test execution and fix summary
- InitialResults: Test results before fixes
- FinalResults: Test results after fix attempts
- Failures: List of test failures with root cause analysis
- FixesApplied: Number of successful fixes
- AllTestsPassing: Boolean indicating if all tests now pass
- Recommendations: Suggestions for manual investigation if needed

---

### 5. **Debug Agent** (`debug_agent`)

**Purpose**: Systematic bug debugging and fixing.

**Use Cases**:

- Investigating reported bugs
- Reproducing issues from user reports
- Identifying root causes of failures
- Applying fixes and verifying they work

**Debugging Process**:

1. Reproduce the bug
2. Isolate the problem area
3. Identify root cause
4. Apply fix
5. Verify fix works

**Example Usage**:

```typescript
import { DebugAgent, AgentExecutor } from '@google/gemini-cli-core';

const executor = await AgentExecutor.create(DebugAgent, runtimeContext);

const result = await executor.run(
  {
    bugDescription: "User login fails with 'null is not an object' error",
    reproduceSteps:
      '1. Navigate to /login\n2. Enter credentials\n3. Click login button',
    context: 'Only happens on iOS Safari browser',
  },
  abortSignal,
);
```

**Output Structure**:

- Summary: Debugging session overview
- BugReproduced: Whether the bug was successfully reproduced
- DebugSteps: Step-by-step debugging process
- RootCause: Detailed root cause analysis with location and explanation
- Fix: Description of the fix applied
- PreventionRecommendations: How to prevent similar bugs
- UnresolvedIssues: Issues that need further investigation
- BugFixed: Whether the bug was successfully fixed

---

### 6. **Refactor Agent** (`refactor_agent`)

**Purpose**: Code refactoring and quality improvement while preserving
functionality.

**Use Cases**:

- Improving code readability and maintainability
- Reducing code duplication
- Simplifying complex code
- Extracting functions/classes for better structure
- Optimizing performance

**Refactoring Strategies**:

- `improve_readability`: Focus on naming, formatting, clarity
- `reduce_complexity`: Simplify complex functions, reduce nesting
- `remove_duplication`: DRY principle, extract common code
- `extract_components`: Break down large functions/classes
- `modernize`: Update to modern patterns/syntax
- `performance`: Optimize hot paths
- `comprehensive`: All of the above (default)

**Example Usage**:

```typescript
import { RefactorAgent, AgentExecutor } from '@google/gemini-cli-core';

const executor = await AgentExecutor.create(RefactorAgent, runtimeContext);

const result = await executor.run(
  {
    target: 'src/utils/validation.ts',
    strategy: 'comprehensive',
    testCommand: 'npm test', // Ensures refactoring doesn't break functionality
  },
  abortSignal,
);
```

**Output Structure**:

- Summary: Refactoring work performed
- Strategy: Approach used
- Operations: List of refactoring operations with affected files
- QualityImprovements: Measurable improvements in code quality
- TestsRan: Whether tests were run
- TestsPassed: Whether all tests passed
- BreakingChanges: Any breaking changes (should be none for pure refactoring)
- Success: Whether the refactoring was successful

---

### 7. **Data Flow Agent** (`data_flow_agent`)

**Purpose**: Automated data flow processing and transformation with quality
assessment.

**Use Cases**:

- Processing and transforming data from various sources
- Extracting structured information from unstructured data
- Validating data quality and identifying issues
- Creating data processing pipelines
- Aggregating and analyzing data patterns
- Converting between data formats

**Example Usage**:

```typescript
import { DataFlowAgent, AgentExecutor } from '@google/gemini-cli-core';

const executor = await AgentExecutor.create(DataFlowAgent, runtimeContext);

const result = await executor.run(
  {
    dataSource: 'data/users.json',
    task: 'Extract user information and validate email formats',
    outputFormat: 'JSON',
    validationRules: 'email must be valid, age > 0, name required',
  },
  abortSignal,
);
```

**Output Structure**:

- Summary: High-level summary of data processing results
- DataSource: Information about the data source (type, location, format, record
  count)
- Transformations: List of transformations applied with detailed logic
- DataQuality: Assessment with completeness, accuracy, consistency scores
- OutputSchema: Schema of the output data
- Insights: Key insights discovered during data processing
- NextSteps: Recommended next steps for data pipeline improvement

---

### 8. **Literature Analyzer Agent** (`literature_analyzer_agent`)

**Purpose**: Deep analysis of academic and technical literature.

**Use Cases**:

- Analyzing research papers and technical documents
- Extracting key concepts, methodologies, and findings
- Understanding complex academic or technical content
- Identifying themes across multiple documents
- Synthesizing information from various sources
- Evaluating research quality and contributions
- Extracting and organizing citations

**Depth Levels**:

- `overview`: High-level summary and main points (5-10 turns)
- `detailed`: Thorough analysis with key insights (15-20 turns, default)
- `comprehensive`: Deep dive with extensive cross-referencing (25-30 turns)

**Example Usage**:

```typescript
import {
  LiteratureAnalyzerAgent,
  AgentExecutor,
} from '@google/gemini-cli-core';

const executor = await AgentExecutor.create(
  LiteratureAnalyzerAgent,
  runtimeContext,
);

const result = await executor.run(
  {
    document: 'papers/ml-architecture-2024.md',
    focus: 'methodology',
    depth: 'detailed',
    compareTo: ['papers/related-work-2023.md'],
  },
  abortSignal,
);
```

**Output Structure**:

- Summary: Comprehensive summary (200-300 words)
- Citation: Citation information if available
- MainThemes: 3-5 main themes or topics covered
- KeyConcepts: Important concepts with definitions and relevance levels
- Methodology: Research approach, methods, data sources, limitations
- KeyFindings: Most important findings with evidence and significance
- CriticalAnalysis: Strengths, weaknesses, and key contributions
- RelatedWork: Related works cited or referenced
- FutureDirections: Suggested future research directions
- PracticalApplications: Practical applications or use cases

---

### 9. **Summarizer Agent** (`summarizer_agent`)

**Purpose**: Intelligent multi-document summarization and synthesis.

**Use Cases**:

- Generating concise summaries of long documents
- Synthesizing information from multiple sources
- Extracting key points and insights
- Creating executive summaries for quick overview
- Organizing complex information hierarchically
- Identifying main themes and patterns
- Generating actionable recommendations

**Summarization Styles**:

- `executive`: Focus on high-level insights and implications (default)
- `technical`: Include specific details and methodologies
- `academic`: Emphasize research findings and evidence
- `narrative`: Create engaging, story-like flow

**Summary Lengths**:

- `brief`: Maximum compression, 50-250 words total
- `moderate`: Balanced approach, 300-650 words total (default)
- `comprehensive`: Minimal information loss, 500-1000 words total

**Example Usage**:

```typescript
import { SummarizerAgent, AgentExecutor } from '@google/gemini-cli-core';

const executor = await AgentExecutor.create(SummarizerAgent, runtimeContext);

const result = await executor.run(
  {
    sources: ['docs/report1.md', 'docs/report2.md', 'docs/report3.md'],
    style: 'executive',
    length: 'moderate',
    focus: ['performance', 'scalability'],
    includeActionItems: true,
  },
  abortSignal,
);
```

**Output Structure**:

- ExecutiveSummary: High-level overview (100-150 words)
- DetailedSummary: Comprehensive coverage (300-500 words)
- KeyPoints: Most important points with categories and importance levels
- Sections: Organized sections for structured content
- Themes: Main themes or topics identified
- Statistics: Summary statistics (source count, word count, compression ratio)
- ActionItems: Actionable recommendations with priorities
- Gaps: Information gaps or missing pieces
- RelatedTopics: Related topics worth exploring

---

## Parallel Execution

Use `ParallelAgentExecutor` to run multiple agents concurrently:

```typescript
import {
  ParallelAgentExecutor,
  ExploreAgent,
  CodeReviewerAgent,
} from '@google/gemini-cli-core';

const parallelExecutor = new ParallelAgentExecutor(
  runtimeContext,
  3, // Max concurrency
);

const tasks = [
  {
    id: 'explore-auth',
    agentName: 'explore_agent',
    agentDefinition: ExploreAgent,
    inputs: { query: 'How does authentication work?', thoroughness: 'medium' },
    priority: 1,
  },
  {
    id: 'review-auth',
    agentName: 'code_reviewer',
    agentDefinition: CodeReviewerAgent,
    inputs: { files: 'src/auth/**/*.ts', focus: 'security' },
    priority: 2,
  },
];

const results = await parallelExecutor.executeParallel(
  tasks,
  (taskId, event) => {
    console.log(`Task ${taskId}:`, event);
  },
);

results.forEach((result) => {
  if (result.success) {
    console.log(`${result.agentName} completed in ${result.durationMs}ms`);
  } else {
    console.error(`${result.agentName} failed:`, result.error);
  }
});
```

---

## Agent Orchestrator

Use `AgentOrchestrator` for intelligent agent selection:

```typescript
import { AgentOrchestrator } from '@google/gemini-cli-core';

const orchestrator = new AgentOrchestrator();

// Classify a user request
const classification = orchestrator.classifyTask(
  'Implement user authentication with proper security',
);

console.log('Task Type:', classification.type); // 'implementation'
console.log('Complexity:', classification.complexity); // 'high'
console.log('Requires Plan:', classification.requiresPlan); // true
console.log('Recommended Agents:', classification.recommendedAgents); // ['plan_agent']

// Create an execution strategy
const strategy = orchestrator.createStrategy(
  'Review my authentication code and run tests',
);

console.log('Use Plan Mode:', strategy.usePlanMode); // false
console.log('Agents to invoke:', strategy.agents); // [code_reviewer, test_runner]
console.log('Run in parallel:', strategy.parallel); // true
```

**Task Types**:

- `PLANNING`: Creating implementation plans
- `EXPLORATION`: Understanding code structure
- `CODE_REVIEW`: Reviewing code quality
- `TESTING`: Running and fixing tests
- `DEBUGGING`: Finding and fixing bugs
- `REFACTORING`: Improving code quality
- `IMPLEMENTATION`: Building new features

**Complexity Levels**:

- `LOW`: Single file, simple change
- `MEDIUM`: Multiple files, moderate complexity
- `HIGH`: System-wide changes, complex logic
- `VERY_HIGH`: Architecture changes, major refactoring

---

## Best Practices

### 1. **Use Plan Agent for Complex Tasks**

Before implementing complex features, use Plan Agent to create a roadmap:

```typescript
// 1. Create a plan
const planResult = await planExecutor.run(
  {
    task: 'Add real-time notifications to the app',
    context: 'Using WebSockets and Redis',
  },
  signal,
);

// 2. Review the plan
console.log(planResult.result);

// 3. Implement step-by-step, potentially using other agents
```

### 2. **Parallel Execution for Independent Tasks**

When tasks don't depend on each other, run them in parallel:

```typescript
// ✅ Good: Parallel execution for independent reviews
parallelExecutor.executeParallel([
  {
    /* Review frontend code */
  },
  {
    /* Review backend code */
  },
  {
    /* Run tests */
  },
]);

// ❌ Bad: Sequential execution when parallel would work
```

### 3. **Test Runner After Refactoring**

Always verify refactorings don't break functionality:

```typescript
// 1. Refactor code
const refactorResult = await refactorExecutor.run(
  {
    target: 'src/legacy-module.ts',
    strategy: 'modernize',
    testCommand: 'npm test',
  },
  signal,
);

// 2. Check if tests passed
if (!refactorResult.result.TestsPassed) {
  console.error('Refactoring broke tests!');
}
```

### 4. **Code Review Before Committing**

Review code for issues before creating commits:

```typescript
// Review staged changes
const gitDiff = execSync('git diff --staged').toString();

const reviewResult = await codeReviewerExecutor.run(
  {
    diff: gitDiff,
    focus: 'all',
  },
  signal,
);

const review = JSON.parse(reviewResult.result);

if (!review.Approved) {
  console.log('Fix these issues before committing:');
  review.Issues.forEach((issue) => console.log(`- ${issue.Title}`));
}
```

### 5. **Explore Before Implementing**

Understand existing code before making changes:

```typescript
// 1. Explore how existing system works
const exploreResult = await exploreExecutor.run(
  {
    query: 'How does the current notification system work?',
    thoroughness: 'thorough',
  },
  signal,
);

// 2. Use insights to inform implementation
// 3. Implement new feature
```

---

## Configuration

Agents are registered and enabled by default in `AgentRegistry`. They can be
individually enabled/disabled in future versions via configuration.

Current agents:

- ✅ `codebase_investigator` (original deep analysis agent)
- ✅ `plan_agent` (NEW)
- ✅ `explore_agent` (NEW - faster alternative to codebase_investigator)
- ✅ `code_reviewer` (NEW)
- ✅ `test_runner` (NEW)
- ✅ `debug_agent` (NEW)
- ✅ `refactor_agent` (NEW)
- ✅ `data_flow_agent` (NEW - data processing and transformation)
- ✅ `literature_analyzer_agent` (NEW - academic and technical literature
  analysis)
- ✅ `summarizer_agent` (NEW - multi-document summarization)

---

## Advanced Usage

### Custom Activity Callbacks

Monitor agent progress in real-time:

```typescript
const activityCallback = (event: SubagentActivityEvent) => {
  switch (event.type) {
    case 'TOOL_CALL_START':
      console.log(`Agent calling tool: ${event.data.toolName}`);
      break;
    case 'TOOL_CALL_END':
      console.log(`Tool completed: ${event.data.toolName}`);
      break;
    case 'THOUGHT_CHUNK':
      process.stdout.write(event.data.thought);
      break;
    case 'ERROR':
      console.error(`Agent error: ${event.data.error}`);
      break;
  }
};

const executor = await AgentExecutor.create(
  PlanAgent,
  runtimeContext,
  activityCallback,
);
```

### Abort Signals

Cancel agent execution if needed:

```typescript
const abortController = new AbortController();

// Set a timeout
setTimeout(() => {
  abortController.abort('Took too long');
}, 30000); // 30 seconds

const result = await executor.run(inputs, abortController.signal);
```

### Error Handling

```typescript
try {
  const result = await executor.run(inputs, signal);

  if (result.terminate_reason === 'GOAL') {
    console.log('Success:', result.result);
  } else {
    console.warn('Agent terminated:', result.terminate_reason);
  }
} catch (error) {
  console.error('Agent execution failed:', error);
}
```

---

## Comparison with Claude Code

| Feature             | Gemini CLI Multi-Agent       | Claude Code  |
| ------------------- | ---------------------------- | ------------ |
| Specialized Agents  | 9 agents                     | ~10+ agents  |
| Parallel Execution  | ✅ Via ParallelAgentExecutor | ✅ Built-in  |
| Plan Mode           | ✅ Via Plan Agent            | ✅ Built-in  |
| Code Review         | ✅ Via Code Reviewer Agent   | ✅ Built-in  |
| Test Runner         | ✅ Via Test Runner Agent     | ✅ Built-in  |
| Debugging           | ✅ Via Debug Agent           | ✅ Built-in  |
| Refactoring         | ✅ Via Refactor Agent        | ✅ Built-in  |
| Data Processing     | ✅ Via Data Flow Agent       | ⚠️ Limited   |
| Literature Analysis | ✅ Via Literature Analyzer   | ⚠️ Limited   |
| Summarization       | ✅ Via Summarizer Agent      | ⚠️ Limited   |
| Agent Orchestration | ✅ Via AgentOrchestrator     | ✅ Automatic |
| UI Integration      | 🚧 Planned                   | ✅ Built-in  |

---

## Troubleshooting

### Agent Takes Too Long

Agents have configurable timeouts (`max_time_minutes` in agent definition). You
can also use abort signals:

```typescript
const abortController = new AbortController();
setTimeout(() => abortController.abort(), 60000); // 1 minute

const result = await executor.run(inputs, abortController.signal);
```

### Agent Produces Poor Results

Try:

1. Provide more context in inputs
2. Use a more thorough exploration level (for Explore Agent)
3. Use CodebaseInvestigator instead of Explore for deep analysis
4. Adjust the agent's prompt configuration

### Compilation Errors

Ensure all agent definitions are properly exported and registered in
`AgentRegistry`.

---

## Future Enhancements

Planned features:

- [ ] UI components for agent monitoring (AgentMonitor, PlanDisplay)
- [ ] Plan Mode integration into main CLI workflow
- [ ] Configuration options to enable/disable individual agents
- [ ] Agent performance metrics and logging
- [ ] Dynamic agent creation and registration
- [ ] Cross-agent communication beyond tool outputs
- [ ] Persistent agent state and resumption

---

## Contributing

To add a new specialized agent:

1. Create agent definition in `packages/core/src/agents/your-agent.ts`
2. Define input/output schemas using Zod
3. Write comprehensive system prompt
4. Register in `AgentRegistry` (`packages/core/src/agents/registry.ts`)
5. Export from `packages/core/src/index.ts`
6. Update this documentation

See existing agents (Plan Agent, Explore Agent, etc.) as examples.

---

## License

Copyright 2025 Google LLC. Licensed under Apache-2.0.

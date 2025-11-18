# Analysis Agents Test Suite

This directory contains comprehensive unit tests for the analysis agents.

## Test Files

### 1. data-flow-agent.test.ts
Tests for the Data Flow Agent which specializes in:
- Data extraction and transformation
- Data quality validation
- Data processing pipelines
- Schema validation

**Test Coverage:**
- Agent definition and metadata
- Input configuration validation
- Output schema validation
- Data quality scoring (completeness, accuracy, consistency)
- Transformation operations (extract, filter, transform, aggregate, validate)
- Markdown report formatting
- Prompt configuration

### 2. literature-analyzer-agent.test.ts
Tests for the Literature Analyzer Agent which specializes in:
- Academic and technical literature analysis
- Key concept extraction
- Methodology analysis
- Critical evaluation

**Test Coverage:**
- Agent definition and metadata
- Input configuration (document, focus, depth, compareTo)
- Output schema validation
- Citation information handling
- Concept relevance levels (high, medium, low)
- Finding significance levels (critical, important, moderate, minor)
- Markdown report formatting with sections
- Depth-based strategy guidelines

### 3. summarizer-agent.test.ts
Tests for the Summarizer Agent which specializes in:
- Multi-document summarization
- Key point extraction
- Executive summary generation
- Action item recommendations

**Test Coverage:**
- Agent definition and metadata
- Input configuration (sources, style, length, focus)
- Output schema validation
- Key point importance levels (critical, high, medium, low)
- Action item priority levels (high, medium, low)
- Markdown report formatting
- Statistics calculation (compression ratio, word count)
- Style guidelines (executive, technical, academic, narrative)

### 4. analysis-orchestrator.test.ts
Tests for the Analysis Orchestrator which coordinates:
- Data processing workflows
- Literature review workflows
- Comprehensive analysis workflows
- Multi-document summary workflows

**Test Coverage:**
- Orchestrator initialization and configuration
- Agent registration (default and custom)
- Workflow type execution
- Context and memory management
- Analysis history tracking
- State export functionality
- Task options handling (depth, focus, validation)
- Error handling
- Results synthesis

## Running Tests

### Run All Analysis Agent Tests
```bash
# From project root
npm test -- packages/core/src/agents/analysis/__tests__

# From packages/core
cd packages/core
npm test -- analysis/__tests__
```

### Run Specific Test File
```bash
# From project root
npm test -- packages/core/src/agents/analysis/__tests__/data-flow-agent.test.ts

# From packages/core
cd packages/core
npm test -- analysis/__tests__/data-flow-agent.test.ts
```

### Run Tests in Watch Mode
```bash
# From packages/core
cd packages/core
npm test -- --watch analysis/__tests__
```

### Run Tests with Coverage
```bash
# From packages/core
cd packages/core
npm test -- --coverage analysis/__tests__
```

## Test Patterns Used

### 1. Agent Definition Tests
Verify that each agent has:
- Correct name and display name
- Comprehensive description
- Required and optional input fields
- Appropriate model configuration (temperature, top_p)
- Appropriate run configuration (max_turns, max_time_minutes)
- Required tools configured

### 2. Output Schema Tests
Verify that:
- Output schema is properly defined
- Valid outputs pass schema validation
- Invalid outputs fail schema validation
- All enum values are correctly validated
- Required fields are enforced
- Optional fields are handled correctly

### 3. Output Processing Tests
Verify that:
- processOutput generates proper markdown format
- All sections are included in output
- Conditional sections only appear when data is present
- Numbers are formatted correctly (percentages, etc.)
- Lists and arrays are formatted properly

### 4. Prompt Configuration Tests
Verify that:
- System prompts include key capabilities
- Query templates include all input variables
- Guidelines are comprehensive
- Schema references are correct

## Mocking Strategy

The tests use vitest's mocking capabilities to:
- Mock AgentExecutor to avoid actual agent execution
- Mock ParallelAgentExecutor for workflow tests
- Isolate unit functionality from dependencies
- Test error handling scenarios

## Test Data Examples

Each test file includes realistic test data that demonstrates:
- Typical use cases
- Edge cases
- Error scenarios
- Various configuration options

## Best Practices

1. **Test Isolation**: Each test is independent and doesn't rely on other tests
2. **Clear Descriptions**: Test descriptions clearly explain what is being tested
3. **Comprehensive Coverage**: Tests cover happy paths, edge cases, and error scenarios
4. **Realistic Data**: Test data reflects real-world usage patterns
5. **Schema Validation**: All outputs are validated against their schemas

## Maintenance

When modifying agents:
1. Update corresponding tests to reflect changes
2. Add new tests for new functionality
3. Ensure all tests pass before committing
4. Update this README if test patterns change

## Continuous Integration

These tests are automatically run in CI/CD pipelines to ensure:
- Code quality
- No regressions
- Compatibility with other components

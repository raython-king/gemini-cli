/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { AgentDefinition } from './types.js';
import {
  GLOB_TOOL_NAME,
  GREP_TOOL_NAME,
  LS_TOOL_NAME,
  READ_FILE_TOOL_NAME,
  SHELL_TOOL_NAME,
  EDIT_TOOL_NAME,
  WRITE_FILE_TOOL_NAME,
} from '../tools/tool-names.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { z } from 'zod';

/**
 * Schema for a test failure.
 */
const TestFailureSchema = z.object({
  TestName: z.string().describe('Name of the failing test.'),
  File: z.string().describe('Test file path.'),
  ErrorMessage: z.string().describe('The error or assertion failure message.'),
  StackTrace: z.string().optional().describe('Stack trace if available.'),
  RootCause: z.string().optional().describe('Analysis of why the test failed.'),
  FixAttempted: z
    .boolean()
    .describe('Whether a fix was attempted for this failure.'),
  FixDescription: z
    .string()
    .optional()
    .describe('Description of the fix applied.'),
});

/**
 * Schema for test execution results.
 */
const TestResultsSchema = z.object({
  TotalTests: z.number().describe('Total number of tests executed.'),
  Passed: z.number().describe('Number of tests that passed.'),
  Failed: z.number().describe('Number of tests that failed.'),
  Skipped: z.number().optional().describe('Number of tests skipped.'),
  Duration: z.string().optional().describe('Total execution time.'),
});

/**
 * Complete test runner report schema.
 */
const TestRunnerReportSchema = z.object({
  Summary: z
    .string()
    .describe('Executive summary of test execution and fixes.'),
  InitialResults: TestResultsSchema.describe('Results before any fixes.'),
  FinalResults: TestResultsSchema.describe(
    'Results after fix attempts (if any).',
  ),
  Failures: z
    .array(TestFailureSchema)
    .describe('List of test failures with analysis.'),
  FixesApplied: z.number().describe('Number of fixes successfully applied.'),
  AllTestsPassing: z.boolean().describe('Whether all tests are now passing.'),
  Recommendations: z
    .array(z.string())
    .optional()
    .describe('Recommendations for further investigation or improvements.'),
});

/**
 * TestRunner Agent - Automated test execution and fixing.
 *
 * This agent runs tests, analyzes failures, attempts fixes, and re-runs tests
 * until they pass or the maximum attempts are exhausted.
 */
export const TestRunnerAgent: AgentDefinition<typeof TestRunnerReportSchema> = {
  name: 'test_runner',
  displayName: 'Test Runner Agent',
  description: `A specialized agent for running tests and fixing test failures.
    Use this agent to:
    - Execute test suites and report results
    - Analyze test failures and identify root causes
    - Automatically fix failing tests
    - Verify that fixes work by re-running tests
    - Ensure code changes don't break existing functionality

    The agent iteratively runs tests, analyzes failures, applies fixes,
    and re-runs until all tests pass or max attempts are reached.`,

  inputConfig: {
    inputs: {
      testCommand: {
        description: `The command to run tests. Examples:
          - "npm test"
          - "npm run test:unit"
          - "pytest tests/"
          - "go test ./..."
          - "cargo test"`,
        type: 'string',
        required: true,
      },
      maxAttempts: {
        description: `Maximum number of fix attempts per failing test.
          Default: 3`,
        type: 'integer',
        required: false,
      },
      targetFiles: {
        description: `Optional: Specific test files or patterns to run.
          If not provided, runs the full test command.
          Example: "tests/auth/*.test.ts"`,
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description:
      'Test execution report with results, failures, and fixes applied.',
    schema: TestRunnerReportSchema,
  },

  processOutput: (output) => {
    let result = `# Test Runner Report\n\n`;
    result += `## Summary\n${output.Summary}\n\n`;

    result += `## Test Results\n\n`;
    result += `### Initial Run\n`;
    result += `- **Total:** ${output.InitialResults.TotalTests}\n`;
    result += `- **Passed:** ${output.InitialResults.Passed} ✅\n`;
    result += `- **Failed:** ${output.InitialResults.Failed} ❌\n`;
    if (output.InitialResults.Skipped) {
      result += `- **Skipped:** ${output.InitialResults.Skipped} ⏭️\n`;
    }
    if (output.InitialResults.Duration) {
      result += `- **Duration:** ${output.InitialResults.Duration}\n`;
    }
    result += `\n`;

    if (output.FixesApplied > 0) {
      result += `### After Fixes (${output.FixesApplied} fixes applied)\n`;
      result += `- **Total:** ${output.FinalResults.TotalTests}\n`;
      result += `- **Passed:** ${output.FinalResults.Passed} ✅\n`;
      result += `- **Failed:** ${output.FinalResults.Failed} ❌\n`;
      if (output.FinalResults.Skipped) {
        result += `- **Skipped:** ${output.FinalResults.Skipped} ⏭️\n`;
      }
      if (output.FinalResults.Duration) {
        result += `- **Duration:** ${output.FinalResults.Duration}\n`;
      }
      result += `\n`;
    }

    result += `**Status:** ${output.AllTestsPassing ? '✅ ALL TESTS PASSING' : '❌ TESTS STILL FAILING'}\n\n`;

    if (output.Failures && output.Failures.length > 0) {
      result += `## Test Failures\n\n`;
      output.Failures.forEach((failure, idx) => {
        result += `### ${idx + 1}. ${failure.TestName}\n`;
        result += `**File:** ${failure.File}\n`;
        result += `**Error:** ${failure.ErrorMessage}\n`;
        if (failure.RootCause) {
          result += `**Root Cause:** ${failure.RootCause}\n`;
        }
        if (failure.FixAttempted) {
          result += `**Fix Status:** ${failure.FixAttempted ? '🔧 Fix attempted' : '⏭️ No fix attempted'}\n`;
          if (failure.FixDescription) {
            result += `**Fix Applied:** ${failure.FixDescription}\n`;
          }
        }
        result += `\n`;
      });
    }

    if (output.Recommendations && output.Recommendations.length > 0) {
      result += `## Recommendations\n`;
      output.Recommendations.forEach((rec, idx) => {
        result += `${idx + 1}. ${rec}\n`;
      });
      result += `\n`;
    }

    return result;
  },

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.2, // Moderate temperature for debugging creativity
    top_p: 0.95,
    thinkingBudget: -1,
  },

  runConfig: {
    max_time_minutes: 8, // Longer timeout for test execution
    max_turns: 20, // More turns for iterative fixing
  },

  toolConfig: {
    // Needs execution and write permissions
    tools: [
      LS_TOOL_NAME,
      READ_FILE_TOOL_NAME,
      GLOB_TOOL_NAME,
      GREP_TOOL_NAME,
      SHELL_TOOL_NAME, // For running tests
      EDIT_TOOL_NAME, // For fixing code
      WRITE_FILE_TOOL_NAME, // For creating test files if needed
    ],
  },

  promptConfig: {
    query: `Run tests and fix failures using the following parameters:

Test Command: \${testCommand}
Max Fix Attempts: \${maxAttempts || "3"}
Target Files: \${targetFiles || "all tests"}

Execute tests, analyze failures, apply fixes, and re-run until all pass or max attempts reached.`,

    systemPrompt: `You are the **Test Runner Agent**, a specialized AI for executing tests, diagnosing failures, and automatically fixing issues.

Your **CORE MISSION** is to ensure all tests pass by running tests, analyzing failures, applying fixes, and iterating until success or max attempts.

## Test Execution Workflow

### Phase 1: Initial Test Run
1. Execute the test command using \`shell\`
2. Parse the test output to extract:
   - Total number of tests
   - Number passing/failing/skipped
   - Specific test names that failed
   - Error messages and stack traces
   - Execution duration
3. Record initial results

### Phase 2: Failure Analysis
For each failing test:
1. **Locate the test file** - Use the test name and stack trace
2. **Read the test code** - Understand what the test expects
3. **Identify the implementation** - Find the code being tested
4. **Analyze the root cause:**
   - Assertion failures: Expected vs actual values
   - Runtime errors: Null pointers, type errors, exceptions
   - Setup issues: Missing mocks, incorrect test data
   - Timing issues: Race conditions, async problems
   - Environment issues: Missing dependencies, config problems

### Phase 3: Fix Application
For each identified issue (up to \`maxAttempts\` per test):
1. **Determine fix strategy:**
   - Fix implementation code (not test code) if logic is wrong
   - Fix test code if test expectations are incorrect
   - Add missing setup/teardown
   - Fix async handling or timing issues
2. **Apply the fix** using \`edit\` or \`write_file\`
3. **Document the fix** for the report
4. **Re-run tests** to verify the fix works

### Phase 4: Iteration
1. After applying fixes, re-run the full test suite
2. Update results (new pass/fail counts)
3. If failures remain and attempts remain, repeat Phase 2-3
4. If all tests pass or max attempts exhausted, finalize report

### Phase 5: Reporting
Generate final report with:
- Initial and final test results
- List of failures with root cause analysis
- Fixes applied and their descriptions
- Whether all tests are now passing
- Recommendations for manual investigation if needed

## Fix Strategies by Error Type

### Assertion Failures
- Check if expected values are correct
- Verify implementation returns correct results
- Update test expectations if requirements changed
- Fix off-by-one errors, wrong calculations

### Runtime Errors
- Add null/undefined checks
- Fix type errors (use proper types)
- Handle exceptions properly
- Initialize variables before use

### Async Issues
- Add proper await keywords
- Use async/await correctly
- Fix race conditions with proper synchronization
- Add delays or wait conditions if needed

### Setup/Teardown Issues
- Add missing beforeEach/afterEach hooks
- Clear state between tests
- Mock external dependencies
- Reset singletons or global state

### Import/Dependency Errors
- Fix import paths
- Install missing dependencies
- Update outdated APIs
- Mock unavailable services

## Best Practices

✅ **DO:**
- Run tests first before making any changes
- Read test code carefully to understand intent
- Fix the root cause, not just the symptom
- Re-run tests after each fix to verify
- Keep fixes minimal and focused
- Document what you changed and why
- Consider edge cases and related tests
- Preserve test coverage and quality

❌ **DON'T:**
- Disable or skip failing tests (unless absolutely necessary)
- Make sweeping changes that might break other tests
- Fix only one test without running the full suite
- Guess at fixes without understanding the problem
- Modify code you don't understand (read context first)
- Exceed maxAttempts per test

## Tool Usage

- **shell:** Run test commands, check dependencies
- **read_file:** Examine test files and implementation
- **edit:** Fix specific lines in existing files
- **write_file:** Create missing test files or fixtures
- **grep:** Search for function definitions, imports, usage
- **glob:** Find related test files

## Output Requirements

Call \`complete_task\` with this JSON structure:

{
  "Summary": "Ran 50 tests, 45 passed initially, fixed 4 failures, 1 remains",
  "InitialResults": {
    "TotalTests": 50,
    "Passed": 45,
    "Failed": 5,
    "Duration": "12.4s"
  },
  "FinalResults": {
    "TotalTests": 50,
    "Passed": 49,
    "Failed": 1,
    "Duration": "11.8s"
  },
  "Failures": [
    {
      "TestName": "auth.test.ts > should validate password",
      "File": "tests/auth.test.ts",
      "ErrorMessage": "Expected true, received false",
      "RootCause": "Password validation regex was incorrect",
      "FixAttempted": true,
      "FixDescription": "Updated regex in validatePassword function"
    }
  ],
  "FixesApplied": 4,
  "AllTestsPassing": false,
  "Recommendations": [
    "Manually investigate remaining timeout issue in integration test"
  ]
}

## Special Considerations

- **Preserve Test Intent:** Don't weaken tests to make them pass
- **Backward Compatibility:** Ensure fixes don't break other code
- **Performance:** Be mindful of slow tests, suggest optimizations
- **Flaky Tests:** Identify non-deterministic failures, suggest stabilization
- **Coverage:** Maintain or improve test coverage with fixes

Remember: Your goal is to make tests pass by fixing real issues, not by making tests less strict. Good test quality is as important as passing tests.`,
  },
};

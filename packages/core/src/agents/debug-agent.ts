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
 * Schema for a debugging step.
 */
const DebugStepSchema = z.object({
  StepNumber: z
    .number()
    .describe('The sequence number of this debugging step.'),
  Action: z
    .string()
    .describe(
      'What action was taken (e.g., "Added logging", "Tested edge case").',
    ),
  Finding: z.string().describe('What was discovered from this step.'),
  FilesExamined: z
    .array(z.string())
    .optional()
    .describe('Files that were examined in this step.'),
});

/**
 * Schema for root cause analysis.
 */
const RootCauseSchema = z.object({
  Description: z.string().describe('Clear explanation of what causes the bug.'),
  Location: z
    .object({
      File: z.string(),
      Line: z.number().optional(),
      Function: z.string().optional(),
    })
    .describe('Where the bug originates.'),
  Explanation: z
    .string()
    .describe('Why this code causes the observed behavior.'),
  ImpactScope: z
    .enum(['LOCAL', 'MODULE', 'SYSTEM'])
    .describe('How widespread the impact of this bug is.'),
});

/**
 * Schema for the fix applied.
 */
const FixSchema = z.object({
  Description: z.string().describe('What was changed to fix the bug.'),
  FilesModified: z.array(z.string()).describe('Files that were modified.'),
  Approach: z
    .string()
    .describe(
      'The strategy used to fix the bug (e.g., "Added null check", "Fixed race condition").',
    ),
  TestingPerformed: z
    .string()
    .describe('How the fix was verified (tests run, manual testing, etc.).'),
  Verified: z.boolean().describe('Whether the fix was verified to work.'),
});

/**
 * Complete debug report schema.
 */
const DebugReportSchema = z.object({
  Summary: z
    .string()
    .describe('Executive summary of the debugging session and outcome.'),
  BugReproduced: z
    .boolean()
    .describe('Whether the bug was successfully reproduced.'),
  DebugSteps: z
    .array(DebugStepSchema)
    .describe('Step-by-step debugging process.'),
  RootCause: RootCauseSchema.optional().describe(
    'Root cause analysis if identified.',
  ),
  Fix: FixSchema.optional().describe('Fix applied if bug was resolved.'),
  PreventionRecommendations: z
    .array(z.string())
    .optional()
    .describe('Recommendations to prevent similar bugs in the future.'),
  UnresolvedIssues: z
    .array(z.string())
    .optional()
    .describe("Issues that still need investigation or couldn't be resolved."),
  BugFixed: z.boolean().describe('Whether the bug was successfully fixed.'),
});

/**
 * Debug Agent - Specialized in identifying and fixing bugs.
 *
 * This agent uses systematic debugging techniques to reproduce bugs,
 * identify root causes, and apply fixes.
 */
export const DebugAgent: AgentDefinition<typeof DebugReportSchema> = {
  name: 'debug_agent',
  displayName: 'Debug Agent',
  description: `A specialized agent for debugging and fixing bugs.
    Use this agent when you need to:
    - Investigate reported bugs or unexpected behavior
    - Reproduce issues from user reports
    - Identify root causes of failures
    - Apply fixes and verify they work
    - Analyze error logs and stack traces

    The agent uses systematic debugging: reproduce, isolate, identify, fix, verify.`,

  inputConfig: {
    inputs: {
      bugDescription: {
        description: `Detailed description of the bug. Include:
          - What is the expected behavior?
          - What actually happens?
          - Error messages or stack traces
          - When does it occur? (specific conditions, inputs, etc.)`,
        type: 'string',
        required: true,
      },
      reproduceSteps: {
        description: `Steps to reproduce the bug. Examples:
          - "Run 'npm start' and navigate to /users/123"
          - "Call getUserById() with null parameter"
          - "Execute the script with input file containing unicode characters"`,
        type: 'string',
        required: false,
      },
      context: {
        description: `Additional context such as:
          - Recent changes that might be related
          - Environment details (OS, versions, etc.)
          - Related bugs or issues
          - Files or components suspected to be involved`,
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description: 'Debugging report with root cause, fix, and verification.',
    schema: DebugReportSchema,
  },

  processOutput: (output) => {
    let result = `# Debug Report\n\n`;
    result += `## Summary\n${output.Summary}\n\n`;
    result += `**Bug Status:** ${output.BugFixed ? '✅ FIXED' : '❌ NOT FIXED'}\n`;
    result += `**Bug Reproduced:** ${output.BugReproduced ? '✅ Yes' : '❌ No'}\n\n`;

    if (output.DebugSteps && output.DebugSteps.length > 0) {
      result += `## Debugging Process\n\n`;
      output.DebugSteps.forEach((step) => {
        result += `### Step ${step.StepNumber}: ${step.Action}\n`;
        result += `**Finding:** ${step.Finding}\n`;
        if (step.FilesExamined && step.FilesExamined.length > 0) {
          result += `**Files Examined:** ${step.FilesExamined.join(', ')}\n`;
        }
        result += `\n`;
      });
    }

    if (output.RootCause) {
      result += `## Root Cause\n`;
      result += `**Location:** ${output.RootCause.Location.File}`;
      if (output.RootCause.Location.Line) {
        result += `:${output.RootCause.Location.Line}`;
      }
      if (output.RootCause.Location.Function) {
        result += ` (${output.RootCause.Location.Function})`;
      }
      result += `\n`;
      result += `**Impact Scope:** ${output.RootCause.ImpactScope}\n\n`;
      result += `**Description:** ${output.RootCause.Description}\n\n`;
      result += `**Explanation:** ${output.RootCause.Explanation}\n\n`;
    }

    if (output.Fix) {
      result += `## Fix Applied\n`;
      result += `**Approach:** ${output.Fix.Approach}\n`;
      result += `**Files Modified:** ${output.Fix.FilesModified.join(', ')}\n\n`;
      result += `**Description:** ${output.Fix.Description}\n\n`;
      result += `**Testing:** ${output.Fix.TestingPerformed}\n`;
      result += `**Verified:** ${output.Fix.Verified ? '✅ Yes' : '❌ No'}\n\n`;
    }

    if (
      output.PreventionRecommendations &&
      output.PreventionRecommendations.length > 0
    ) {
      result += `## Prevention Recommendations\n`;
      output.PreventionRecommendations.forEach((rec, idx) => {
        result += `${idx + 1}. ${rec}\n`;
      });
      result += `\n`;
    }

    if (output.UnresolvedIssues && output.UnresolvedIssues.length > 0) {
      result += `## Unresolved Issues\n`;
      output.UnresolvedIssues.forEach((issue, idx) => {
        result += `${idx + 1}. ${issue}\n`;
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
    max_time_minutes: 10, // Longer timeout for complex debugging
    max_turns: 25, // More turns for iterative investigation
  },

  toolConfig: {
    // Full access to read, execute, and modify
    tools: [
      LS_TOOL_NAME,
      READ_FILE_TOOL_NAME,
      GLOB_TOOL_NAME,
      GREP_TOOL_NAME,
      SHELL_TOOL_NAME,
      EDIT_TOOL_NAME,
      WRITE_FILE_TOOL_NAME,
    ],
  },

  promptConfig: {
    query: `Debug and fix the following bug:

Bug Description:
<bug>
\${bugDescription}
</bug>

${
  /* Template will conditionally include provided inputs */
  ''
}
Reproduce Steps:
<steps>
\${reproduceSteps}
</steps>

Additional Context:
<context>
\${context}
</context>

Systematically debug the issue, identify the root cause, apply a fix, and verify it works.`,

    systemPrompt: `You are the **Debug Agent**, a specialized AI for systematically debugging and fixing software bugs.

Your **CORE MISSION** is to identify the root cause of bugs and apply correct, verified fixes using proven debugging methodologies.

## Systematic Debugging Process

### Phase 1: Reproduce the Bug
1. **Understand the Bug Report**
   - Parse the bug description carefully
   - Identify expected vs actual behavior
   - Note any error messages, stack traces, or logs
   - Identify preconditions and trigger conditions

2. **Attempt Reproduction**
   - Follow the reproduce steps provided
   - If steps not provided, infer them from the description
   - Use \`shell\` to run commands, execute scripts
   - Look for error messages or unexpected outputs
   - **Critical:** Don't proceed until bug is reproduced or deemed non-reproducible

3. **Document Reproduction**
   - Record exact steps that trigger the bug
   - Capture error messages and outputs
   - Note environmental conditions

### Phase 2: Isolate the Bug
1. **Narrow Down the Scope**
   - Identify which component/module is involved
   - Use \`grep\` to find relevant code sections
   - Trace the execution path from user action to error
   - Identify the boundary between working and broken code

2. **Eliminate False Leads**
   - Test hypotheses about what might be wrong
   - Rule out red herrings (error messages from symptoms, not causes)
   - Focus on the proximate cause, not distant effects

3. **Use Debugging Techniques**
   - **Binary search:** Comment out half the code, narrow down the issue
   - **Add logging:** Insert temporary debug statements (if safe)
   - **Trace data flow:** Follow variables from input to output
   - **Check assumptions:** Verify preconditions, postconditions

### Phase 3: Identify Root Cause
1. **Analyze the Buggy Code**
   - Read the problematic function/method carefully
   - Understand what it's supposed to do
   - Identify where it deviates from expected behavior

2. **Common Bug Patterns**
   - **Null/undefined:** Missing null checks, uninitialized variables
   - **Off-by-one:** Loop boundaries, array indexing
   - **Type errors:** Wrong types, implicit conversions
   - **Race conditions:** Async issues, timing problems
   - **Logic errors:** Wrong algorithm, incorrect conditions
   - **State management:** Incorrect state updates, stale data
   - **Resource leaks:** Unclosed files, memory leaks
   - **API misuse:** Incorrect library usage, wrong parameters

3. **Determine Impact Scope**
   - LOCAL: Affects only one function/method
   - MODULE: Affects multiple functions in a module
   - SYSTEM: Affects multiple modules or subsystems

### Phase 4: Apply the Fix
1. **Design the Fix**
   - Choose the most appropriate solution
   - Consider edge cases and side effects
   - Ensure fix doesn't break other functionality
   - Prefer minimal, targeted changes

2. **Implement the Fix**
   - Use \`edit\` to modify the problematic code
   - Make surgical changes, not sweeping rewrites
   - Add comments explaining the fix if non-obvious
   - Consider adding defensive checks

3. **Common Fix Strategies**
   - Add null/undefined checks
   - Fix loop boundaries or conditions
   - Correct async/await usage
   - Add proper error handling
   - Fix race conditions with locks or proper sequencing
   - Update algorithm logic
   - Add input validation

### Phase 5: Verify the Fix
1. **Re-run Reproduction Steps**
   - Execute the exact steps that triggered the bug
   - Verify the bug no longer occurs
   - Check that error messages are gone

2. **Test Edge Cases**
   - Test boundary conditions
   - Try invalid inputs
   - Test related functionality

3. **Run Automated Tests**
   - Execute relevant test suite: \`npm test\`, \`pytest\`, etc.
   - Ensure no regressions introduced
   - Add new tests if coverage was missing

4. **Document Verification**
   - Record what testing was performed
   - Note whether fix was verified
   - List any remaining concerns

### Phase 6: Prevention Analysis
1. **Identify Why Bug Occurred**
   - Missing validation?
   - Insufficient testing?
   - Misunderstood requirements?
   - Edge case not considered?

2. **Recommend Preventions**
   - Add tests for this scenario
   - Improve input validation
   - Add assertions or type checks
   - Update documentation
   - Code review focus areas

## Tool Usage Strategy

- **grep/glob:** Find functions, classes, error messages
- **read_file:** Examine implementation, tests, logs
- **shell:** Reproduce bug, run tests, execute scripts
- **edit:** Apply fixes to code
- **write_file:** Create test files, add logging utilities
- **ls:** Understand project structure

## Output Requirements

Call \`complete_task\` with this JSON structure:

{
  "Summary": "Fixed null pointer exception in getUserById by adding null check",
  "BugReproduced": true,
  "DebugSteps": [
    {
      "StepNumber": 1,
      "Action": "Ran reproduce command",
      "Finding": "Confirmed error occurs with null input",
      "FilesExamined": ["src/user-service.ts"]
    }
  ],
  "RootCause": {
    "Description": "Function doesn't check for null userId before database query",
    "Location": {
      "File": "src/user-service.ts",
      "Line": 45,
      "Function": "getUserById"
    },
    "Explanation": "When userId is null, the database query crashes instead of returning gracefully",
    "ImpactScope": "MODULE"
  },
  "Fix": {
    "Description": "Added null check at function entry",
    "FilesModified": ["src/user-service.ts"],
    "Approach": "Added guard clause to return null for null input",
    "TestingPerformed": "Re-ran reproduction steps, ran unit tests",
    "Verified": true
  },
  "PreventionRecommendations": [
    "Add input validation to all public API functions",
    "Add unit tests for null/undefined inputs"
  ],
  "BugFixed": true
}

## Best Practices

✅ **DO:**
- Reproduce the bug before attempting fixes
- Be systematic: don't skip debugging phases
- Test fixes thoroughly before claiming success
- Consider side effects and edge cases
- Add tests to prevent regression
- Document the root cause clearly
- Keep fixes minimal and focused

❌ **DON'T:**
- Apply fixes without understanding the root cause
- Make changes without verifying they work
- Introduce new bugs while fixing old ones
- Ignore error messages or warnings
- Skip verification steps
- Fix symptoms instead of root causes
- Make sweeping changes when a targeted fix will do

## Special Debugging Scenarios

**Intermittent Bugs:**
- Look for race conditions, timing issues
- Check for shared state or global variables
- Test multiple times to reproduce

**Environment-Specific Bugs:**
- Check for OS-specific code
- Verify dependency versions
- Check environment variables

**Performance Bugs:**
- Use profiling, not guessing
- Look for O(n²) algorithms, memory leaks
- Check database query efficiency

**Integration Bugs:**
- Verify API contracts
- Check data serialization/deserialization
- Test error handling at boundaries

Remember: Debugging is detective work. Be methodical, test hypotheses, and don't jump to conclusions.`,
  },
};

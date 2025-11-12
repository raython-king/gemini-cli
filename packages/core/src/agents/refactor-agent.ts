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
 * Schema for a single refactoring operation.
 */
const RefactoringOperationSchema = z.object({
  Type: z
    .enum([
      'RENAME',
      'EXTRACT_FUNCTION',
      'EXTRACT_CLASS',
      'INLINE',
      'MOVE',
      'REMOVE_DUPLICATION',
      'SIMPLIFY',
      'IMPROVE_NAMES',
      'RESTRUCTURE',
    ])
    .describe('The type of refactoring performed.'),
  Description: z.string().describe('What was refactored and why.'),
  FilesAffected: z
    .array(z.string())
    .describe('Files modified by this operation.'),
  LinesChanged: z
    .number()
    .optional()
    .describe('Approximate number of lines changed.'),
});

/**
 * Schema for code quality improvements.
 */
const QualityImprovementSchema = z.object({
  Metric: z
    .enum([
      'READABILITY',
      'MAINTAINABILITY',
      'PERFORMANCE',
      'TESTABILITY',
      'COMPLEXITY',
      'DUPLICATION',
    ])
    .describe('The quality metric improved.'),
  Before: z.string().describe('Description of the state before refactoring.'),
  After: z.string().describe('Description of the state after refactoring.'),
  ImpactScore: z
    .number()
    .min(1)
    .max(10)
    .describe('How significant the improvement is (1-10).'),
});

/**
 * Complete refactoring report schema.
 */
const RefactorReportSchema = z.object({
  Summary: z
    .string()
    .describe('Executive summary of the refactoring work performed.'),
  Strategy: z
    .string()
    .describe('The overall refactoring strategy and approach used.'),
  Operations: z
    .array(RefactoringOperationSchema)
    .describe('List of refactoring operations performed.'),
  QualityImprovements: z
    .array(QualityImprovementSchema)
    .describe('Measurable improvements in code quality.'),
  TestsRan: z
    .boolean()
    .describe('Whether tests were run to verify refactoring.'),
  TestsPassed: z
    .boolean()
    .describe('Whether all tests passed after refactoring.'),
  BreakingChanges: z
    .array(z.string())
    .optional()
    .describe(
      'Any breaking changes introduced (should be none for pure refactoring).',
    ),
  RemainingIssues: z
    .array(z.string())
    .optional()
    .describe('Code quality issues that still remain.'),
  Success: z.boolean().describe('Whether the refactoring was successful.'),
});

/**
 * Refactor Agent - Specialized in code refactoring and quality improvement.
 *
 * This agent systematically improves code quality while preserving functionality.
 * It applies refactoring techniques, runs tests to verify correctness, and ensures
 * no breaking changes are introduced.
 */
export const RefactorAgent: AgentDefinition<typeof RefactorReportSchema> = {
  name: 'refactor_agent',
  displayName: 'Refactor Agent',
  description: `A specialized agent for refactoring code and improving quality.
    Use this agent to:
    - Improve code readability and maintainability
    - Reduce code duplication
    - Simplify complex code
    - Extract functions/classes for better structure
    - Rename for clarity
    - Optimize performance while preserving behavior
    - Ensure all refactorings are verified by tests

    The agent follows safe refactoring practices: change structure, not behavior.`,

  inputConfig: {
    inputs: {
      target: {
        description: `What to refactor. Examples:
          - "src/utils/validation.ts" (specific file)
          - "src/auth/" (entire directory)
          - "getUserById function" (specific function)
          - "Remove duplication in API handlers"`,
        type: 'string',
        required: true,
      },
      strategy: {
        description: `Refactoring strategy. Options:
          - "improve_readability": Focus on naming, formatting, clarity
          - "reduce_complexity": Simplify complex functions, reduce nesting
          - "remove_duplication": DRY principle, extract common code
          - "extract_components": Break down large functions/classes
          - "modernize": Update to modern patterns/syntax
          - "performance": Optimize hot paths
          - "comprehensive": All of the above (default)`,
        type: 'string',
        required: false,
      },
      testCommand: {
        description: `Command to run tests for verification.
          Example: "npm test", "pytest", "go test"
          If not provided, agent will try to auto-detect.`,
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description:
      'Refactoring report with operations, improvements, and test results.',
    schema: RefactorReportSchema,
  },

  processOutput: (output) => {
    let result = `# Refactoring Report\n\n`;
    result += `## Summary\n${output.Summary}\n\n`;
    result += `**Status:** ${output.Success ? '✅ SUCCESS' : '❌ FAILED'}\n`;
    result += `**Tests:** ${output.TestsRan ? (output.TestsPassed ? '✅ All Passed' : '❌ Some Failed') : '⚠️ Not Run'}\n\n`;

    result += `## Strategy\n${output.Strategy}\n\n`;

    if (output.Operations && output.Operations.length > 0) {
      result += `## Refactoring Operations (${output.Operations.length})\n\n`;
      output.Operations.forEach((op, idx) => {
        result += `### ${idx + 1}. ${op.Type.replace(/_/g, ' ')}\n`;
        result += `${op.Description}\n`;
        result += `**Files:** ${op.FilesAffected.join(', ')}\n`;
        if (op.LinesChanged) {
          result += `**Lines Changed:** ~${op.LinesChanged}\n`;
        }
        result += `\n`;
      });
    }

    if (output.QualityImprovements && output.QualityImprovements.length > 0) {
      result += `## Quality Improvements\n\n`;
      output.QualityImprovements.forEach((improvement, idx) => {
        result += `### ${idx + 1}. ${improvement.Metric} (Impact: ${improvement.ImpactScore}/10)\n`;
        result += `**Before:** ${improvement.Before}\n`;
        result += `**After:** ${improvement.After}\n\n`;
      });
    }

    if (output.BreakingChanges && output.BreakingChanges.length > 0) {
      result += `## ⚠️ Breaking Changes\n`;
      output.BreakingChanges.forEach((change, idx) => {
        result += `${idx + 1}. ${change}\n`;
      });
      result += `\n`;
    }

    if (output.RemainingIssues && output.RemainingIssues.length > 0) {
      result += `## Remaining Issues\n`;
      output.RemainingIssues.forEach((issue, idx) => {
        result += `${idx + 1}. ${issue}\n`;
      });
      result += `\n`;
    }

    return result;
  },

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.2, // Moderate temperature for creative refactoring
    top_p: 0.95,
    thinkingBudget: -1,
  },

  runConfig: {
    max_time_minutes: 8,
    max_turns: 20,
  },

  toolConfig: {
    // Full access for reading, modifying, and testing
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
    query: `Refactor the following target:

Target: \${target}
Strategy: \${strategy || "comprehensive"}
Test Command: \${testCommand || "auto-detect"}

Apply safe refactorings, verify with tests, and ensure no breaking changes.`,

    systemPrompt: `You are the **Refactor Agent**, a specialized AI expert in code refactoring and quality improvement.

Your **CORE MISSION** is to improve code quality through systematic refactoring while preserving functionality and ensuring all changes are verified by tests.

## Core Refactoring Principles

### The Golden Rule
**PRESERVE BEHAVIOR:** Refactoring changes code structure, not functionality. If behavior changes, it's not refactoring—it's rewriting.

### Safety First
1. **Run tests before refactoring** to establish baseline
2. **Make small, incremental changes**
3. **Run tests after each change** to verify safety
4. **Never skip test verification**
5. **Revert if tests fail** and debug the issue

## Refactoring Strategies

### 1. Improve Readability
**Goal:** Make code easier to understand
- **Rename variables/functions** for clarity (avoid abbreviations)
- **Extract magic numbers** to named constants
- **Add/improve comments** for complex logic
- **Format consistently** (indentation, spacing)
- **Simplify boolean expressions** (remove double negatives)
- **Use descriptive names** that explain purpose

**Example:**
\`\`\`typescript
// Before
function calc(a, b, t) { return t === 1 ? a + b : a * b; }

// After
function calculateTotal(price, quantity, operationType) {
  const isAddition = operationType === OperationType.Add;
  return isAddition ? price + quantity : price * quantity;
}
\`\`\`

### 2. Reduce Complexity
**Goal:** Simplify complex code
- **Extract functions** from long methods (>20 lines)
- **Reduce nesting** (guard clauses, early returns)
- **Simplify conditionals** (strategy pattern, lookup tables)
- **Break down complex expressions**
- **Limit function parameters** (<5 ideally)
- **Single Responsibility Principle**

**Techniques:**
- **Guard clauses:** Return early to reduce nesting
- **Extract method:** Pull out complex logic into named functions
- **Replace conditionals with polymorphism**

### 3. Remove Duplication (DRY)
**Goal:** Eliminate repeated code
- **Extract common logic** into shared functions
- **Create utility functions** for repeated operations
- **Use loops/maps** instead of repeated statements
- **Abstract patterns** into reusable components
- **Inheritance/composition** for shared behavior

**Look for:**
- Copy-pasted code blocks
- Similar functions with slight variations
- Repeated logic patterns

### 4. Extract Components
**Goal:** Break down large units
- **Extract functions** from long methods
- **Extract classes** from large classes (>300 lines)
- **Extract modules** for cohesive functionality
- **Separate concerns** (UI, logic, data)

**Guidelines:**
- Functions should do one thing well
- Classes should have single responsibility
- Modules should have clear, focused purpose

### 5. Modernize
**Goal:** Use modern language features
- **Arrow functions** (JavaScript/TypeScript)
- **Async/await** instead of callbacks
- **Destructuring** for cleaner code
- **Optional chaining** (?.) and nullish coalescing (??)
- **Template literals** instead of concatenation
- **const/let** instead of var

### 6. Performance Optimization
**Goal:** Improve execution speed
- **Memoization** for expensive calculations
- **Lazy loading** for large data
- **Efficient algorithms** (O(n) instead of O(n²))
- **Batch operations** instead of loops
- **Cache results** of pure functions
- **Avoid premature optimization** (profile first!)

## Refactoring Process

### Phase 1: Analysis
1. **Read the target code** thoroughly
2. **Understand its purpose** and current structure
3. **Identify code smells:**
   - Long methods (>20 lines)
   - Large classes (>300 lines)
   - Duplicated code
   - Magic numbers
   - Poor naming
   - Deep nesting (>3 levels)
   - Complex conditionals
   - Dead code

4. **Find related code** that might be affected
5. **Locate existing tests** for the code

### Phase 2: Plan
1. **Prioritize refactorings** by impact and safety
2. **Choose refactoring techniques** appropriate for each issue
3. **Determine test strategy** for verification
4. **Plan incremental steps** (not big bang!)

### Phase 3: Execute
For each refactoring:
1. **Run baseline tests** (confirm they pass)
2. **Apply one refactoring** (small, focused change)
3. **Run tests again** (verify no breakage)
4. **If tests fail:** revert and debug
5. **If tests pass:** commit to progress, move to next refactoring
6. **Document the change**

### Phase 4: Verify
1. **Run full test suite**
2. **Check for breaking changes** in public APIs
3. **Verify performance** hasn't degraded
4. **Review code quality metrics**

### Phase 5: Report
Document:
- What was refactored
- Techniques used
- Quality improvements
- Test results
- Any remaining issues

## Common Refactoring Operations

### Rename
Change names to be more descriptive and clear
\`\`\`typescript
// Before: getUserData(u)
// After: getUserProfileInformation(userId)
\`\`\`

### Extract Function
Pull out complex logic into well-named function
\`\`\`typescript
// Before: Inline calculation with magic numbers
// After: calculateDiscountedPrice(basePrice, discountRate)
\`\`\`

### Extract Class
Split large class into multiple focused classes

### Inline
Collapse unnecessary abstraction back into caller

### Move
Relocate function/class to more appropriate module

### Remove Duplication
Consolidate repeated code into shared function

### Simplify
Reduce complexity through guard clauses, better structure

## Test Verification

### Auto-detect Test Command
Look for:
- \`package.json\` → "test" script
- \`pytest\` → Python project
- \`go test\` → Go project
- \`cargo test\` → Rust project
- \`mvn test\` → Java/Maven

### Run Tests
Execute tests using \`shell\`:
\`\`\`
npm test
pytest
go test ./...
\`\`\`

### Interpret Results
- **All pass:** Refactoring is safe ✅
- **Any fail:** Revert or fix before proceeding ❌
- **New failures:** Refactoring broke something

## Output Requirements

Call \`complete_task\` with this JSON:

{
  "Summary": "Refactored validation.ts: extracted 3 functions, reduced complexity",
  "Strategy": "comprehensive",
  "Operations": [
    {
      "Type": "EXTRACT_FUNCTION",
      "Description": "Extracted email validation logic into validateEmail()",
      "FilesAffected": ["src/utils/validation.ts"],
      "LinesChanged": 15
    }
  ],
  "QualityImprovements": [
    {
      "Metric": "COMPLEXITY",
      "Before": "Single 80-line function with nested conditionals",
      "After": "4 focused functions, max 20 lines each",
      "ImpactScore": 8
    }
  ],
  "TestsRan": true,
  "TestsPassed": true,
  "Success": true
}

## Best Practices

✅ **DO:**
- Make small, incremental changes
- Run tests frequently
- Preserve existing behavior
- Improve names and clarity
- Reduce complexity and duplication
- Document significant changes
- Verify with tests

❌ **DON'T:**
- Make large, risky changes
- Skip test verification
- Change behavior (that's not refactoring!)
- Introduce breaking changes
- Refactor code you don't understand
- Optimize without profiling
- Remove tests or reduce coverage

## Special Considerations

**Legacy Code:**
- Add tests first if missing (characterization tests)
- Refactor in very small steps
- Be extra cautious about hidden dependencies

**Performance Refactoring:**
- Profile first to identify bottlenecks
- Measure before and after
- Don't sacrifice readability without good reason

**Public APIs:**
- Avoid breaking changes
- Deprecate before removing
- Consider versioning

Remember: Refactoring is about making code better without changing what it does. Always verify with tests!`,
  },
};

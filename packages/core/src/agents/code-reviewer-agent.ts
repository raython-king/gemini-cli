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
} from '../tools/tool-names.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { z } from 'zod';

/**
 * Schema for a code review issue.
 */
const ReviewIssueSchema = z.object({
  File: z.string().describe('The file where the issue was found.'),
  Line: z.number().optional().describe('Specific line number if applicable.'),
  Severity: z
    .enum(['INFO', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
    .describe('How serious this issue is.'),
  Category: z
    .enum([
      'BUG',
      'SECURITY',
      'PERFORMANCE',
      'MAINTAINABILITY',
      'STYLE',
      'BEST_PRACTICE',
      'TESTING',
      'DOCUMENTATION',
    ])
    .describe('The category of the issue.'),
  Title: z.string().describe('Short, descriptive title of the issue.'),
  Description: z
    .string()
    .describe('Detailed explanation of what the problem is.'),
  Suggestion: z
    .string()
    .describe('Specific recommendation for how to fix or improve.'),
  CodeSnippet: z
    .string()
    .optional()
    .describe('The problematic code snippet (if relevant).'),
});

/**
 * Schema for overall code quality metrics.
 */
const QualityMetricsSchema = z.object({
  OverallScore: z
    .number()
    .min(0)
    .max(10)
    .describe('Overall code quality score (0-10).'),
  Strengths: z.array(z.string()).describe('Positive aspects of the code.'),
  AreasForImprovement: z
    .array(z.string())
    .describe('General areas that need attention.'),
});

/**
 * Complete code review report schema.
 */
const CodeReviewReportSchema = z.object({
  Summary: z
    .string()
    .describe('Executive summary of the review (2-3 sentences).'),
  Issues: z.array(ReviewIssueSchema).describe('List of identified issues.'),
  QualityMetrics: QualityMetricsSchema.describe('Overall quality assessment.'),
  SecurityConcerns: z
    .array(z.string())
    .optional()
    .describe('Specific security vulnerabilities or concerns.'),
  TestingRecommendations: z
    .array(z.string())
    .optional()
    .describe('Recommendations for testing the changes.'),
  Approved: z
    .boolean()
    .describe(
      'Whether the code is approved for merging (false if any HIGH or CRITICAL issues).',
    ),
});

/**
 * CodeReviewer Agent - Automated code review and quality assessment.
 *
 * This agent examines code changes (or entire files) and provides detailed feedback
 * on quality, security, performance, maintainability, and best practices.
 */
export const CodeReviewerAgent: AgentDefinition<typeof CodeReviewReportSchema> =
  {
    name: 'code_reviewer',
    displayName: 'Code Reviewer Agent',
    description: `A specialized agent for conducting thorough code reviews.
    Use this agent to:
    - Review code changes before committing
    - Identify bugs, security vulnerabilities, and performance issues
    - Assess code quality and maintainability
    - Ensure adherence to best practices and coding standards
    - Provide constructive feedback and improvement suggestions

    The agent performs comprehensive analysis across multiple dimensions:
    correctness, security, performance, maintainability, testing, and documentation.`,

    inputConfig: {
      inputs: {
        files: {
          description: `List of file paths to review (comma-separated or newline-separated).
          Example: "src/auth.ts, src/utils/validation.ts"`,
          type: 'string',
          required: false,
        },
        diff: {
          description: `Git diff output showing the changes to review.
          Provide this if reviewing specific changes rather than entire files.
          Use 'git diff' or 'git diff HEAD~1' to get the diff.`,
          type: 'string',
          required: false,
        },
        focus: {
          description: `Optional focus areas for the review:
          - "security": Focus on security vulnerabilities
          - "performance": Focus on performance issues
          - "testing": Focus on test coverage and quality
          - "all": Comprehensive review (default)`,
          type: 'string',
          required: false,
        },
      },
    },

    outputConfig: {
      outputName: 'review',
      description:
        'Comprehensive code review report with issues, metrics, and recommendations.',
      schema: CodeReviewReportSchema,
    },

    processOutput: (output) => {
      let result = `# Code Review Report\n\n`;
      result += `## Summary\n${output.Summary}\n\n`;

      result += `## Quality Metrics\n`;
      result += `**Overall Score:** ${output.QualityMetrics.OverallScore}/10\n\n`;
      result += `**Approval Status:** ${output.Approved ? '✅ APPROVED' : '❌ NEEDS CHANGES'}\n\n`;

      if (output.QualityMetrics.Strengths.length > 0) {
        result += `### Strengths\n`;
        output.QualityMetrics.Strengths.forEach((s) => {
          result += `- ${s}\n`;
        });
        result += `\n`;
      }

      if (output.QualityMetrics.AreasForImprovement.length > 0) {
        result += `### Areas for Improvement\n`;
        output.QualityMetrics.AreasForImprovement.forEach((a) => {
          result += `- ${a}\n`;
        });
        result += `\n`;
      }

      if (output.Issues && output.Issues.length > 0) {
        result += `## Issues Found (${output.Issues.length})\n\n`;

        // Group issues by severity
        const severityOrder = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'];
        severityOrder.forEach((severity) => {
          const issuesOfSeverity = output.Issues.filter(
            (i) => i.Severity === severity,
          );
          if (issuesOfSeverity.length === 0) return;

          result += `### ${severity} (${issuesOfSeverity.length})\n\n`;
          issuesOfSeverity.forEach((issue, idx) => {
            result += `#### ${idx + 1}. [${issue.Category}] ${issue.Title}\n`;
            result += `**File:** ${issue.File}`;
            if (issue.Line) {
              result += `:${issue.Line}`;
            }
            result += `\n\n`;
            result += `${issue.Description}\n\n`;
            result += `**Suggestion:** ${issue.Suggestion}\n\n`;
            if (issue.CodeSnippet) {
              result += `\`\`\`\n${issue.CodeSnippet}\n\`\`\`\n\n`;
            }
          });
        });
      } else {
        result += `## Issues Found\nNo issues identified! ✨\n\n`;
      }

      if (output.SecurityConcerns && output.SecurityConcerns.length > 0) {
        result += `## Security Concerns\n`;
        output.SecurityConcerns.forEach((concern, idx) => {
          result += `${idx + 1}. ${concern}\n`;
        });
        result += `\n`;
      }

      if (
        output.TestingRecommendations &&
        output.TestingRecommendations.length > 0
      ) {
        result += `## Testing Recommendations\n`;
        output.TestingRecommendations.forEach((rec, idx) => {
          result += `${idx + 1}. ${rec}\n`;
        });
        result += `\n`;
      }

      return result;
    },

    modelConfig: {
      model: DEFAULT_GEMINI_MODEL,
      temp: 0.1, // Low temperature for consistent, conservative reviews
      top_p: 0.95,
      thinkingBudget: -1,
    },

    runConfig: {
      max_time_minutes: 4,
      max_turns: 10,
    },

    toolConfig: {
      // Read-only tools to examine code and context
      tools: [
        LS_TOOL_NAME,
        READ_FILE_TOOL_NAME,
        GLOB_TOOL_NAME,
        GREP_TOOL_NAME,
      ],
    },

    promptConfig: {
      query: `Conduct a thorough code review based on the following inputs:

${
  /* Template will conditionally include provided inputs */
  ''
}
Files to review:
<files>
\${files}
</files>

Diff to review:
<diff>
\${diff}
</diff>

Focus area: \${focus || "all"}

Provide a comprehensive review with specific, actionable feedback.`,

      systemPrompt: `You are the **Code Reviewer Agent**, an expert software engineer specializing in code quality, security, and best practices.

Your **CORE MISSION** is to conduct thorough, constructive code reviews that improve code quality and catch issues before they reach production.

## Review Dimensions

### 1. **Correctness & Bugs**
   - Logic errors, edge cases, null pointer issues
   - Incorrect algorithm implementations
   - Type mismatches or unsafe type conversions
   - Off-by-one errors, race conditions
   - Resource leaks (memory, file handles, connections)

### 2. **Security**
   - SQL injection, XSS, CSRF vulnerabilities
   - Authentication/authorization bypasses
   - Insecure data handling (passwords, tokens, PII)
   - Improper input validation and sanitization
   - Hardcoded secrets or credentials
   - Path traversal vulnerabilities
   - Insufficient error handling that leaks information

### 3. **Performance**
   - Inefficient algorithms (O(n²) where O(n) is possible)
   - Unnecessary loops or repeated calculations
   - Memory-intensive operations
   - Missing caching opportunities
   - Database N+1 queries
   - Blocking operations in async contexts
   - Large object allocations in hot paths

### 4. **Maintainability**
   - Code clarity and readability
   - Proper naming conventions
   - Function/method length and complexity
   - Code duplication (DRY principle)
   - Separation of concerns
   - Appropriate use of design patterns
   - Magic numbers and strings

### 5. **Testing**
   - Test coverage for new code
   - Edge case testing
   - Error path testing
   - Integration test needs
   - Test quality and clarity

### 6. **Documentation**
   - Missing or outdated comments
   - Unclear function/API documentation
   - Complex logic that needs explanation
   - README updates for new features

### 7. **Best Practices**
   - Language-specific idioms
   - Framework conventions
   - Team coding standards
   - Proper error handling
   - Logging and observability

## Review Process

### Step 1: Understand the Changes
- If given \`files\`, read each file completely
- If given \`diff\`, parse the changes and read surrounding context
- Use \`grep\` to find related code that might be affected
- Understand the purpose and intent of the changes

### Step 2: Analyze Each Dimension
- Systematically check each review dimension
- Focus on the specified \`focus\` area if provided
- Look for both issues and positive aspects

### Step 3: Prioritize Issues
- **CRITICAL**: Security vulnerabilities, data loss, major bugs
- **HIGH**: Significant bugs, performance issues, breaking changes
- **MEDIUM**: Code quality issues, minor bugs, maintainability concerns
- **LOW**: Style issues, minor improvements, suggestions
- **INFO**: Observations, questions, learning opportunities

### Step 4: Provide Constructive Feedback
- Be specific: cite file names, line numbers, code snippets
- Explain *why* something is an issue, not just *what*
- Suggest concrete fixes with examples when possible
- Balance criticism with recognition of good practices
- Assume good intent; be respectful and professional

## Output Requirements

Call \`complete_task\` with this JSON structure:

{
  "Summary": "Brief overview of the review findings",
  "Issues": [
    {
      "File": "path/to/file.ts",
      "Line": 42,
      "Severity": "HIGH",
      "Category": "SECURITY",
      "Title": "SQL Injection Vulnerability",
      "Description": "User input is directly interpolated into SQL query...",
      "Suggestion": "Use parameterized queries or an ORM...",
      "CodeSnippet": "const query = \`SELECT * FROM users WHERE id=\${userId}\`"
    }
  ],
  "QualityMetrics": {
    "OverallScore": 7,
    "Strengths": ["Good error handling", "Clear naming"],
    "AreasForImprovement": ["Test coverage", "Documentation"]
  },
  "SecurityConcerns": ["Potential SQL injection in auth module"],
  "TestingRecommendations": ["Add tests for edge cases", "Test error paths"],
  "Approved": false /* false if any HIGH or CRITICAL issues */
}

## Best Practices

✅ **DO:**
- Read the code carefully and understand context
- Look for subtle bugs, not just obvious ones
- Consider how the code might fail
- Check for consistency with existing codebase patterns
- Provide specific, actionable suggestions
- Recognize good code quality when you see it
- Think about future maintainability

❌ **DON'T:**
- Nitpick on minor style issues (unless specifically asked)
- Approve code with HIGH or CRITICAL security/correctness issues
- Make vague comments like "this looks bad"
- Focus only on formatting if there are logic issues
- Review code you don't understand (use \`read_file\` to learn context)

## Special Considerations

**Security Focus:** Prioritize OWASP Top 10, check all user inputs, review authentication/authorization
**Performance Focus:** Profile hot paths, check algorithm complexity, look for unnecessary work
**Testing Focus:** Evaluate test quality, coverage, and edge case handling

Remember: Your goal is to improve code quality while being constructive and helpful. A good review catches issues early and helps developers learn.`,
    },
  };

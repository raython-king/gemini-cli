/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Example usage of Agent UI components.
 *
 * This file demonstrates how to integrate AgentMonitor and PlanDisplay
 * components with the multi-agent system.
 */

import type React from 'react';
import { useState, useEffect } from 'react';
import { Box } from 'ink';
import { AgentMonitor, PlanDisplay } from './index.js';
import type { AgentInfo, Plan } from './types.js';

/**
 * Example 1: Monitoring multiple agents running in parallel
 */
export const MultiAgentMonitorExample: React.FC = () => {
  const [agents, setAgents] = useState<AgentInfo[]>([
    {
      id: 'explore-1',
      name: 'explore_agent',
      displayName: 'Explore Agent',
      status: 'running',
      progress: 60,
      startTime: new Date(Date.now() - 30000), // Started 30s ago
      currentActivity: 'Analyzing authentication implementation',
      toolCalls: [
        {
          toolName: 'grep',
          startTime: new Date(Date.now() - 25000),
          endTime: new Date(Date.now() - 23000),
          success: true,
        },
        {
          toolName: 'read_file',
          startTime: new Date(Date.now() - 20000),
          endTime: new Date(Date.now() - 18000),
          success: true,
        },
        {
          toolName: 'glob',
          startTime: new Date(Date.now() - 15000),
          // Still running
        },
      ],
    },
    {
      id: 'review-1',
      name: 'code_reviewer',
      displayName: 'Code Reviewer',
      status: 'completed',
      startTime: new Date(Date.now() - 60000), // Started 60s ago
      endTime: new Date(Date.now() - 10000), // Completed 10s ago
      toolCalls: [
        {
          toolName: 'read_file',
          startTime: new Date(Date.now() - 55000),
          endTime: new Date(Date.now() - 50000),
          success: true,
        },
        {
          toolName: 'grep',
          startTime: new Date(Date.now() - 45000),
          endTime: new Date(Date.now() - 40000),
          success: true,
        },
      ],
    },
    {
      id: 'test-1',
      name: 'test_runner',
      displayName: 'Test Runner',
      status: 'failed',
      startTime: new Date(Date.now() - 45000),
      endTime: new Date(Date.now() - 5000),
      error: 'Tests failed: 3 failures out of 15 tests',
      toolCalls: [
        {
          toolName: 'bash',
          startTime: new Date(Date.now() - 40000),
          endTime: new Date(Date.now() - 35000),
          success: false,
          error: 'npm test exited with code 1',
        },
      ],
    },
  ]);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setAgents((prev) =>
        prev.map((agent) => {
          if (agent.status === 'running' && agent.progress !== undefined) {
            return {
              ...agent,
              progress: Math.min(100, agent.progress + 5),
            };
          }
          return agent;
        }),
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return <AgentMonitor agents={agents} showToolHistory={true} maxToolCalls={3} />;
};

/**
 * Example 2: Displaying a plan with interactive confirmation
 */
export const PlanDisplayExample: React.FC = () => {
  const plan: Plan = {
    Summary:
      'Implement user authentication with JWT tokens using Express.js backend and React frontend. The system will include login, logout, and token refresh mechanisms with secure HTTP-only cookies.',
    Steps: [
      {
        StepNumber: 1,
        Description: 'Install required dependencies (jsonwebtoken, bcrypt, cookie-parser)',
        AffectedFiles: ['package.json', 'package-lock.json'],
        EstimatedComplexity: 'LOW',
        Dependencies: [],
        Rationale:
          'These libraries are foundational for JWT token generation, password hashing, and cookie handling.',
      },
      {
        StepNumber: 2,
        Description: 'Create authentication middleware for Express',
        AffectedFiles: ['src/middleware/auth.ts', 'src/types/express.d.ts'],
        EstimatedComplexity: 'MEDIUM',
        Dependencies: [1],
        Rationale:
          'Middleware will verify JWT tokens on protected routes and attach user info to requests.',
      },
      {
        StepNumber: 3,
        Description: 'Implement auth routes (login, logout, refresh)',
        AffectedFiles: ['src/routes/auth.ts', 'src/controllers/authController.ts'],
        EstimatedComplexity: 'HIGH',
        Dependencies: [1, 2],
        Rationale:
          'These endpoints handle user authentication flow including token generation and validation.',
      },
      {
        StepNumber: 4,
        Description: 'Create React authentication context and hooks',
        AffectedFiles: [
          'src/client/contexts/AuthContext.tsx',
          'src/client/hooks/useAuth.ts',
        ],
        EstimatedComplexity: 'MEDIUM',
        Dependencies: [3],
        Rationale: 'Provides client-side authentication state management and API integration.',
      },
      {
        StepNumber: 5,
        Description: 'Add login and logout UI components',
        AffectedFiles: [
          'src/client/components/LoginForm.tsx',
          'src/client/components/ProtectedRoute.tsx',
        ],
        EstimatedComplexity: 'MEDIUM',
        Dependencies: [4],
        Rationale: 'User-facing components for authentication interactions.',
      },
      {
        StepNumber: 6,
        Description: 'Write tests for authentication flow',
        AffectedFiles: ['tests/auth.test.ts', 'tests/client/auth.test.tsx'],
        EstimatedComplexity: 'HIGH',
        Dependencies: [3, 5],
        Rationale: 'Ensure authentication works correctly and securely.',
      },
    ],
    Risks: [
      {
        Description: 'Token expiration handling may cause UX issues if not implemented smoothly',
        Severity: 'MEDIUM',
        Mitigation:
          'Implement automatic token refresh in the background before expiration. Show clear error messages on auth failures.',
      },
      {
        Description: 'XSS vulnerabilities if tokens are stored in localStorage',
        Severity: 'HIGH',
        Mitigation:
          'Use HTTP-only cookies for token storage instead of localStorage. Implement CSRF protection.',
      },
      {
        Description: 'Password storage security',
        Severity: 'CRITICAL',
        Mitigation:
          'Use bcrypt with appropriate cost factor (12+) for password hashing. Never log or expose passwords.',
      },
    ],
    Dependencies: [
      {
        Type: 'LIBRARY',
        Name: 'jsonwebtoken',
        Reason: 'JWT token generation and verification',
      },
      {
        Type: 'LIBRARY',
        Name: 'bcrypt',
        Reason: 'Secure password hashing',
      },
      {
        Type: 'LIBRARY',
        Name: 'cookie-parser',
        Reason: 'Parse and handle HTTP cookies in Express',
      },
      {
        Type: 'SERVICE',
        Name: 'User Database',
        Reason: 'Store user credentials and profile information',
      },
    ],
    EstimatedDuration: '2-3 days',
    AlternativeApproaches: [
      {
        Description: 'Use OAuth 2.0 with third-party providers (Google, GitHub)',
        Pros: [
          'No need to manage passwords',
          'Better security through delegated authentication',
          'Easier user experience (fewer passwords to remember)',
        ],
        Cons: [
          'Dependency on external services',
          'More complex initial setup',
          'May require user accounts with third-party providers',
        ],
      },
      {
        Description: 'Use session-based authentication instead of JWT',
        Pros: [
          'Simpler to implement',
          'Built-in support in most frameworks',
          'Easier to invalidate sessions',
        ],
        Cons: [
          'Less scalable for distributed systems',
          'Requires server-side session storage',
          'Not suitable for stateless APIs',
        ],
      },
    ],
  };

  const handleConfirm = (): void => {
    console.log('Plan confirmed - proceeding with implementation');
    // Navigate to implementation phase...
  };

  const handleReject = (): void => {
    console.log('Plan rejected - requesting revision');
    // Ask for plan modifications...
  };

  return (
    <PlanDisplay
      plan={plan}
      interactive={true}
      onConfirm={handleConfirm}
      onReject={handleReject}
      showAlternatives={true}
    />
  );
};

/**
 * Example 3: Complete workflow - Plan display followed by agent monitoring
 */
export const CompleteWorkflowExample: React.FC = () => {
  const [phase, setPhase] = useState<'planning' | 'implementation'>('planning');
  const [agents, setAgents] = useState<AgentInfo[]>([]);

  const plan: Plan = {
    Summary: 'Add search functionality to the application',
    Steps: [
      {
        StepNumber: 1,
        Description: 'Create search API endpoint',
        AffectedFiles: ['src/api/search.ts'],
        EstimatedComplexity: 'MEDIUM',
        Dependencies: [],
        Rationale: 'Backend endpoint for search queries',
      },
    ],
    Risks: [],
    Dependencies: [],
    EstimatedDuration: '4-6 hours',
  };

  const handlePlanConfirm = (): void => {
    setPhase('implementation');

    // Start implementation agents
    setAgents([
      {
        id: 'impl-1',
        name: 'refactor_agent',
        displayName: 'Implementation Agent',
        status: 'running',
        startTime: new Date(),
        currentActivity: 'Creating search API endpoint',
        toolCalls: [],
      },
    ]);
  };

  return (
    <Box flexDirection="column">
      {phase === 'planning' ? (
        <PlanDisplay
          plan={plan}
          interactive={true}
          onConfirm={handlePlanConfirm}
          onReject={() => console.log('Plan rejected')}
        />
      ) : (
        <AgentMonitor agents={agents} showToolHistory={true} />
      )}
    </Box>
  );
};

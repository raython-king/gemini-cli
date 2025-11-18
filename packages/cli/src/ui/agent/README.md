# Agent UI Components

This directory contains UI components for the multi-agent system integration in Gemini CLI.

## Components

### AgentMonitor

A real-time monitoring component for agent execution that displays:

- List of running agents with their current status
- Progress indicators for each agent
- Tool call history with timing information
- Success/failure status with error messages
- Organized view by agent status (running, completed, failed)

**Props:**

```typescript
interface AgentMonitorProps {
  agents: AgentInfo[];           // List of agents to monitor
  showToolHistory?: boolean;     // Show tool call history (default: true)
  maxToolCalls?: number;         // Max tool calls to display per agent (default: 5)
}
```

**Example Usage:**

```typescript
import { AgentMonitor } from './ui/agent';

const agents: AgentInfo[] = [
  {
    id: 'agent-1',
    name: 'plan_agent',
    displayName: 'Plan Agent',
    status: 'running',
    progress: 45,
    startTime: new Date(),
    currentActivity: 'Analyzing codebase structure',
    toolCalls: [
      {
        toolName: 'glob',
        startTime: new Date(Date.now() - 2000),
        endTime: new Date(Date.now() - 1000),
        success: true,
      },
    ],
  },
];

<AgentMonitor agents={agents} showToolHistory={true} maxToolCalls={5} />
```

### PlanDisplay

A component for displaying Plan Agent output in a structured, readable format:

- Executive summary of the plan
- Ordered implementation steps with complexity levels
- Dependencies between steps
- Risk assessment with severity levels and mitigation strategies
- External dependencies (libraries, APIs, services)
- Alternative approaches (optional)
- Interactive confirmation (optional)

**Props:**

```typescript
interface PlanDisplayProps {
  plan: Plan;                    // The plan to display
  interactive?: boolean;         // Enable interactive confirmation (default: false)
  onConfirm?: () => void;       // Callback when user confirms
  onReject?: () => void;        // Callback when user rejects
  showAlternatives?: boolean;   // Show alternative approaches (default: true)
  expandAll?: boolean;          // Expand all sections (default: false)
}
```

**Example Usage:**

```typescript
import { PlanDisplay } from './ui/agent';

const plan: Plan = {
  Summary: 'Implement user authentication with JWT tokens using Express.js and React',
  Steps: [
    {
      StepNumber: 1,
      Description: 'Set up JWT library and middleware',
      AffectedFiles: ['package.json', 'src/middleware/auth.ts'],
      EstimatedComplexity: 'LOW',
      Dependencies: [],
      Rationale: 'Foundation for authentication system',
    },
  ],
  Risks: [
    {
      Description: 'Token expiration handling',
      Severity: 'MEDIUM',
      Mitigation: 'Implement refresh token mechanism',
    },
  ],
  Dependencies: [
    {
      Type: 'LIBRARY',
      Name: 'jsonwebtoken',
      Reason: 'JWT token generation and validation',
    },
  ],
  EstimatedDuration: '2-3 hours',
};

<PlanDisplay
  plan={plan}
  interactive={true}
  onConfirm={() => console.log('Plan confirmed')}
  onReject={() => console.log('Plan rejected')}
/>
```

## Types

All TypeScript types are exported from `types.ts`:

- `AgentInfo` - Agent monitoring information
- `AgentStatus` - Agent execution status
- `ToolCallRecord` - Tool call history entry
- `Plan` - Complete plan structure
- `PlanStep` - Individual implementation step
- `Risk` - Risk assessment item
- `Dependency` - External dependency
- `AlternativeApproach` - Alternative implementation approach

## Integration with Multi-Agent System

These components integrate with the core multi-agent system:

```typescript
import { AgentExecutor, PlanAgent } from '@google/gemini-cli-core';
import { AgentMonitor, PlanDisplay } from './ui/agent';

// Track agent execution
const agentInfo: AgentInfo = {
  id: 'plan-1',
  name: 'plan_agent',
  status: 'pending',
  startTime: new Date(),
  toolCalls: [],
};

// Create activity callback to update UI
const activityCallback = (event: SubagentActivityEvent) => {
  if (event.type === 'TOOL_CALL_START') {
    agentInfo.toolCalls.push({
      toolName: event.data.toolName as string,
      startTime: new Date(),
    });
  }
  // Update UI state...
};

// Execute agent
const executor = await AgentExecutor.create(
  PlanAgent,
  runtimeContext,
  activityCallback,
);

const result = await executor.run({ task: 'Implement auth' }, signal);

// Display plan
const plan = JSON.parse(result.result);
<PlanDisplay plan={plan} interactive={true} />
```

## Styling

Components use the Ink framework (React for CLI) and follow the existing UI patterns:

- `theme` from `../semantic-colors.js` for consistent colors
- `formatDuration` from `../utils/formatters.js` for time formatting
- `Box` and `Text` components from `ink` for layout
- Border styles and spacing consistent with other UI components

## Future Enhancements

Planned improvements:

- [ ] Collapse/expand functionality for agent details
- [ ] Filtering and search for agent history
- [ ] Export agent execution reports
- [ ] Performance metrics visualization
- [ ] Parallel agent execution visualization
- [ ] Agent dependency graph display
- [ ] Real-time progress bars with ETA
- [ ] Interactive step-by-step execution

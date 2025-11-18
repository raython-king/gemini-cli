/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import { useEffect, useState } from 'react';
import { Box, Text } from 'ink';
import { theme } from '../semantic-colors.js';
import { formatDuration } from '../utils/formatters.js';
import type { AgentInfo, ToolCallRecord } from './types.js';

interface AgentMonitorProps {
  /** List of agents to monitor */
  agents: AgentInfo[];
  /** Whether to show detailed tool call history */
  showToolHistory?: boolean;
  /** Maximum number of tool calls to display per agent */
  maxToolCalls?: number;
}

/**
 * AgentMonitor - Real-time monitoring component for agent execution.
 *
 * Displays:
 * - List of running agents
 * - Progress and status for each agent
 * - Tool call history
 * - Real-time updates
 */
export const AgentMonitor: React.FC<AgentMonitorProps> = ({
  agents,
  showToolHistory = true,
  maxToolCalls = 5,
}) => {
  const [currentTime, setCurrentTime] = useState(new Date());

  // Update current time every second for duration calculations
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  if (agents.length === 0) {
    return (
      <Box
        borderStyle="round"
        borderColor={theme.border.default}
        paddingY={1}
        paddingX={2}
      >
        <Text color={theme.text.secondary}>
          No agents currently running.
        </Text>
      </Box>
    );
  }

  const getStatusColor = (status: AgentInfo['status']): string => {
    switch (status) {
      case 'pending':
        return theme.text.secondary;
      case 'running':
        return theme.status.info;
      case 'completed':
        return theme.status.success;
      case 'failed':
        return theme.status.error;
      case 'cancelled':
        return theme.status.warning;
      default:
        return theme.text.primary;
    }
  };

  const getStatusIcon = (status: AgentInfo['status']): string => {
    switch (status) {
      case 'pending':
        return '⏸';
      case 'running':
        return '⚙';
      case 'completed':
        return '✓';
      case 'failed':
        return '✗';
      case 'cancelled':
        return '⊗';
      default:
        return '?';
    }
  };

  const calculateDuration = (agent: AgentInfo): number => {
    const endTime = agent.endTime || currentTime;
    return endTime.getTime() - agent.startTime.getTime();
  };

  const renderToolCallHistory = (toolCalls: ToolCallRecord[]): React.JSX.Element | null => {
    if (!showToolHistory || toolCalls.length === 0) {
      return null;
    }

    const recentCalls = toolCalls.slice(-maxToolCalls);

    return (
      <Box flexDirection="column" marginLeft={4} marginTop={1}>
        <Text bold color={theme.text.primary}>
          Tool Calls ({toolCalls.length} total):
        </Text>
        {recentCalls.map((call, index) => {
          const duration = call.endTime
            ? call.endTime.getTime() - call.startTime.getTime()
            : currentTime.getTime() - call.startTime.getTime();

          const statusColor = call.success === false
            ? theme.status.error
            : call.endTime
            ? theme.status.success
            : theme.status.info;

          const statusIcon = call.success === false
            ? '✗'
            : call.endTime
            ? '✓'
            : '⋯';

          return (
            <Box key={index} marginLeft={2}>
              <Text color={statusColor}>{statusIcon}</Text>
              <Box marginLeft={1}>
                <Text color={theme.text.link}>{call.toolName}</Text>
              </Box>
              <Box marginLeft={1}>
                <Text color={theme.text.secondary}>
                  ({formatDuration(duration)})
                </Text>
              </Box>
              {call.error && (
                <Box marginLeft={1}>
                  <Text color={theme.status.error}>
                    - {call.error}
                  </Text>
                </Box>
              )}
            </Box>
          );
        })}
        {toolCalls.length > maxToolCalls && (
          <Box marginLeft={2}>
            <Text color={theme.text.secondary}>
              ... and {toolCalls.length - maxToolCalls} more
            </Text>
          </Box>
        )}
      </Box>
    );
  };

  const renderAgent = (agent: AgentInfo): React.JSX.Element => {
    const duration = calculateDuration(agent);
    const statusColor = getStatusColor(agent.status);
    const statusIcon = getStatusIcon(agent.status);

    return (
      <Box key={agent.id} flexDirection="column" marginY={1}>
        {/* Agent header with status */}
        <Box>
          <Text color={statusColor} bold>
            {statusIcon} {agent.displayName || agent.name}
          </Text>
          <Box marginLeft={2}>
            <Text color={theme.text.secondary}>
              ({formatDuration(duration)})
            </Text>
          </Box>
          {agent.progress !== undefined && agent.status === 'running' && (
            <Box marginLeft={2}>
              <Text color={theme.status.info}>
                {agent.progress}%
              </Text>
            </Box>
          )}
        </Box>

        {/* Current activity */}
        {agent.currentActivity && agent.status === 'running' && (
          <Box marginLeft={4} marginTop={0}>
            <Text color={theme.text.secondary}>
              → {agent.currentActivity}
            </Text>
          </Box>
        )}

        {/* Error message */}
        {agent.error && (
          <Box marginLeft={4} marginTop={0}>
            <Text color={theme.status.error}>
              Error: {agent.error}
            </Text>
          </Box>
        )}

        {/* Tool call history */}
        {renderToolCallHistory(agent.toolCalls)}
      </Box>
    );
  };

  const runningAgents = agents.filter((a) => a.status === 'running');
  const completedAgents = agents.filter((a) => a.status === 'completed');
  const failedAgents = agents.filter((a) => a.status === 'failed');
  const otherAgents = agents.filter(
    (a) => !['running', 'completed', 'failed'].includes(a.status),
  );

  return (
    <Box
      borderStyle="round"
      borderColor={theme.border.default}
      flexDirection="column"
      paddingY={1}
      paddingX={2}
    >
      <Text bold color={theme.text.accent}>
        Agent Monitor
      </Text>

      {/* Summary statistics */}
      <Box marginTop={1}>
        <Text color={theme.text.primary}>
          Running: <Text color={theme.status.info}>{runningAgents.length}</Text>
          {' | '}
          Completed: <Text color={theme.status.success}>{completedAgents.length}</Text>
          {' | '}
          Failed: <Text color={theme.status.error}>{failedAgents.length}</Text>
        </Text>
      </Box>

      {/* Divider */}
      <Box
        borderStyle="single"
        borderBottom={true}
        borderTop={false}
        borderLeft={false}
        borderRight={false}
        borderColor={theme.border.default}
        marginY={1}
      />

      {/* Running agents first */}
      {runningAgents.length > 0 && (
        <Box flexDirection="column">
          <Text bold color={theme.status.info}>
            Running Agents:
          </Text>
          {runningAgents.map(renderAgent)}
        </Box>
      )}

      {/* Completed agents */}
      {completedAgents.length > 0 && (
        <Box flexDirection="column" marginTop={runningAgents.length > 0 ? 1 : 0}>
          <Text bold color={theme.status.success}>
            Completed Agents:
          </Text>
          {completedAgents.map(renderAgent)}
        </Box>
      )}

      {/* Failed agents */}
      {failedAgents.length > 0 && (
        <Box
          flexDirection="column"
          marginTop={(runningAgents.length > 0 || completedAgents.length > 0) ? 1 : 0}
        >
          <Text bold color={theme.status.error}>
            Failed Agents:
          </Text>
          {failedAgents.map(renderAgent)}
        </Box>
      )}

      {/* Other agents */}
      {otherAgents.length > 0 && (
        <Box
          flexDirection="column"
          marginTop={
            (runningAgents.length > 0 || completedAgents.length > 0 || failedAgents.length > 0)
              ? 1
              : 0
          }
        >
          {otherAgents.map(renderAgent)}
        </Box>
      )}
    </Box>
  );
};

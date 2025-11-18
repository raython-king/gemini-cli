/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import { useState, useCallback } from 'react';
import { Box, Text } from 'ink';
import { theme } from '../semantic-colors.js';
import type { Plan, PlanStep, Risk, Dependency, AlternativeApproach } from './types.js';
import { useKeypress } from '../hooks/useKeypress.js';

interface PlanDisplayProps {
  /** The plan to display */
  plan: Plan;
  /** Whether to show the plan in interactive mode */
  interactive?: boolean;
  /** Callback when user confirms the plan (in interactive mode) */
  onConfirm?: () => void;
  /** Callback when user rejects the plan (in interactive mode) */
  onReject?: () => void;
  /** Whether to show alternative approaches */
  showAlternatives?: boolean;
  /** Whether to expand all sections by default */
  expandAll?: boolean;
}

/**
 * PlanDisplay - Component for displaying Plan Agent output.
 *
 * Displays:
 * - Plan summary
 * - Implementation steps with dependencies
 * - Risk assessment
 * - Dependencies
 * - Alternative approaches (optional)
 * - Interactive confirmation (optional)
 */
export const PlanDisplay: React.FC<PlanDisplayProps> = ({
  plan,
  interactive = false,
  onConfirm,
  onReject,
  showAlternatives = true,
  expandAll = false,
}) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    expandAll ? new Set(['steps', 'risks', 'dependencies', 'alternatives']) : new Set(['steps']),
  );

  const toggleSection = useCallback((section: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(section)) {
        next.delete(section);
      } else {
        next.add(section);
      }
      return next;
    });
  }, []);

  useKeypress(
    (key) => {
      if (!interactive) return;

      if (key.name === 'y' || key.name === 'return') {
        onConfirm?.();
      } else if (key.name === 'n') {
        onReject?.();
      }
    },
    { isActive: interactive },
  );

  const getComplexityColor = (complexity: PlanStep['EstimatedComplexity']): string => {
    switch (complexity) {
      case 'LOW':
        return theme.status.success;
      case 'MEDIUM':
        return theme.status.warning;
      case 'HIGH':
        return theme.status.error;
      default:
        return theme.text.primary;
    }
  };

  const getSeverityColor = (severity: Risk['Severity']): string => {
    switch (severity) {
      case 'LOW':
        return theme.status.success;
      case 'MEDIUM':
        return theme.status.warning;
      case 'HIGH':
        return theme.status.error;
      case 'CRITICAL':
        return theme.status.error;
      default:
        return theme.text.primary;
    }
  };

  const getDependencyTypeIcon = (type: Dependency['Type']): string => {
    switch (type) {
      case 'LIBRARY':
        return '📦';
      case 'API':
        return '🔌';
      case 'TOOL':
        return '🔧';
      case 'SERVICE':
        return '☁';
      case 'FILE':
        return '📄';
      default:
        return '•';
    }
  };

  const renderStep = (step: PlanStep): React.JSX.Element => {
    const complexityColor = getComplexityColor(step.EstimatedComplexity);

    return (
      <Box key={step.StepNumber} flexDirection="column" marginY={1}>
        {/* Step header */}
        <Box>
          <Text bold color={theme.text.accent}>
            Step {step.StepNumber}:
          </Text>
          <Box marginLeft={1}>
            <Text color={theme.text.primary}>{step.Description}</Text>
          </Box>
        </Box>

        {/* Complexity */}
        <Box marginLeft={4}>
          <Text color={theme.text.secondary}>Complexity: </Text>
          <Text bold color={complexityColor}>
            {step.EstimatedComplexity}
          </Text>
        </Box>

        {/* Dependencies */}
        {step.Dependencies.length > 0 && (
          <Box marginLeft={4}>
            <Text color={theme.text.secondary}>
              Depends on steps: {step.Dependencies.join(', ')}
            </Text>
          </Box>
        )}

        {/* Affected files */}
        {step.AffectedFiles.length > 0 && (
          <Box marginLeft={4} flexDirection="column">
            <Text color={theme.text.secondary}>Affected files:</Text>
            {step.AffectedFiles.map((file, index) => (
              <Box key={index} marginLeft={2}>
                <Text color={theme.text.link}>• {file}</Text>
              </Box>
            ))}
          </Box>
        )}

        {/* Rationale */}
        <Box marginLeft={4} marginTop={0}>
          <Text color={theme.text.secondary} italic>
            {step.Rationale}
          </Text>
        </Box>
      </Box>
    );
  };

  const renderRisk = (risk: Risk, index: number): React.JSX.Element => {
    const severityColor = getSeverityColor(risk.Severity);

    return (
      <Box key={index} flexDirection="column" marginY={1}>
        <Box>
          <Text bold color={severityColor}>
            [{risk.Severity}]
          </Text>
          <Box marginLeft={1}>
            <Text color={theme.text.primary}>{risk.Description}</Text>
          </Box>
        </Box>
        <Box marginLeft={4}>
          <Text color={theme.text.secondary}>Mitigation: </Text>
          <Text color={theme.text.primary}>{risk.Mitigation}</Text>
        </Box>
      </Box>
    );
  };

  const renderDependency = (dependency: Dependency, index: number): React.JSX.Element => {
    const icon = getDependencyTypeIcon(dependency.Type);

    return (
      <Box key={index} flexDirection="column" marginY={1}>
        <Box>
          <Text>{icon}</Text>
          <Box marginLeft={1}>
            <Text bold color={theme.text.link}>
              {dependency.Name}
            </Text>
          </Box>
          <Box marginLeft={1}>
            <Text color={theme.text.secondary}>({dependency.Type})</Text>
          </Box>
        </Box>
        <Box marginLeft={4}>
          <Text color={theme.text.secondary}>{dependency.Reason}</Text>
        </Box>
      </Box>
    );
  };

  const renderAlternative = (alternative: AlternativeApproach, index: number): React.JSX.Element => {
    return (
      <Box key={index} flexDirection="column" marginY={1}>
        <Text bold color={theme.text.primary}>
          Alternative {index + 1}: {alternative.Description}
        </Text>

        {/* Pros */}
        {alternative.Pros.length > 0 && (
          <Box marginLeft={4} flexDirection="column" marginTop={0}>
            <Text color={theme.status.success} bold>
              Pros:
            </Text>
            {alternative.Pros.map((pro, i) => (
              <Box key={i} marginLeft={2}>
                <Text color={theme.status.success}>+ {pro}</Text>
              </Box>
            ))}
          </Box>
        )}

        {/* Cons */}
        {alternative.Cons.length > 0 && (
          <Box marginLeft={4} flexDirection="column" marginTop={0}>
            <Text color={theme.status.error} bold>
              Cons:
            </Text>
            {alternative.Cons.map((con, i) => (
              <Box key={i} marginLeft={2}>
                <Text color={theme.status.error}>- {con}</Text>
              </Box>
            ))}
          </Box>
        )}
      </Box>
    );
  };

  const isExpanded = (section: string): boolean => expandedSections.has(section);

  return (
    <Box
      borderStyle="round"
      borderColor={interactive ? theme.border.focused : theme.border.default}
      flexDirection="column"
      paddingY={1}
      paddingX={2}
    >
      <Text bold color={theme.text.accent}>
        Implementation Plan
      </Text>

      {/* Summary */}
      <Box marginTop={1} flexDirection="column">
        <Text bold color={theme.text.primary}>
          Summary:
        </Text>
        <Box marginLeft={2}>
          <Text color={theme.text.primary}>{plan.Summary}</Text>
        </Box>
      </Box>

      {/* Estimated Duration */}
      <Box marginTop={1}>
        <Text bold color={theme.text.primary}>
          Estimated Duration:
        </Text>
        <Box marginLeft={1}>
          <Text color={theme.status.warning}>{plan.EstimatedDuration}</Text>
        </Box>
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

      {/* Steps Section */}
      <Box flexDirection="column">
        <Box>
          <Text bold color={theme.text.accent}>
            {isExpanded('steps') ? '▼' : '▶'} Implementation Steps ({plan.Steps.length})
          </Text>
        </Box>
        {isExpanded('steps') && (
          <Box flexDirection="column">
            {plan.Steps.map(renderStep)}
          </Box>
        )}
      </Box>

      {/* Risks Section */}
      {plan.Risks.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Box>
            <Text bold color={theme.status.warning}>
              {isExpanded('risks') ? '▼' : '▶'} Risks ({plan.Risks.length})
            </Text>
          </Box>
          {isExpanded('risks') && (
            <Box flexDirection="column">
              {plan.Risks.map(renderRisk)}
            </Box>
          )}
        </Box>
      )}

      {/* Dependencies Section */}
      {plan.Dependencies.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Box>
            <Text bold color={theme.text.primary}>
              {isExpanded('dependencies') ? '▼' : '▶'} Dependencies ({plan.Dependencies.length})
            </Text>
          </Box>
          {isExpanded('dependencies') && (
            <Box flexDirection="column">
              {plan.Dependencies.map(renderDependency)}
            </Box>
          )}
        </Box>
      )}

      {/* Alternative Approaches Section */}
      {showAlternatives && plan.AlternativeApproaches && plan.AlternativeApproaches.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Box>
            <Text bold color={theme.text.secondary}>
              {isExpanded('alternatives') ? '▼' : '▶'} Alternative Approaches ({plan.AlternativeApproaches.length})
            </Text>
          </Box>
          {isExpanded('alternatives') && (
            <Box flexDirection="column">
              {plan.AlternativeApproaches.map(renderAlternative)}
            </Box>
          )}
        </Box>
      )}

      {/* Interactive confirmation */}
      {interactive && (
        <>
          <Box
            borderStyle="single"
            borderBottom={true}
            borderTop={false}
            borderLeft={false}
            borderRight={false}
            borderColor={theme.border.default}
            marginY={1}
          />
          <Box>
            <Text color={theme.text.primary}>
              Proceed with this plan?
            </Text>
            <Box marginLeft={1}>
              <Text color={theme.status.success}>[Y]es</Text>
            </Box>
            <Box marginLeft={1}>
              <Text color={theme.status.error}>[N]o</Text>
            </Box>
          </Box>
        </>
      )}
    </Box>
  );
};

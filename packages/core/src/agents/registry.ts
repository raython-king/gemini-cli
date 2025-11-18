/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Config } from '../config/config.js';
import type { AgentDefinition } from './types.js';
import { CodebaseInvestigatorAgent } from './codebase-investigator.js';
import { PlanAgent } from './plan-agent.js';
import { ExploreAgent } from './explore-agent.js';
import { CodeReviewerAgent } from './code-reviewer-agent.js';
import { TestRunnerAgent } from './test-runner-agent.js';
import { DebugAgent } from './debug-agent.js';
import { RefactorAgent } from './refactor-agent.js';
import { type z } from 'zod';
import { debugLogger } from '../utils/debugLogger.js';
import { DataFlowAgent } from './analysis/data-flow-agent.js';
import { LiteratureAnalyzerAgent } from './analysis/literature-analyzer-agent.js';
import { SummarizerAgent } from './analysis/summarizer-agent.js';

/**
 * Manages the discovery, loading, validation, and registration of
 * AgentDefinitions.
 */
export class AgentRegistry {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private readonly agents = new Map<string, AgentDefinition<any>>();

  constructor(private readonly config: Config) {}

  /**
   * Discovers and loads agents.
   */
  async initialize(): Promise<void> {
    this.loadBuiltInAgents();

    if (this.config.getDebugMode()) {
      debugLogger.log(
        `[AgentRegistry] Initialized with ${this.agents.size} agents.`,
      );
    }
  }

  private loadBuiltInAgents(): void {
    const investigatorSettings = this.config.getCodebaseInvestigatorSettings();

    // Register the original Codebase Investigator Agent if enabled
    if (investigatorSettings?.enabled) {
      const agentDef = {
        ...CodebaseInvestigatorAgent,
        modelConfig: {
          ...CodebaseInvestigatorAgent.modelConfig,
          model:
            investigatorSettings.model ??
            CodebaseInvestigatorAgent.modelConfig.model,
          thinkingBudget:
            investigatorSettings.thinkingBudget ??
            CodebaseInvestigatorAgent.modelConfig.thinkingBudget,
        },
        runConfig: {
          ...CodebaseInvestigatorAgent.runConfig,
          max_time_minutes:
            investigatorSettings.maxTimeMinutes ??
            CodebaseInvestigatorAgent.runConfig.max_time_minutes,
          max_turns:
            investigatorSettings.maxNumTurns ??
            CodebaseInvestigatorAgent.runConfig.max_turns,
        },
      };
      this.registerAgent(agentDef);
    }

    // Register new specialized agents (enabled by default)
    // TODO: Add config options to enable/disable individual agents
    // Register Plan Agent
    this.registerAgent(PlanAgent);

    // Register Explore Agent (faster alternative to CodebaseInvestigator)
    this.registerAgent(ExploreAgent);

    // Register Code Reviewer Agent
    this.registerAgent(CodeReviewerAgent);

    // Register Test Runner Agent
    this.registerAgent(TestRunnerAgent);

    // Register Debug Agent
    this.registerAgent(DebugAgent);

    // Register Refactor Agent
    this.registerAgent(RefactorAgent);

    // Register Analysis Agents
    this.registerAgent(DataFlowAgent);
    this.registerAgent(LiteratureAnalyzerAgent);
    this.registerAgent(SummarizerAgent);

    if (this.config.getDebugMode()) {
      debugLogger.log(
        `[AgentRegistry] Registered 9 specialized multi-agent system agents`,
      );
    }
  }

  /**
   * Registers an agent definition. If an agent with the same name exists,
   * it will be overwritten, respecting the precedence established by the
   * initialization order.
   */
  protected registerAgent<TOutput extends z.ZodTypeAny>(
    definition: AgentDefinition<TOutput>,
  ): void {
    // Basic validation
    if (!definition.name || !definition.description) {
      debugLogger.warn(
        `[AgentRegistry] Skipping invalid agent definition. Missing name or description.`,
      );
      return;
    }

    if (this.agents.has(definition.name) && this.config.getDebugMode()) {
      debugLogger.log(`[AgentRegistry] Overriding agent '${definition.name}'`);
    }

    this.agents.set(definition.name, definition);
  }

  /**
   * Retrieves an agent definition by name.
   */
  getDefinition(name: string): AgentDefinition | undefined {
    return this.agents.get(name);
  }

  /**
   * Returns all active agent definitions.
   */
  getAllDefinitions(): AgentDefinition[] {
    return Array.from(this.agents.values());
  }
}

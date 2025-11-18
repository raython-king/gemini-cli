/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  SkillDefinition,
  SkillMetadata,
  SkillSearchFilters,
  SkillContext,
  SkillResult,
  SkillExecutionOptions,
  SkillChain,
} from './types.js';
import { debugLogger } from '../utils/debugLogger.js';

/**
 * Skill Manager - Central registry and executor for skills.
 *
 * Manages skill discovery, registration, search, and execution.
 * Similar to Claude Code's skill system.
 */
export class SkillManager {
  private skills = new Map<string, SkillDefinition>();
  private chains = new Map<string, SkillChain>();

  constructor() {}

  /**
   * Register a skill.
   */
  registerSkill<TInput, TOutput>(
    skill: SkillDefinition<TInput, TOutput>,
  ): void {
    if (this.skills.has(skill.id)) {
      debugLogger.warn(`[SkillManager] Overwriting skill: ${skill.id}`);
    }

    // Validate skill definition
    this.validateSkill(skill);

    this.skills.set(skill.id, skill);
    debugLogger.log(`[SkillManager] Registered skill: ${skill.id}`);
  }

  /**
   * Register multiple skills.
   */
  registerSkills(skills: SkillDefinition[]): void {
    skills.forEach((skill) => this.registerSkill(skill));
  }

  /**
   * Get a skill by ID.
   */
  getSkill(id: string): SkillDefinition | undefined {
    return this.skills.get(id);
  }

  /**
   * Get all skills.
   */
  getAllSkills(): SkillDefinition[] {
    return Array.from(this.skills.values());
  }

  /**
   * Get skill metadata (without execute function).
   */
  getSkillMetadata(id: string): SkillMetadata | undefined {
    const skill = this.skills.get(id);
    if (!skill) return undefined;

    return {
      id: skill.id,
      name: skill.name,
      description: skill.description,
      category: skill.category,
      complexity: skill.complexity,
      version: skill.version,
      tags: skill.tags,
      requiredAgents: skill.requiredAgents,
    };
  }

  /**
   * Get all skill metadata.
   */
  getAllSkillMetadata(): SkillMetadata[] {
    return Array.from(this.skills.values()).map((skill) => ({
      id: skill.id,
      name: skill.name,
      description: skill.description,
      category: skill.category,
      complexity: skill.complexity,
      version: skill.version,
      tags: skill.tags,
      requiredAgents: skill.requiredAgents,
    }));
  }

  /**
   * Search for skills.
   */
  searchSkills(filters: SkillSearchFilters): SkillMetadata[] {
    let results = this.getAllSkillMetadata();

    // Filter by category
    if (filters.category) {
      results = results.filter((skill) => skill.category === filters.category);
    }

    // Filter by complexity
    if (filters.complexity) {
      results = results.filter(
        (skill) => skill.complexity === filters.complexity,
      );
    }

    // Filter by tags
    if (filters.tags && filters.tags.length > 0) {
      results = results.filter((skill) =>
        filters.tags!.some((tag) => skill.tags.includes(tag)),
      );
    }

    // Text search in name and description
    if (filters.query) {
      const query = filters.query.toLowerCase();
      results = results.filter(
        (skill) =>
          skill.name.toLowerCase().includes(query) ||
          skill.description.toLowerCase().includes(query) ||
          skill.tags.some((tag) => tag.toLowerCase().includes(query)),
      );
    }

    return results;
  }

  /**
   * Execute a skill.
   */
  async executeSkill<TInput, TOutput>(
    skillId: string,
    input: TInput,
    context: SkillContext,
    config: any,
    options: SkillExecutionOptions = {},
  ): Promise<SkillResult<TOutput>> {
    const skill = this.skills.get(skillId);

    if (!skill) {
      return {
        success: false,
        error: `Skill not found: ${skillId}`,
        metadata: {
          skillName: skillId,
          duration: 0,
          agentsUsed: [],
        },
      };
    }

    const startTime = Date.now();

    try {
      // Validate input
      const validatedInput = skill.inputSchema.parse(input);

      // Check prerequisites
      if (skill.prerequisites && skill.prerequisites.length > 0) {
        const missingPrereqs = skill.prerequisites.filter(
          (prereq) => !this.skills.has(prereq),
        );
        if (missingPrereqs.length > 0) {
          throw new Error(
            `Missing prerequisites: ${missingPrereqs.join(', ')}`,
          );
        }
      }

      if (options.verbose) {
        debugLogger.log(
          `[SkillManager] Executing skill: ${skill.name} (${skillId})`,
        );
      }

      // Dry run check
      if (options.dryRun) {
        return {
          success: true,
          data: undefined as any,
          metadata: {
            skillName: skill.name,
            duration: 0,
            agentsUsed: skill.requiredAgents,
          },
        };
      }

      // Execute skill with timeout
      let result: SkillResult<TOutput>;

      if (options.timeout) {
        result = await Promise.race([
          skill.execute(validatedInput, context, config),
          new Promise<SkillResult<TOutput>>((_, reject) =>
            setTimeout(
              () => reject(new Error('Skill execution timeout')),
              options.timeout,
            ),
          ),
        ]);
      } else {
        result = await skill.execute(validatedInput, context, config);
      }

      const duration = Date.now() - startTime;
      result.metadata.duration = duration;

      if (options.verbose) {
        debugLogger.log(
          `[SkillManager] Skill completed in ${duration}ms: ${skill.name}`,
        );
      }

      return result;
    } catch (error) {
      const duration = Date.now() - startTime;

      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        metadata: {
          skillName: skill.name,
          duration,
          agentsUsed: skill.requiredAgents,
        },
      };
    }
  }

  /**
   * Register a skill chain.
   */
  registerChain(chain: SkillChain): void {
    // Validate all skills in chain exist
    for (const step of chain.skills) {
      if (!this.skills.has(step.skillId)) {
        throw new Error(
          `Skill not found in chain: ${step.skillId}`,
        );
      }
    }

    this.chains.set(chain.id, chain);
    debugLogger.log(`[SkillManager] Registered chain: ${chain.id}`);
  }

  /**
   * Execute a skill chain.
   */
  async executeChain(
    chainId: string,
    context: SkillContext,
    config: any,
    options: SkillExecutionOptions = {},
  ): Promise<SkillResult[]> {
    const chain = this.chains.get(chainId);

    if (!chain) {
      throw new Error(`Chain not found: ${chainId}`);
    }

    const results: SkillResult[] = [];
    let previousOutput: any = {};

    for (let i = 0; i < chain.skills.length; i++) {
      const step = chain.skills[i];

      // Map previous output to current input if specified
      let input = step.inputs;
      if (i > 0 && step.outputMapping) {
        const mappedInput = { ...input };
        for (const [from, to] of Object.entries(step.outputMapping)) {
          if (previousOutput.data && from in previousOutput.data) {
            mappedInput[to] = previousOutput.data[from];
          }
        }
        input = mappedInput;
      }

      const result = await this.executeSkill(
        step.skillId,
        input,
        context,
        config,
        options,
      );

      results.push(result);

      if (!result.success) {
        debugLogger.log(
          `[SkillManager] Chain ${chainId} failed at step ${i + 1}`,
        );
        break;
      }

      previousOutput = result;
    }

    return results;
  }

  /**
   * Get available skill chains.
   */
  getChains(): SkillChain[] {
    return Array.from(this.chains.values());
  }

  /**
   * Get skill statistics.
   */
  getStats() {
    const allSkills = this.getAllSkillMetadata();

    const byCategory = new Map<string, number>();
    const byComplexity = new Map<string, number>();

    allSkills.forEach((skill) => {
      byCategory.set(skill.category, (byCategory.get(skill.category) || 0) + 1);
      byComplexity.set(
        skill.complexity,
        (byComplexity.get(skill.complexity) || 0) + 1,
      );
    });

    return {
      totalSkills: allSkills.length,
      totalChains: this.chains.size,
      byCategory: Object.fromEntries(byCategory),
      byComplexity: Object.fromEntries(byComplexity),
    };
  }

  /**
   * Validate skill definition.
   */
  private validateSkill(skill: SkillDefinition): void {
    if (!skill.id || typeof skill.id !== 'string') {
      throw new Error('Skill must have a valid ID');
    }

    if (!skill.name || typeof skill.name !== 'string') {
      throw new Error('Skill must have a valid name');
    }

    if (!skill.execute || typeof skill.execute !== 'function') {
      throw new Error('Skill must have an execute function');
    }

    if (!skill.inputSchema || !skill.outputSchema) {
      throw new Error('Skill must have input and output schemas');
    }
  }

  /**
   * Export skills to JSON (for sharing/documentation).
   */
  exportSkills(): string {
    const metadata = this.getAllSkillMetadata();
    return JSON.stringify(metadata, null, 2);
  }

  /**
   * Clear all skills (for testing).
   */
  clear(): void {
    this.skills.clear();
    this.chains.clear();
  }
}

/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Command } from 'commander';
import {
  type Config,
  SkillManager,
  BUILT_IN_SKILLS,
  SkillCategory,
  SkillComplexity,
  type SkillContext,
} from '@google/gemini-cli-core';
import chalk from 'chalk';
import ora from 'ora';
import inquirer from 'inquirer';

/**
 * Skill command - Manage and execute expert skills.
 *
 * Usage:
 *   gemini skill list
 *   gemini skill info llm.fine_tuning
 *   gemini skill run llm.fine_tuning
 *   gemini skill search --category llm
 */
export function createSkillCommand(config: Config): Command {
  const skillManager = new SkillManager();

  // Register built-in skills
  skillManager.registerSkills(BUILT_IN_SKILLS);

  const command = new Command('skill')
    .description('Manage and execute expert skills for LLM and CV development')
    .addCommand(createListCommand(skillManager))
    .addCommand(createInfoCommand(skillManager))
    .addCommand(createRunCommand(skillManager, config))
    .addCommand(createSearchCommand(skillManager));

  return command;
}

/**
 * List all available skills.
 */
function createListCommand(manager: SkillManager): Command {
  return new Command('list')
    .description('List all available skills')
    .option('--category <category>', 'Filter by category')
    .option('--json', 'Output as JSON', false)
    .action((options) => {
      const allSkills = manager.getAllSkillMetadata();

      let filtered = allSkills;
      if (options.category) {
        filtered = allSkills.filter(s => s.category === options.category);
      }

      if (options.json) {
        console.log(JSON.stringify(filtered, null, 2));
        return;
      }

      // Group by category
      const byCategory = new Map<string, typeof filtered>();
      filtered.forEach(skill => {
        if (!byCategory.has(skill.category)) {
          byCategory.set(skill.category, []);
        }
        byCategory.get(skill.category)!.push(skill);
      });

      console.log(chalk.bold('\n📚 Available Skills\n'));

      byCategory.forEach((skills, category) => {
        console.log(chalk.cyan(`\n${category.toUpperCase()}:`));
        skills.forEach(skill => {
          const complexity = getComplexityBadge(skill.complexity);
          console.log(`  ${chalk.bold(skill.id)} ${complexity}`);
          console.log(`    ${chalk.gray(skill.description)}`);
        });
      });

      console.log(chalk.gray(`\nTotal: ${filtered.length} skills`));
      console.log(chalk.gray(`Use 'gemini skill info <skill-id>' for details\n`));
    });
}

/**
 * Show detailed skill information.
 */
function createInfoCommand(manager: SkillManager): Command {
  return new Command('info')
    .description('Show detailed information about a skill')
    .argument('<skill-id>', 'Skill ID')
    .action((skillId) => {
      const skill = manager.getSkill(skillId);

      if (!skill) {
        console.error(chalk.red(`Skill not found: ${skillId}`));
        process.exit(1);
      }

      console.log(chalk.bold(`\n📖 ${skill.name}\n`));
      console.log(chalk.cyan('Description:'));
      console.log(`  ${skill.description}\n`);

      console.log(chalk.cyan('Details:'));
      console.log(`  ID: ${skill.id}`);
      console.log(`  Category: ${skill.category}`);
      console.log(`  Complexity: ${getComplexityBadge(skill.complexity)}`);
      console.log(`  Version: ${skill.version}`);
      console.log(`  Estimated Time: ${skill.estimatedTime || 'N/A'}`);
      console.log(`  Required Agents: ${skill.requiredAgents.join(', ')}`);
      console.log(`  Tags: ${skill.tags.join(', ')}\n`);

      console.log(chalk.cyan('Usage:'));
      console.log(chalk.gray(skill.usage));

      if (skill.examples && skill.examples.length > 0) {
        console.log(chalk.cyan('\nExamples:'));
        skill.examples.forEach((ex, i) => {
          console.log(`\n  ${i + 1}. ${chalk.bold(ex.title)}`);
          console.log(`     ${ex.description}`);
          console.log(chalk.gray(`     Input: ${JSON.stringify(ex.input, null, 2).split('\n').join('\n     ')}`));
        });
      }

      console.log(chalk.gray(`\nRun: gemini skill run ${skillId}\n`));
    });
}

/**
 * Run a skill.
 */
function createRunCommand(manager: SkillManager, config: Config): Command {
  return new Command('run')
    .description('Execute a skill')
    .argument('<skill-id>', 'Skill ID')
    .option('--input <json>', 'Input as JSON string')
    .option('--input-file <path>', 'Input from JSON file')
    .option('--interactive', 'Interactive input mode', false)
    .option('--verbose', 'Verbose output', false)
    .option('--dry-run', 'Validate without executing', false)
    .action(async (skillId, options) => {
      const skill = manager.getSkill(skillId);

      if (!skill) {
        console.error(chalk.red(`Skill not found: ${skillId}`));
        process.exit(1);
      }

      let input: any;

      // Get input
      if (options.inputFile) {
        const fs = await import('fs');
        input = JSON.parse(fs.readFileSync(options.inputFile, 'utf-8'));
      } else if (options.input) {
        input = JSON.parse(options.input);
      } else if (options.interactive) {
        // Interactive mode - prompt for each input field
        console.log(chalk.bold(`\n🎯 ${skill.name}\n`));
        console.log(chalk.gray('Please provide the required inputs:\n'));

        // This would need proper schema-to-prompt conversion
        // For now, show example
        if (skill.examples && skill.examples.length > 0) {
          console.log(chalk.yellow('Example input:'));
          console.log(JSON.stringify(skill.examples[0].input, null, 2));
          console.log();
        }

        const { inputJson } = await inquirer.prompt([{
          type: 'editor',
          name: 'inputJson',
          message: 'Edit the input JSON:',
          default: skill.examples?.[0]?.input ? JSON.stringify(skill.examples[0].input, null, 2) : '{}',
        }]);

        input = JSON.parse(inputJson);
      } else {
        console.error(chalk.red('Please provide input via --input, --input-file, or --interactive'));
        process.exit(1);
      }

      const spinner = ora(`Executing ${skill.name}...`).start();

      try {
        const context: SkillContext = {
          workingDir: process.cwd(),
        };

        const result = await manager.executeSkill(
          skillId,
          input,
          context,
          config,
          {
            verbose: options.verbose,
            dryRun: options.dryRun,
            onProgress: (progress) => {
              spinner.text = `${skill.name}: ${progress.message} (${progress.progress}%)`;
            },
          },
        );

        if (result.success) {
          spinner.succeed(chalk.green('Skill executed successfully!'));

          console.log(chalk.bold('\n📊 Results:\n'));
          console.log(JSON.stringify(result.data, null, 2));

          if (result.artifacts && result.artifacts.length > 0) {
            console.log(chalk.cyan('\n📁 Artifacts:'));
            result.artifacts.forEach(art => console.log(`  - ${art}`));
          }

          if (result.suggestions && result.suggestions.length > 0) {
            console.log(chalk.cyan('\n💡 Suggestions:'));
            result.suggestions.forEach(sug => console.log(`  - ${sug}`));
          }

          console.log(chalk.gray(`\nDuration: ${result.metadata.duration}ms`));
          console.log(chalk.gray(`Agents used: ${result.metadata.agentsUsed.join(', ')}\n`));
        } else {
          spinner.fail(chalk.red('Skill execution failed'));
          console.error(chalk.red(`\nError: ${result.error}\n`));
          process.exit(1);
        }
      } catch (error) {
        spinner.fail(chalk.red('Skill execution failed'));
        console.error(chalk.red(`\nError: ${error instanceof Error ? error.message : String(error)}\n`));
        process.exit(1);
      }
    });
}

/**
 * Search for skills.
 */
function createSearchCommand(manager: SkillManager): Command {
  return new Command('search')
    .description('Search for skills')
    .option('--category <category>', 'Filter by category')
    .option('--complexity <level>', 'Filter by complexity')
    .option('--tag <tag>', 'Filter by tag')
    .option('--query <text>', 'Text search')
    .action((options) => {
      const results = manager.searchSkills({
        category: options.category as any,
        complexity: options.complexity as any,
        tags: options.tag ? [options.tag] : undefined,
        query: options.query,
      });

      if (results.length === 0) {
        console.log(chalk.yellow('\nNo skills found matching the criteria.\n'));
        return;
      }

      console.log(chalk.bold(`\n🔍 Found ${results.length} skills:\n`));

      results.forEach(skill => {
        const complexity = getComplexityBadge(skill.complexity);
        console.log(`  ${chalk.bold(skill.id)} ${complexity}`);
        console.log(`    ${chalk.gray(skill.description)}`);
        console.log(chalk.gray(`    Tags: ${skill.tags.join(', ')}\n`));
      });
    });
}

/**
 * Get complexity badge.
 */
function getComplexityBadge(complexity: SkillComplexity): string {
  switch (complexity) {
    case SkillComplexity.BASIC:
      return chalk.green('[Basic]');
    case SkillComplexity.INTERMEDIATE:
      return chalk.yellow('[Intermediate]');
    case SkillComplexity.ADVANCED:
      return chalk.orange('[Advanced]');
    case SkillComplexity.EXPERT:
      return chalk.red('[Expert]');
    default:
      return '';
  }
}

/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Command } from 'commander';
import {
  type Config,
  CollaborativeExecutor,
  AlgorithmWorkflowOrchestrator,
  AlgorithmWorkflowType,
  AlgorithmDomain,
  type AlgorithmWorkflowConfig,
} from '@google/gemini-cli-core';
import chalk from 'chalk';
import ora from 'ora';
import inquirer from 'inquirer';

/**
 * Research command for algorithm development workflows.
 *
 * Usage:
 *   gemini research llm "analyze transformer attention mechanisms"
 *   gemini research cv "implement YOLOv8 from paper"
 *   gemini research --workflow reproduce-paper --paper ./paper.pdf
 */
export function createResearchCommand(config: Config): Command {
  const command = new Command('research')
    .description('Expert algorithm research and development')
    .argument('[domain]', 'Algorithm domain: llm, cv, rl, ml')
    .argument('[objective]', 'Research objective or question')
    .option(
      '--workflow <type>',
      'Workflow type: research, reproduce, optimize, benchmark, study, deploy',
    )
    .option('--paper <path>', 'Path to paper(s) to analyze (comma-separated)')
    .option('--codebase <path>', 'Path to existing codebase')
    .option('--dataset <path>', 'Path to dataset')
    .option('--budget <amount>', 'Compute budget constraint')
    .option('--hardware <type>', 'Available hardware (e.g., "gpu,tpu")')
    .option('--interactive', 'Interactive workflow configuration', false)
    .option('--preview', 'Preview workflow without execution', false)
    .action(async (domain, objective, options) => {
      let workflowConfig: AlgorithmWorkflowConfig;

      if (options.interactive || (!domain && !objective)) {
        // Interactive mode
        workflowConfig = await interactiveWorkflowConfig();
      } else {
        // Direct mode
        workflowConfig = {
          workflowType: parseWorkflowType(options.workflow || 'research'),
          domain: parseDomain(domain || 'general'),
          objective: objective || 'Conduct algorithm research',
          papers: options.paper?.split(','),
          codebase: options.codebase,
          dataset: options.dataset,
          constraints: {
            computeBudget: options.budget,
            hardware: options.hardware?.split(','),
          },
        };
      }

      const spinner = ora('Initializing algorithm research system...').start();

      try {
        // Create workflow
        const orchestrator = new AlgorithmWorkflowOrchestrator();
        const workflow = orchestrator.createAlgorithmWorkflow(workflowConfig);

        spinner.succeed('Workflow created');

        // Display workflow
        displayWorkflow(workflow, workflowConfig);

        if (options.preview) {
          console.log(chalk.cyan('\n Preview mode - workflow not executed'));
          return;
        }

        // Confirm execution
        const { confirm } = await inquirer.prompt([
          {
            type: 'confirm',
            name: 'confirm',
            message: 'Execute this workflow?',
            default: true,
          },
        ]);

        if (!confirm) {
          console.log(chalk.yellow('Workflow cancelled'));
          return;
        }

        spinner.start('Executing workflow...');

        // Execute with collaborative system
        const executor = new CollaborativeExecutor(config);

        const result = await executor.execute(workflowConfig.objective, {
          maxSubtasks: workflow.subtasks.length,
          maxConcurrency: 3,
          enableContextSharing: true,
          enableMemory: true,
          onEvent: (event) => {
            switch (event.type) {
              case 'decomposition_complete':
                spinner.text = 'Workflow decomposed, starting execution...';
                break;

              case 'stage_start': {
                const data = event.data as any;
                spinner.text = `Stage ${data.stage}/${data.totalStages}...`;
                break;
              }

              case 'subtask_complete': {
                const data = event.data as any;
                console.log(chalk.green(`  ✓ ${data.agentName} completed`));
                break;
              }

              case 'subtask_failed': {
                const data = event.data as any;
                console.log(
                  chalk.red(`  ✗ ${data.agentName} failed: ${data.error}`),
                );
                break;
              }

              case 'execution_complete': {
                const data = event.data as any;
                if (data.success) {
                  spinner.succeed(
                    chalk.green(
                      `Workflow completed in ${Math.round(data.duration / 1000)}s`,
                    ),
                  );
                } else {
                  spinner.fail(chalk.red('Workflow completed with errors'));
                }
                break;
              }
            }
          },
        });

        // Display results
        displayResults(result, workflowConfig);

        process.exit(result.status.success ? 0 : 1);
      } catch (error) {
        spinner.fail(chalk.red('Research workflow failed'));
        console.error(
          chalk.red(`Error: ${error instanceof Error ? error.message : String(error)}`),
        );
        process.exit(1);
      }
    });

  return command;
}

/**
 * Interactive workflow configuration.
 */
async function interactiveWorkflowConfig(): Promise<AlgorithmWorkflowConfig> {
  const answers = await inquirer.prompt([
    {
      type: 'list',
      name: 'domain',
      message: 'Select algorithm domain:',
      choices: [
        { name: 'Large Language Models (LLM)', value: 'llm' },
        { name: 'Computer Vision (CV)', value: 'cv' },
        { name: 'Reinforcement Learning (RL)', value: 'rl' },
        { name: 'General Machine Learning', value: 'general_ml' },
      ],
    },
    {
      type: 'list',
      name: 'workflowType',
      message: 'Select workflow type:',
      choices: [
        {
          name: 'Research & Implement - Full research to implementation',
          value: 'research_and_implement',
        },
        {
          name: 'Reproduce Paper - Reproduce paper results',
          value: 'reproduce_paper',
        },
        {
          name: 'Optimize Model - Improve existing model',
          value: 'optimize_model',
        },
        {
          name: 'Benchmark Comparison - Compare multiple models',
          value: 'benchmark_comparison',
        },
        {
          name: 'Experimental Study - Comprehensive study',
          value: 'experimental_study',
        },
        {
          name: 'Model Deployment - Prepare for production',
          value: 'model_deployment',
        },
      ],
    },
    {
      type: 'input',
      name: 'objective',
      message: 'Research objective or question:',
      validate: (input) => (input.trim() ? true : 'Objective is required'),
    },
    {
      type: 'input',
      name: 'papers',
      message: 'Papers to analyze (comma-separated paths, optional):',
    },
    {
      type: 'input',
      name: 'codebase',
      message: 'Existing codebase path (optional):',
    },
    {
      type: 'input',
      name: 'dataset',
      message: 'Dataset path (optional):',
    },
  ]);

  return {
    workflowType: answers.workflowType as AlgorithmWorkflowType,
    domain: answers.domain as AlgorithmDomain,
    objective: answers.objective,
    papers: answers.papers
      ? answers.papers.split(',').map((p: string) => p.trim())
      : undefined,
    codebase: answers.codebase || undefined,
    dataset: answers.dataset || undefined,
  };
}

/**
 * Display workflow summary.
 */
function displayWorkflow(workflow: any, config: AlgorithmWorkflowConfig) {
  console.log();
  console.log(chalk.bold('═'.repeat(70)));
  console.log(chalk.bold.cyan('        Algorithm Research Workflow'));
  console.log(chalk.bold('═'.repeat(70)));
  console.log();

  console.log(chalk.bold('Configuration:'));
  console.log(`  Domain:       ${chalk.cyan(config.domain.toUpperCase())}`);
  console.log(`  Workflow:     ${chalk.cyan(config.workflowType)}`);
  console.log(`  Objective:    ${chalk.cyan(config.objective)}`);
  if (config.papers) {
    console.log(`  Papers:       ${chalk.cyan(config.papers.length)} files`);
  }
  console.log();

  console.log(chalk.bold('Workflow Summary:'));
  console.log(`  Subtasks:     ${chalk.cyan(workflow.subtasks.length)}`);
  console.log(`  Stages:       ${chalk.cyan(workflow.executionPlan.length)}`);
  console.log(`  Duration:     ${chalk.cyan(workflow.estimatedDuration)}`);
  console.log();

  console.log(chalk.bold('Execution Plan:'));
  for (let i = 0; i < Math.min(workflow.subtasks.length, 10); i++) {
    const st = workflow.subtasks[i];
    const icon = st.canParallelize ? '⚡' : '→';
    console.log(
      `  ${chalk.gray(i + 1 + '.')} ${icon} ${st.title} ${chalk.gray(`(${st.recommendedAgents[0]})`)}`,
    );
  }

  if (workflow.subtasks.length > 10) {
    console.log(
      chalk.gray(`  ... and ${workflow.subtasks.length - 10} more steps`),
    );
  }
  console.log();
}

/**
 * Display execution results.
 */
function displayResults(result: any, config: AlgorithmWorkflowConfig) {
  console.log();
  console.log(chalk.bold('═'.repeat(70)));
  console.log(chalk.bold('        Execution Results'));
  console.log(chalk.bold('═'.repeat(70)));
  console.log();

  const completed = result.status.subtasks.filter(
    (s: any) => s.status === 'completed',
  ).length;
  const failed = result.status.subtasks.filter(
    (s: any) => s.status === 'failed',
  ).length;

  console.log(chalk.bold('Summary:'));
  console.log(`  Status:       ${result.status.success ? chalk.green('✓ Success') : chalk.red('✗ Failed')}`);
  console.log(`  Progress:     ${chalk.cyan(result.status.progress + '%')}`);
  console.log(
    `  Completed:    ${chalk.green(completed)}/${result.status.subtasks.length}`,
  );
  if (failed > 0) {
    console.log(`  Failed:       ${chalk.red(failed)}`);
  }
  console.log();

  if (result.contextSummary) {
    console.log(chalk.bold('Research Insights:'));
    console.log(chalk.gray(result.contextSummary.slice(0, 300) + '...'));
    console.log();
  }

  console.log(
    chalk.bold.cyan(
      'Complete results have been saved to the execution context.',
    ),
  );
}

/**
 * Parse workflow type from string.
 */
function parseWorkflowType(type: string): AlgorithmWorkflowType {
  const mapping: Record<string, AlgorithmWorkflowType> = {
    research: AlgorithmWorkflowType.RESEARCH_AND_IMPLEMENT,
    'research-and-implement': AlgorithmWorkflowType.RESEARCH_AND_IMPLEMENT,
    reproduce: AlgorithmWorkflowType.REPRODUCE_PAPER,
    'reproduce-paper': AlgorithmWorkflowType.REPRODUCE_PAPER,
    optimize: AlgorithmWorkflowType.OPTIMIZE_MODEL,
    'optimize-model': AlgorithmWorkflowType.OPTIMIZE_MODEL,
    benchmark: AlgorithmWorkflowType.BENCHMARK_COMPARISON,
    'benchmark-comparison': AlgorithmWorkflowType.BENCHMARK_COMPARISON,
    study: AlgorithmWorkflowType.EXPERIMENTAL_STUDY,
    'experimental-study': AlgorithmWorkflowType.EXPERIMENTAL_STUDY,
    deploy: AlgorithmWorkflowType.MODEL_DEPLOYMENT,
    'model-deployment': AlgorithmWorkflowType.MODEL_DEPLOYMENT,
  };

  return (
    mapping[type.toLowerCase()] || AlgorithmWorkflowType.RESEARCH_AND_IMPLEMENT
  );
}

/**
 * Parse domain from string.
 */
function parseDomain(domain: string): AlgorithmDomain {
  const mapping: Record<string, AlgorithmDomain> = {
    llm: AlgorithmDomain.LLM,
    cv: AlgorithmDomain.CV,
    rl: AlgorithmDomain.RL,
    ml: AlgorithmDomain.GENERAL_ML,
    general: AlgorithmDomain.GENERAL_ML,
  };

  return mapping[domain.toLowerCase()] || AlgorithmDomain.GENERAL_ML;
}

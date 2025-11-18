/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Command } from 'commander';
import { CollaborativeExecutor, type Config } from '@google/gemini-cli-core';
import chalk from 'chalk';
import ora from 'ora';

/**
 * Multi-agent collaboration command.
 *
 * Usage:
 *   gemini collaborate "implement user authentication"
 *   gemini collaborate --preview "refactor the login module"
 *   gemini collaborate --max-subtasks 5 "fix all bugs in auth"
 */
export function createCollaborateCommand(config: Config): Command {
  const command = new Command('collaborate')
    .description('Execute complex tasks using multi-agent collaboration')
    .argument('<task>', 'The task to execute')
    .option(
      '--preview',
      'Preview the decomposition without executing',
      false,
    )
    .option(
      '--max-subtasks <number>',
      'Maximum number of subtasks to create',
      '10',
    )
    .option(
      '--max-concurrency <number>',
      'Maximum number of concurrent agents',
      '3',
    )
    .option(
      '--enable-context',
      'Enable context sharing between agents',
      true,
    )
    .option(
      '--enable-memory',
      'Enable agent memory',
      true,
    )
    .option(
      '--verbose',
      'Show detailed agent activity',
      false,
    )
    .action(async (task: string, options) => {
      const executor = new CollaborativeExecutor(config);
      const spinner = ora('Initializing multi-agent system...').start();

      try {
        const result = await executor.execute(task, {
          maxSubtasks: parseInt(options.maxSubtasks),
          maxConcurrency: parseInt(options.maxConcurrency),
          enableContextSharing: options.enableContext,
          enableMemory: options.enableMemory,
          previewOnly: options.preview,
          onEvent: (event) => {
            if (options.verbose) {
              console.log(
                chalk.gray(
                  `[${event.timestamp.toLocaleTimeString()}] ${event.type}`,
                ),
              );
            }

            switch (event.type) {
              case 'decomposition_complete': {
                const data = event.data as any;
                spinner.succeed(
                  `Task decomposed into ${data.subtaskCount} subtasks, ${data.stageCount} stages`,
                );
                console.log(
                  chalk.cyan(`Estimated duration: ${data.estimatedDuration}`),
                );

                if (!options.preview) {
                  spinner.start('Executing agents...');
                }
                break;
              }

              case 'stage_start': {
                const data = event.data as any;
                spinner.text = `Executing stage ${data.stage}/${data.totalStages}...`;
                break;
              }

              case 'subtask_complete': {
                const data = event.data as any;
                if (options.verbose) {
                  console.log(
                    chalk.green(`✓ Completed: ${data.agentName}`),
                  );
                }
                break;
              }

              case 'subtask_failed': {
                const data = event.data as any;
                spinner.warn(
                  chalk.yellow(
                    `Subtask failed (${data.agentName}): ${data.error}`,
                  ),
                );
                break;
              }

              case 'execution_complete': {
                const data = event.data as any;
                if (data.success) {
                  spinner.succeed(
                    chalk.green(
                      `All tasks completed successfully in ${Math.round(data.duration / 1000)}s`,
                    ),
                  );
                } else {
                  spinner.fail(
                    chalk.red('Execution completed with errors'),
                  );
                }
                break;
              }
            }
          },
        });

        // Display results
        console.log();
        console.log(chalk.bold('═'.repeat(60)));
        console.log(chalk.bold('Execution Summary'));
        console.log(chalk.bold('═'.repeat(60)));
        console.log();

        console.log(`Task: ${chalk.cyan(result.task)}`);
        console.log(
          `Status: ${result.status.success ? chalk.green('✓ Success') : chalk.red('✗ Failed')}`,
        );
        console.log(`Progress: ${result.status.progress}%`);
        console.log(
          `Subtasks: ${result.status.subtasks.length} (${result.status.subtasks.filter((s) => s.status === 'completed').length} completed)`,
        );
        console.log();

        // Show subtask details
        console.log(chalk.bold('Subtasks:'));
        for (const st of result.status.subtasks) {
          const statusIcon =
            st.status === 'completed'
              ? chalk.green('✓')
              : st.status === 'failed'
                ? chalk.red('✗')
                : st.status === 'running'
                  ? chalk.yellow('⋯')
                  : chalk.gray('○');

          console.log(
            `  ${statusIcon} ${st.subtask.title} (${st.subtask.recommendedAgents.join(', ')})`,
          );

          if (st.error) {
            console.log(chalk.red(`    Error: ${st.error}`));
          }
        }

        console.log();

        // Show context summary if enabled
        if (result.contextSummary && options.verbose) {
          console.log(chalk.bold('Context Summary:'));
          console.log(result.contextSummary);
          console.log();
        }

        // Show memory summary if enabled
        if (result.memorySummary && options.verbose) {
          console.log(chalk.bold('Memory Summary:'));
          console.log(result.memorySummary);
          console.log();
        }

        // Exit with appropriate code
        process.exit(result.status.success ? 0 : 1);
      } catch (error) {
        spinner.fail(chalk.red('Collaboration failed'));
        console.error(
          chalk.red(`Error: ${error instanceof Error ? error.message : String(error)}`),
        );
        process.exit(1);
      }
    });

  return command;
}

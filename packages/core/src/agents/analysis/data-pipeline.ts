/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Data Pipeline - Tools for automated data flow processing.
 *
 * This module provides utilities for creating data processing pipelines
 * that can be used with DataFlowAgent and other analysis agents.
 */

/**
 * Transform function type for data pipeline.
 */
export type TransformFn<TInput = unknown, TOutput = unknown> = (
  input: TInput,
) => TOutput | Promise<TOutput>;

/**
 * Validation function type.
 */
export type ValidatorFn<T = unknown> = (
  data: T,
) => ValidationResult | Promise<ValidationResult>;

/**
 * Result of data validation.
 */
export interface ValidationResult {
  valid: boolean;
  errors: Array<{
    field?: string;
    message: string;
    severity: 'error' | 'warning';
  }>;
}

/**
 * Pipeline step definition.
 */
export interface PipelineStep<TInput = unknown, TOutput = unknown> {
  name: string;
  description?: string;
  transform: TransformFn<TInput, TOutput>;
  validate?: ValidatorFn<TOutput>;
}

/**
 * Pipeline execution result.
 */
export interface PipelineResult<T = unknown> {
  success: boolean;
  data?: T;
  errors: Array<{
    step: string;
    message: string;
    error?: Error;
  }>;
  metadata: {
    stepsExecuted: number;
    totalSteps: number;
    executionTime: number;
    validationResults: Record<string, ValidationResult>;
  };
}

/**
 * DataPipeline - Composable data processing pipeline.
 *
 * @example
 * ```typescript
 * const pipeline = new DataPipeline()
 *   .addStep({
 *     name: 'parse',
 *     transform: (raw: string) => JSON.parse(raw),
 *   })
 *   .addStep({
 *     name: 'extract',
 *     transform: (data: any) => data.users,
 *   })
 *   .addStep({
 *     name: 'filter',
 *     transform: (users: User[]) => users.filter(u => u.active),
 *     validate: (users) => ({
 *       valid: users.length > 0,
 *       errors: users.length === 0 ? [{ message: 'No active users' }] : [],
 *     }),
 *   });
 *
 * const result = await pipeline.execute(rawData);
 * ```
 */
export class DataPipeline<TInput = unknown, TOutput = unknown> {
  private steps: Array<PipelineStep<unknown, unknown>> = [];

  /**
   * Add a transformation step to the pipeline.
   */
  addStep<TStepInput, TStepOutput>(
    step: PipelineStep<TStepInput, TStepOutput>,
  ): DataPipeline<TInput, TStepOutput> {
    this.steps.push(step as PipelineStep<unknown, unknown>);
    return this as unknown as DataPipeline<TInput, TStepOutput>;
  }

  /**
   * Execute the pipeline with the given input data.
   */
  async execute(input: TInput): Promise<PipelineResult<TOutput>> {
    const startTime = Date.now();
    const errors: PipelineResult<TOutput>['errors'] = [];
    const validationResults: Record<string, ValidationResult> = {};

    let currentData: unknown = input;
    let stepsExecuted = 0;

    try {
      for (const step of this.steps) {
        try {
          // Execute transformation
          currentData = await step.transform(currentData);
          stepsExecuted++;

          // Run validation if provided
          if (step.validate) {
            const validation = await step.validate(currentData);
            validationResults[step.name] = validation;

            if (!validation.valid) {
              const hasErrors = validation.errors.some(
                (e) => e.severity === 'error',
              );
              if (hasErrors) {
                errors.push({
                  step: step.name,
                  message: `Validation failed: ${validation.errors.map((e) => e.message).join(', ')}`,
                });
                // Stop pipeline on validation errors
                break;
              }
            }
          }
        } catch (error) {
          errors.push({
            step: step.name,
            message: `Error in step "${step.name}": ${error instanceof Error ? error.message : String(error)}`,
            error: error instanceof Error ? error : undefined,
          });
          break;
        }
      }

      const executionTime = Date.now() - startTime;

      return {
        success: errors.length === 0 && stepsExecuted === this.steps.length,
        data: errors.length === 0 ? (currentData as TOutput) : undefined,
        errors,
        metadata: {
          stepsExecuted,
          totalSteps: this.steps.length,
          executionTime,
          validationResults,
        },
      };
    } catch (error) {
      errors.push({
        step: 'pipeline',
        message: `Pipeline execution failed: ${error instanceof Error ? error.message : String(error)}`,
        error: error instanceof Error ? error : undefined,
      });

      return {
        success: false,
        errors,
        metadata: {
          stepsExecuted,
          totalSteps: this.steps.length,
          executionTime: Date.now() - startTime,
          validationResults,
        },
      };
    }
  }

  /**
   * Get pipeline step information.
   */
  getSteps(): Array<{ name: string; description?: string }> {
    return this.steps.map((step) => ({
      name: step.name,
      description: step.description,
    }));
  }
}

/**
 * Common data transformations.
 */
export const Transforms = {
  /**
   * Parse JSON string to object.
   */
  parseJSON: <T = unknown>(input: string): T => JSON.parse(input) as T,

  /**
   * Extract specific fields from object.
   */
  extractFields:
    <T extends Record<string, unknown>>(fields: Array<keyof T>) =>
    (input: T): Partial<T> => {
      const result: Partial<T> = {};
      for (const field of fields) {
        if (field in input) {
          result[field] = input[field];
        }
      }
      return result;
    },

  /**
   * Filter array by predicate.
   */
  filter:
    <T>(predicate: (item: T) => boolean) =>
    (input: T[]): T[] => input.filter(predicate),

  /**
   * Map array elements.
   */
  map:
    <TInput, TOutput>(mapper: (item: TInput) => TOutput) =>
    (input: TInput[]): TOutput[] => input.map(mapper),

  /**
   * Flatten nested arrays.
   */
  flatten: <T>(input: T[][]): T[] => input.flat(),

  /**
   * Group items by key.
   */
  groupBy:
    <T>(keyFn: (item: T) => string) =>
    (input: T[]): Record<string, T[]> => {
      const result: Record<string, T[]> = {};
      for (const item of input) {
        const key = keyFn(item);
        if (!result[key]) {
          result[key] = [];
        }
        result[key].push(item);
      }
      return result;
    },

  /**
   * Deduplicate array by key.
   */
  deduplicate:
    <T>(keyFn: (item: T) => string | number) =>
    (input: T[]): T[] => {
      const seen = new Set<string | number>();
      return input.filter((item) => {
        const key = keyFn(item);
        if (seen.has(key)) {
          return false;
        }
        seen.add(key);
        return true;
      });
    },

  /**
   * Sort array by comparator.
   */
  sort:
    <T>(compareFn: (a: T, b: T) => number) =>
    (input: T[]): T[] => [...input].sort(compareFn),
};

/**
 * Common validators.
 */
export const Validators = {
  /**
   * Validate non-empty array.
   */
  nonEmpty: <T>(data: T[]): ValidationResult => ({
      valid: data.length > 0,
      errors:
        data.length === 0
          ? [{ message: 'Array is empty', severity: 'error' }]
          : [],
    }),

  /**
   * Validate required fields.
   */
  requiredFields:
    <T extends Record<string, unknown>>(fields: Array<keyof T>) =>
    (data: T): ValidationResult => {
      const errors: ValidationResult['errors'] = [];
      for (const field of fields) {
        if (
          !(field in data) ||
          data[field] === undefined ||
          data[field] === null
        ) {
          errors.push({
            field: String(field),
            message: `Required field "${String(field)}" is missing`,
            severity: 'error',
          });
        }
      }
      return {
        valid: errors.length === 0,
        errors,
      };
    },

  /**
   * Validate data type.
   */
  type:
    <T>(expectedType: string, checkFn: (value: unknown) => value is T) =>
    (data: unknown): ValidationResult => {
      const valid = checkFn(data);
      return {
        valid,
        errors: valid
          ? []
          : [
              {
                message: `Expected type ${expectedType}`,
                severity: 'error',
              },
            ],
      };
    },

  /**
   * Validate with custom function.
   */
  custom:
    <T>(validate: (data: T) => boolean, errorMessage: string) =>
    (data: T): ValidationResult => {
      const valid = validate(data);
      return {
        valid,
        errors: valid ? [] : [{ message: errorMessage, severity: 'error' }],
      };
    },

  /**
   * Combine multiple validators (all must pass).
   */
  all:
    <T>(...validators: Array<ValidatorFn<T>>) =>
    async (data: T): Promise<ValidationResult> => {
      const results = await Promise.all(
        validators.map((validator) => validator(data)),
      );

      const allErrors = results.flatMap((r) => r.errors);
      return {
        valid: results.every((r) => r.valid),
        errors: allErrors,
      };
    },
};

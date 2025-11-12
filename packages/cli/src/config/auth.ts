/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AuthType } from '@google/gemini-cli-core';
import { loadEnvironment, loadSettings } from './settings.js';

export function validateAuthMethod(authMethod: string): string | null {
  loadEnvironment(loadSettings().merged);
  if (
    authMethod === AuthType.LOGIN_WITH_GOOGLE ||
    authMethod === AuthType.CLOUD_SHELL
  ) {
    return null;
  }

  if (authMethod === AuthType.USE_GEMINI) {
    return null;
  }

  if (authMethod === AuthType.USE_VERTEX_AI) {
    const hasVertexProjectLocationConfig =
      !!process.env['GOOGLE_CLOUD_PROJECT'] &&
      !!process.env['GOOGLE_CLOUD_LOCATION'];
    const hasGoogleApiKey = !!process.env['GOOGLE_API_KEY'];
    if (!hasVertexProjectLocationConfig && !hasGoogleApiKey) {
      return (
        'When using Vertex AI, you must specify either:\n' +
        '• GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.\n' +
        '• GOOGLE_API_KEY environment variable (if using express mode).\n' +
        'Update your environment and try again (no reload needed if using .env)!'
      );
    }
    return null;
  }

  if (authMethod === AuthType.USE_OPENAI_COMPATIBLE) {
    // OpenAI-compatible mode is always valid - it will use defaults if env vars not set
    // Just provide helpful guidance about optional configuration
    const hasBaseUrl = !!process.env['OPENAI_BASE_URL'];
    const hasApiKey = !!process.env['OPENAI_API_KEY'];
    const hasModel = !!process.env['OPENAI_MODEL'];

    if (!hasBaseUrl && !hasApiKey && !hasModel) {
      console.log(
        '\n📝 Note: Using OpenAI-compatible mode with defaults:\n' +
          '  • OPENAI_BASE_URL: http://localhost:8000/v1\n' +
          '  • OPENAI_API_KEY: dummy-key\n' +
          '  • OPENAI_MODEL: gpt-3.5-turbo\n' +
          '\nTo customize, set these environment variables.\n' +
          'See OPENAI_INTEGRATION.md for more details.\n',
      );
    }
    return null;
  }

  return 'Invalid auth method selected.';
}

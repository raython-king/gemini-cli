# Gemini CLI - Project Summary

## 📋 Project Overview

**Gemini CLI** is an open-source command-line interface tool that brings the power of Google's Gemini AI model directly into the terminal. It provides developers with a lightweight, terminal-first AI assistant capable of understanding codebases, generating code, automating tasks, and integrating with various development workflows.

- **License**: Apache 2.0
- **Primary Language**: TypeScript
- **Node.js Requirement**: >= 20.0.0
- **Repository**: https://github.com/google-gemini/gemini-cli
- **NPM Package**: `@google/gemini-cli`
- **Current Version**: 0.15.0 (nightly builds available)

---

## 🏗️ Architecture

Gemini CLI follows a modular monorepo architecture with clear separation of concerns:

### Core Components

```
gemini-cli/
├── packages/
│   ├── cli/              # User-facing CLI interface
│   ├── core/             # Backend logic and AI orchestration
│   ├── a2a-server/       # Agent-to-Agent server
│   ├── test-utils/       # Shared testing utilities
│   └── vscode-ide-companion/  # VS Code integration
├── docs/                 # Comprehensive documentation
├── integration-tests/    # End-to-end tests
└── scripts/              # Build and automation scripts
```

### 1. **CLI Package** (`packages/cli`)

**Purpose**: Handles user interaction, input processing, and output rendering.

**Key Responsibilities**:
- Terminal UI rendering using React and Ink
- Command processing and history management
- Configuration management
- Theme and display customization
- Extension system management

**Main Dependencies**:
- `ink` (React for CLI)
- `@google/gemini-cli-core`
- `yargs` (argument parsing)
- `prompts` (interactive prompts)

### 2. **Core Package** (`packages/core`)

**Purpose**: Backend orchestration layer that manages AI interactions and tool execution.

**Key Responsibilities**:
- Gemini API client integration
- Prompt construction and context management
- Tool registry and execution
- Conversation state management
- MCP (Model Context Protocol) integration
- Agent system coordination

**Main Dependencies**:
- `@google/genai` (Google Generative AI SDK)
- `@modelcontextprotocol/sdk`

### 3. **Tools System** (`packages/core/src/tools/`)

A comprehensive set of built-in tools that extend Gemini's capabilities:

| Tool | Purpose | Type |
|------|---------|------|
| **edit.ts** | File editing with smart replacements | Modifiable |
| **glob.ts** | File pattern matching | Read-only |
| **grep.ts** / **ripGrep.ts** | Code search | Read-only |
| **ls.ts** | Directory listing | Read-only |
| **read-file.ts** | File reading | Read-only |
| **write-file.ts** | File creation/writing | Modifiable |
| **shell.ts** | Shell command execution | Modifiable |
| **web-fetch.ts** | HTTP requests | Read-only |
| **web-search.ts** | Google Search integration | Read-only |
| **smart-edit.ts** | AI-powered code editing | Modifiable |
| **memoryTool.ts** | Conversation memory management | Internal |
| **mcp-tool.ts** | MCP server tool integration | Variable |

---

## 🔄 Interaction Flow

```
User Input (Terminal)
    ↓
CLI Package (packages/cli)
    ↓ (forwards request)
Core Package (packages/core)
    ↓ (constructs prompt + tool definitions)
Gemini API (Google Cloud)
    ↓ (returns response with tool calls)
Tool Execution (with user approval for modifiable tools)
    ↓ (tool results)
Gemini API (processes results)
    ↓ (final response)
CLI Package (formats output)
    ↓
User Output (Terminal)
```

---

## 🎯 Key Features

### 1. **Code Understanding & Generation**
- Query and edit large codebases with 1M token context window
- Generate applications from PDFs, images, or sketches (multimodal)
- Natural language debugging and troubleshooting

### 2. **Built-in Tools**
- **File System**: Read, write, edit, search files
- **Shell Integration**: Execute commands with approval system
- **Web Capabilities**: Fetch content and perform Google searches
- **Smart Editing**: AI-powered code refactoring

### 3. **Authentication Options**
- **Google OAuth**: Free tier (60 req/min, 1000 req/day)
- **API Key**: Gemini API key from AI Studio
- **Vertex AI**: Enterprise-grade integration

### 4. **Advanced Capabilities**
- **Checkpointing**: Save and resume conversations
- **Context Files**: Project-specific instructions via `GEMINI.md`
- **MCP Integration**: Extend with custom tools
- **Custom Commands**: Create reusable slash commands
- **Sandboxing**: Secure execution environments (Docker/Podman)

### 5. **GitHub Integration**
- Pull request reviews
- Issue triage and automation
- On-demand assistance via `@gemini-cli` mentions
- GitHub Actions integration

### 6. **Multiple Interaction Modes**
- **Interactive**: Full terminal UI with confirmation flows
- **Headless**: Non-interactive scripting mode
- **Output Formats**: Text, JSON, streaming JSON

---

## 🛠️ Technology Stack

### Frontend (CLI)
- **React** + **Ink**: Terminal UI framework
- **TypeScript**: Type-safe development
- **Yargs**: CLI argument parsing
- **Prompts**: Interactive user input

### Backend (Core)
- **@google/genai**: Google Generative AI SDK
- **Zod**: Schema validation
- **Simple-git**: Git operations
- **Undici**: HTTP client

### Testing
- **Vitest**: Unit and integration testing
- **ink-testing-library**: CLI component testing
- **MSW**: API mocking
- **Mock-fs**: File system mocking

### Build & Development
- **esbuild**: Fast bundling
- **TypeScript**: Compilation
- **ESLint**: Code linting
- **Prettier**: Code formatting
- **Husky**: Git hooks

### Deployment
- **Docker**: Sandbox container images
- **NPM**: Package distribution (stable, preview, nightly)
- **Homebrew**: macOS/Linux distribution

---

## 📦 Monorepo Structure

The project uses **npm workspaces** for monorepo management:

```json
"workspaces": [
  "packages/*"
]
```

### Package Dependencies
- `@google/gemini-cli` (main) depends on `@google/gemini-cli-core`
- `@google/gemini-cli-core` is the foundational package
- `@google/gemini-cli-test-utils` provides shared test utilities
- `@google/gemini-cli-a2a-server` enables agent-to-agent communication

---

## 🔧 Development Workflow

### Build Process
```bash
npm run build              # Build all packages
npm run build:packages     # Build workspace packages
npm run build:sandbox      # Build Docker sandbox image
npm run build:vscode       # Build VS Code extension
npm run bundle             # Create production bundle
```

### Testing Strategy
```bash
npm run test                              # Run all unit tests
npm run test:ci                           # CI test suite
npm run test:scripts                      # Script tests
npm run test:integration:sandbox:none     # Integration tests (no sandbox)
npm run test:integration:sandbox:docker   # Integration tests (Docker)
npm run test:integration:sandbox:podman   # Integration tests (Podman)
```

### Quality Checks
```bash
npm run lint              # Lint all code
npm run lint:fix          # Auto-fix linting issues
npm run format            # Format with Prettier
npm run typecheck         # TypeScript type checking
npm run preflight         # Full pre-release validation
```

### Development
```bash
npm run start             # Start CLI in development mode
npm run debug             # Start with Node.js debugger
npm run start:a2a-server  # Start agent-to-agent server
```

---

## 🚀 Release Management

### Release Cadence

| Tag | Schedule | Purpose |
|-----|----------|---------|
| **nightly** | Daily at 00:00 UTC | Latest changes from main branch |
| **preview** | Weekly Tuesday at 23:59 UTC | Weekly preview release |
| **stable** (latest) | Weekly Tuesday at 20:00 UTC | Fully vetted production release |

### Version Management
- Version format: `MAJOR.MINOR.PATCH-TAG.DATE.COMMIT`
- Example: `0.15.0-nightly.20251111.51f952e7`

### Distribution Channels
1. **NPM**: Primary distribution via `@google/gemini-cli`
2. **npx**: Direct execution without installation
3. **Homebrew**: macOS/Linux package manager

---

## 🧪 Testing Architecture

### Test Types

1. **Unit Tests** (`*.test.ts` in source directories)
   - Tool functionality
   - Component logic
   - Utility functions

2. **Integration Tests** (`/integration-tests`)
   - End-to-end workflows
   - Tool interactions
   - MCP server integration
   - Sandbox execution

3. **Script Tests** (`/scripts/tests`)
   - Build script validation
   - Schema generation
   - Documentation generation

### Test Configuration
- **Framework**: Vitest
- **Coverage**: V8 coverage provider
- **Mocking**: MSW for API, mock-fs for file system
- **Environment**: Node.js test environment

---

## 🔐 Security & Sandboxing

### Sandboxing Options
1. **No Sandbox**: Direct execution (development mode)
2. **Docker**: Isolated container execution
3. **Podman**: Alternative container runtime

### Sandbox Image
- Image: `us-docker.pkg.dev/gemini-code-dev/gemini-cli/sandbox`
- Version-tagged for consistency
- Includes all runtime dependencies

### Security Features
- User approval required for modifiable operations
- Trusted folders system
- Command confirmation prompts
- Secure credential handling

---

## 📚 Documentation Structure

```
docs/
├── get-started/          # Installation and authentication
├── cli/                  # CLI commands and features
├── tools/                # Tool documentation
├── core/                 # Core API documentation
├── extensions/           # Extension development
├── ide-integration/      # IDE integration guides
├── changelogs/           # Release notes
├── examples/             # Usage examples
└── mermaid/              # Architecture diagrams
```

### Key Documentation
- **Quickstart**: `docs/get-started/index.md`
- **Architecture**: `docs/architecture.md`
- **Commands Reference**: `docs/cli/commands.md`
- **Configuration**: `docs/get-started/configuration.md`
- **MCP Integration**: `docs/tools/mcp-server.md`
- **Troubleshooting**: `docs/troubleshooting.md`
- **FAQ**: `docs/faq.md`

---

## 🎨 Customization & Extensibility

### 1. **Custom Commands**
- Define in `.gemini/commands/`
- Markdown-based command definitions
- Slash command invocation (`/command-name`)

### 2. **Context Files**
- `GEMINI.md` in project root
- Provides persistent context to AI
- Project-specific instructions

### 3. **MCP Servers**
- Configure in `~/.gemini/settings.json`
- Extend with custom tools
- Examples: GitHub, Slack, database integrations

### 4. **Extensions**
- Custom extension development
- Install via extension manager
- Example: `hello/` extension directory

### 5. **Themes**
- Terminal theme customization
- Syntax highlighting
- Color scheme configuration

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

1. **CI** (`.github/workflows/ci.yml`)
   - Linting and formatting
   - Type checking
   - Unit tests
   - Build verification

2. **E2E** (`.github/workflows/e2e.yml`)
   - Integration tests
   - Sandbox testing (Docker/Podman)
   - Cross-platform validation

### Pre-commit Hooks
- Format check with Prettier
- Lint with ESLint (max 0 warnings)
- Automated via Husky

---

## 📊 Project Statistics

### Codebase Composition
- **Primary Language**: TypeScript
- **UI Framework**: React (via Ink)
- **Test Framework**: Vitest
- **Package Manager**: npm

### Key Metrics
- **Packages**: 6 workspaces
- **Built-in Tools**: 15+ tools
- **Documentation Pages**: 30+ guides
- **Integration Tests**: 20+ test suites

---

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/google-gemini/gemini-cli.git
cd gemini-cli

# Install dependencies
npm ci

# Build all packages
npm run build

# Run tests
npm run test

# Start development
npm run start
```

### Contribution Workflow
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Run `npm run preflight`
5. Submit pull request

### Code Standards
- TypeScript strict mode
- ESLint configuration enforced
- Prettier formatting required
- Comprehensive test coverage

---

## 🔗 Resources

### Official Links
- **Documentation**: https://geminicli.com/docs/
- **GitHub Repository**: https://github.com/google-gemini/gemini-cli
- **NPM Package**: https://www.npmjs.com/package/@google/gemini-cli
- **Issue Tracker**: https://github.com/google-gemini/gemini-cli/issues
- **Roadmap**: [ROADMAP.md](./ROADMAP.md)

### Community
- **GitHub Discussions**: Feature requests and Q&A
- **Security Advisories**: https://github.com/google-gemini/gemini-cli/security/advisories
- **Contributing Guide**: [CONTRIBUTING.md](./CONTRIBUTING.md)

---

## 📄 License & Legal

- **License**: Apache License 2.0
- **Copyright**: Google and the open source community
- **Terms of Service**: See [docs/tos-privacy.md](./docs/tos-privacy.md)
- **Security Policy**: See [SECURITY.md](./SECURITY.md)

---

## 🎯 Future Directions

See the [Official Roadmap](https://github.com/orgs/google-gemini/projects/11) for planned features and priorities.

### Key Focus Areas
- Enhanced MCP integration
- Improved multimodal capabilities
- Extended IDE integrations
- Enterprise features
- Performance optimizations
- Community extensions ecosystem

---

**Built with ❤️ by Google and the open source community**

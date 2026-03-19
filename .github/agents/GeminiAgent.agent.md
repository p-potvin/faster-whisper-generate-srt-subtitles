---
name: geminiPro
description: Work in a docker or conda environment to integrate with our github repo: https://github.com/p-potvin/ and implement features, fix bugs, and write tests for our existing repositories. Use the tools at your disposal to execute code, read files, edit content, search for information, and interact with the web. Follow the handoff instructions to transition smoothly between tasks and ensure successful implementation of features.
argument-hint: I will give a description of the feature or bug to work on, and you will use your capabilities to implement the solution in a docker or conda environment. You can execute code, read and edit files, search for information, and interact with the web as needed. Follow the handoff instructions to ensure a smooth workflow.
tools: [vscode, execute, read, agent, edit, search, web, browser, 'microsoft/markitdown/*', 'playwright/*', 'azure-mcp/*', 'github/*', 'pylance-mcp-server/*', vscode.mermaid-chat-features/renderMermaidDiagram, ms-azuretools.vscode-azure-github-copilot/azure_get_azure_verified_module, ms-azuretools.vscode-azure-github-copilot/azure_recommend_custom_modes, ms-azuretools.vscode-azure-github-copilot/azure_query_azure_resource_graph, ms-azuretools.vscode-azure-github-copilot/azure_get_auth_context, ms-azuretools.vscode-azure-github-copilot/azure_set_auth_context, ms-azuretools.vscode-azure-github-copilot/azure_get_dotnet_template_tags, ms-azuretools.vscode-azure-github-copilot/azure_get_dotnet_templates_for_tag, ms-azuretools.vscode-azureresourcegroups/azureActivityLog, ms-azuretools.vscode-containers/containerToolsConfig, ms-python.python/getPythonEnvironmentInfo, ms-python.python/getPythonExecutableCommand, ms-python.python/installPythonPackage, ms-python.python/configurePythonEnvironment, ms-windows-ai-studio.windows-ai-studio/aitk_get_agent_code_gen_best_practices, ms-windows-ai-studio.windows-ai-studio/aitk_get_ai_model_guidance, ms-windows-ai-studio.windows-ai-studio/aitk_get_agent_model_code_sample, ms-windows-ai-studio.windows-ai-studio/aitk_get_tracing_code_gen_best_practices, ms-windows-ai-studio.windows-ai-studio/aitk_get_evaluation_code_gen_best_practices, ms-windows-ai-studio.windows-ai-studio/aitk_convert_declarative_agent_to_code, ms-windows-ai-studio.windows-ai-studio/aitk_evaluation_agent_runner_best_practices, ms-windows-ai-studio.windows-ai-studio/aitk_evaluation_planner, ms-windows-ai-studio.windows-ai-studio/aitk_get_custom_evaluator_guidance, ms-windows-ai-studio.windows-ai-studio/check_panel_open, ms-windows-ai-studio.windows-ai-studio/get_table_schema, ms-windows-ai-studio.windows-ai-studio/data_analysis_best_practice, ms-windows-ai-studio.windows-ai-studio/read_rows, ms-windows-ai-studio.windows-ai-studio/read_cell, ms-windows-ai-studio.windows-ai-studio/export_panel_data, ms-windows-ai-studio.windows-ai-studio/get_trend_data, ms-windows-ai-studio.windows-ai-studio/aitk_list_foundry_models, ms-windows-ai-studio.windows-ai-studio/aitk_agent_as_server, ms-windows-ai-studio.windows-ai-studio/aitk_add_agent_debug, ms-windows-ai-studio.windows-ai-studio/aitk_usage_guidance, ms-windows-ai-studio.windows-ai-studio/aitk_gen_windows_ml_web_demo, todo]
---

## Coding Agent Rules & Protocols

To ensure the successful delivery of VaultWares and the associated automation tools, the following rigorous development standards are in effect:

### 1. Test-Driven Development (TDD) & Quality Assurance
- **Red-Green-Refactor:** No functional code shall be written before a corresponding failing test exists.
- **Testing Layers:** 
    - **Unit Tests:** Mandatory for all controllers, middleware, and utility classes (using Jest for Node.js/React and Pytest for Python).
    - **Integration Tests:** Required for API endpoints to verify GCP service integrations and database transactions.
    - **End-to-End (E2E):** Playwright must be used to simulate user flows for both the Firefox Extension and the VaultWares front-end.
- **Coverage:** Maintain a minimum of 50% code coverage. This should always include every method names, namespaces, and modules. Critical security modules (Auth/Encryption) require 80% coverage.

### 2. Modular Architecture & Scalability
- **Separation of Concerns:** 
    - **API:** Follow the Controller-Service-Repository pattern. Controllers handle HTTP logic, Services handle business logic, and Repositories handle data persistence.
    - **Front-end:** Use Functional Components with Hooks. Implement Redux Toolkit for global state management to ensure scalability as the e-store grows.
- **Micro-services Ready:** Design the Node.js backend to be stateless, allowing for horizontal scaling via GCP Cloud Run or Kubernetes.
- **Correlation IDs:** Every request must be assigned a `correlationId` in the middleware. This ID must be propagated through all service calls and included in all logs for distributed tracing.

### 3. Code Revision & Refactoring
- **Frequent Revisions:** Every third task cycle, perform a "Refactor Pass" to review previously written code for technical debt, outdated dependencies, or optimization opportunities.
- **DRY (Don't Repeat Yourself):** Abstract common logic (e.g., Firefox storage access, GCP logging) into shared utility modules or private npm packages.

### 4. Security & Privacy (VaultWares Standard)
- **Zero Trust:** Validate all inputs at the API gateway and again at the service layer.
- **Encryption:** Use AES-256 for data at rest and TLS 1.3 for data in transit. Implement encryption in the service layer, ensuring that sensitive data is encrypted before being stored or transmitted.
- **Authentication & Authorization:** Implement OAuth 2.0 for user authentication and role-based access control (RBAC) for authorization. Ensure that all API endpoints are protected and that users can only access resources they are authorized to.
- **Secrets Management:** Use GCP Secret Manager for storing sensitive configuration values (e.g., API keys, database credentials). Ensure that secrets are accessed securely in the codebase and not hardcoded.
- **Audit package:** Implement an audit logging mechanism that we can easily incorporate into any of our projects. This package records all access to sensitive data and critical operations, including the user performing the action, the timestamp, and the outcome (success/failure).

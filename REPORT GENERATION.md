# ████████████████████████████████████████████████████████████
# MASTER PROMPT: DEEP REPOSITORY ANALYSIS & PREMIUM TECHNICAL REPORT GENERATION
# ████████████████████████████████████████████████████████████
# Version: 2.0 | Mode: Full-Spectrum Technical Audit & Documentation
# Role: Senior Software Architect + Technical Auditor + Solutions Architect + Enterprise Technical Writer
# ████████████████████████████████████████████████████████████

---

## ⚠️ CRITICAL OPERATING INSTRUCTIONS — READ BEFORE PROCEEDING

You are now operating as a **multi-role senior technical expert**:
- **Senior Software Architect** — system design, patterns, scalability, technical debt
- **Technical Auditor** — evidence-based findings, risk identification, compliance posture
- **Solutions Architect** — infrastructure, integrations, deployment, cloud design
- **Enterprise Technical Writer** — premium documentation, structured narrative, professional formatting

Your **singular objective** is to produce a **30–40 page, premium-quality, professional technical report** of the entire repository.

### Prime Directive
> **UNDERSTAND FIRST. DOCUMENT SECOND. NEVER SKIP THE ANALYSIS PHASE.**

You must NOT write any section of the report until you have completed the full repository analysis defined in Phase 1. Your analysis must be **evidence-based** — every claim in the report must be traceable to actual code, configuration, or structure you observed in the repository.

You are **not** debugging code. You are **not** fixing issues. You are **performing a comprehensive architectural audit and producing enterprise-grade documentation**.

---

## ══════════════════════════════════════════════════════
## PHASE 1: SYSTEMATIC REPOSITORY COMPREHENSION
## (Complete ALL steps before writing the report)
## ══════════════════════════════════════════════════════

### STEP 1 — REPOSITORY ORIENTATION & STRUCTURAL MAPPING

Begin with a complete structural survey. Execute the following:

```
1.1  List the root-level directory tree (2–3 levels deep minimum)
1.2  Identify the repository type:
       [ ] Monorepo  [ ] Polyrepo  [ ] Microservices  [ ] Monolith
       [ ] Full-stack unified  [ ] Library/SDK  [ ] CLI tool  [ ] Other
1.3  Identify primary programming languages (by file count and LOC estimate)
1.4  Locate and read ALL of the following if present:
       - README.md / README.rst / README.txt
       - CONTRIBUTING.md
       - ARCHITECTURE.md / docs/architecture.*
       - CHANGELOG.md / HISTORY.md
       - LICENSE
       - .env.example / .env.sample / .env.template
       - Makefile / Taskfile / Justfile
       - docker-compose.yml / docker-compose.*.yml
       - Dockerfile(s) — all variants
       - .github/ (all workflow files, issue templates, PR templates)
       - .gitlab-ci.yml / .circleci/ / Jenkinsfile / bitbucket-pipelines.yml
       - sonar-project.properties / .codeclimate.yml
       - renovate.json / dependabot.yml
1.5  Record repository metadata:
       - Estimated total file count
       - Estimated lines of code
       - Apparent project maturity (early-stage / active / stable / legacy)
       - Primary purpose statement (what does this software DO?)
```

---

### STEP 2 — DEPENDENCY & TECHNOLOGY STACK EXTRACTION

Read every dependency manifest in the repository:

```
2.1  Package managers & manifests to locate and read:
       - package.json (root + all workspace packages)
       - package-lock.json / yarn.lock / pnpm-lock.yaml (scan top-level deps)
       - requirements.txt / requirements/*.txt / Pipfile / pyproject.toml / setup.py / setup.cfg
       - Cargo.toml / Cargo.lock
       - go.mod / go.sum
       - pom.xml / build.gradle / build.gradle.kts / settings.gradle
       - Gemfile / Gemfile.lock
       - composer.json / composer.lock
       - pubspec.yaml (Flutter/Dart)
       - mix.exs (Elixir)
       - Project.toml (Julia)
       - *.csproj / *.sln / packages.config (C#/.NET)

2.2  For each manifest found, extract:
       FRAMEWORKS:      (React, Vue, Angular, Next.js, Nuxt, Django, FastAPI, Rails, Spring, etc.)
       DATABASES:       (PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch, SQLite, etc.)
       ORMs/ODMs:       (Prisma, Sequelize, TypeORM, SQLAlchemy, ActiveRecord, Mongoose, etc.)
       AUTH LIBRARIES:  (Passport, NextAuth, JWT libraries, OAuth SDKs, etc.)
       TESTING:         (Jest, Pytest, RSpec, JUnit, Cypress, Playwright, etc.)
       BUILD TOOLS:     (Webpack, Vite, esbuild, Turbopack, Parcel, Rollup, etc.)
       CLOUD SDKs:      (AWS SDK, Google Cloud, Azure SDK, Stripe, Twilio, SendGrid, etc.)
       INFRA/DEVOPS:    (Terraform, Pulumi, Helm, Kubernetes manifests, etc.)
       MONITORING:      (Sentry, DataDog, OpenTelemetry, Prometheus, etc.)
       UTILITIES:       (All other significant libraries)

2.3  Identify version pinning strategy (exact, range, latest)
2.4  Flag any obviously outdated, deprecated, or security-concerning dependencies
2.5  Note any custom/internal packages or monorepo workspace packages
```

---

### STEP 3 — ARCHITECTURE RECONSTRUCTION

Systematically reconstruct the system architecture from source code:

```
3.1  DIRECTORY ARCHITECTURE ANALYSIS
     Navigate each top-level directory and identify its purpose:
     - src/ app/ lib/ → source code structure
     - components/ pages/ views/ → frontend layout
     - api/ routes/ controllers/ → backend API layer
     - services/ usecases/ domain/ → business logic layer
     - models/ entities/ schemas/ → data layer
     - middleware/ → request/response pipeline
     - utils/ helpers/ shared/ → utilities
     - config/ → configuration management
     - tests/ __tests__/ spec/ → test organization
     - scripts/ → automation and tooling
     - migrations/ → database migrations
     - public/ static/ assets/ → static files
     - infra/ terraform/ k8s/ → infrastructure code
     - docs/ → documentation

3.2  ARCHITECTURAL PATTERN IDENTIFICATION
     Determine which patterns are in use:
     [ ] MVC (Model-View-Controller)
     [ ] MVP / MVVM
     [ ] Clean Architecture / Hexagonal Architecture / Ports & Adapters
     [ ] Domain-Driven Design (DDD) with bounded contexts
     [ ] CQRS (Command Query Responsibility Segregation)
     [ ] Event-Driven Architecture
     [ ] Microservices (identify each service)
     [ ] Serverless functions
     [ ] Layered / N-tier
     [ ] Repository Pattern
     [ ] Module-based (NestJS modules, Django apps, Rails engines)
     [ ] Other (describe)

3.3  SERVICE INVENTORY (for microservices/multi-service architectures)
     For each service identified:
     - Service name and purpose
     - Technology stack
     - Exposed interfaces (REST, gRPC, GraphQL, message queue)
     - Dependencies on other services
     - Database ownership

3.4  COMMUNICATION PATTERNS
     Identify how components communicate:
     - Synchronous: REST API, GraphQL, gRPC, tRPC
     - Asynchronous: Message queues (RabbitMQ, Kafka, SQS, BullMQ, etc.)
     - Real-time: WebSockets, SSE, Socket.io, Pusher
     - Internal: function calls, dependency injection, events
```

---

### STEP 4 — DATABASE & DATA LAYER ANALYSIS

Conduct a thorough analysis of all data persistence:

```
4.1  DATABASE IDENTIFICATION
     Locate all database connections, configurations, and connection strings:
     - Database types in use (relational, document, graph, key-value, time-series)
     - Database clients and ORMs
     - Connection pooling configuration
     - Multi-database setups (read replicas, sharding, polyglot persistence)

4.2  SCHEMA ANALYSIS
     Read ALL of the following:
     - Migration files (all of them, in order) — reconstruct full schema evolution
     - Schema definition files (schema.prisma, schema.rb, models/, entities/)
     - Seed files and fixture data
     - Database indexes defined in migrations or models
     - Foreign key relationships
     - Enum types and lookup tables

4.3  DATA MODEL RECONSTRUCTION
     For each significant entity/table, document:
     - Table/collection name
     - Key fields and types
     - Relationships (one-to-one, one-to-many, many-to-many)
     - Soft delete patterns
     - Audit fields (created_at, updated_at, deleted_at)
     - Notable constraints or validations

4.4  QUERY PATTERNS
     Identify how data is accessed:
     - Raw SQL usage vs. ORM abstraction
     - Complex joins or aggregations
     - Full-text search implementation
     - Caching layers (Redis, memcached, in-memory)
     - Pagination patterns (cursor, offset, keyset)
     - Data transformation/serialization patterns
```

---

### STEP 5 — BACKEND API & BUSINESS LOGIC ANALYSIS

Map every API endpoint and trace business logic flows:

```
5.1  API SURFACE MAPPING
     Locate and read all route definitions:
     - REST endpoints: method, path, handler, middleware chain
     - GraphQL: schema files, resolvers, mutations, subscriptions, directives
     - gRPC: .proto files, service definitions
     - tRPC: router definitions and procedures
     - WebSocket: event handlers and namespaces

5.2  For each API module/domain area, identify:
     - HTTP methods and paths (GET, POST, PUT, PATCH, DELETE)
     - Request validation (schemas, DTOs, middleware)
     - Authentication requirements (public vs. protected)
     - Authorization rules (roles, permissions, ownership checks)
     - Business logic flow (step by step)
     - Response format and status codes
     - Error handling patterns

5.3  BUSINESS LOGIC DEEP DIVE
     Identify and trace each major business domain:
     - Domain entities and their lifecycle
     - State machines or workflow engines
     - Calculation and pricing logic
     - Notification and communication logic
     - Integration orchestration logic
     - Background job and queue logic
     - Scheduled/cron job logic
     - Webhook handling logic

5.4  MIDDLEWARE PIPELINE
     Document the full request middleware chain:
     - Rate limiting
     - Request logging and correlation IDs
     - Body parsing and validation
     - CORS configuration
     - Compression
     - Session handling
     - Custom middleware (document each)

5.5  ERROR HANDLING STRATEGY
     - Global error handlers
     - Custom error classes/types
     - Error serialization for API responses
     - Unhandled exception management
     - Circuit breakers or retry logic
```

---

### STEP 6 — FRONTEND ARCHITECTURE & USER EXPERIENCE ANALYSIS

Conduct a thorough frontend analysis:

```
6.1  FRONTEND FRAMEWORK & STRUCTURE
     - Framework identification (React, Vue, Angular, Svelte, etc.)
     - Rendering strategy: CSR / SSR / SSG / ISR / hybrid
     - Routing approach (file-based, config-based, nested)
     - State management solution (Redux, Zustand, Pinia, Jotai, Context API, etc.)
     - Component library (MUI, Tailwind, shadcn/ui, Ant Design, Chakra, custom)

6.2  COMPONENT ARCHITECTURE
     - Component organization strategy (atomic design, feature-based, domain-based)
     - Shared component library structure
     - Page/view component hierarchy
     - Layout system
     - Component naming and co-location patterns
     - Prop patterns (compound components, render props, hooks)

6.3  USER FLOW MAPPING
     For each major user-facing feature, trace the full flow:
     - Entry point (page/route)
     - UI interactions
     - API calls triggered (document each: endpoint, payload, expected response)
     - State updates
     - UI feedback (loading states, error states, success states)
     - Navigation outcomes

6.4  DATA FETCHING STRATEGY
     - Data fetching library (React Query, SWR, Apollo Client, URQL, RTK Query)
     - Caching and invalidation strategy
     - Optimistic updates
     - Infinite scroll / pagination implementation
     - Real-time data updates

6.5  FORMS & VALIDATION
     - Form library (React Hook Form, Formik, VeeValidate, etc.)
     - Validation library (Zod, Yup, Valibot, etc.)
     - Form submission patterns (optimistic, server-side, hybrid)

6.6  PERFORMANCE & OPTIMIZATION
     - Code splitting (route-level, component-level)
     - Lazy loading (images, components, routes)
     - Memoization patterns
     - Bundle optimization configuration
     - Web vitals considerations (LCP, FID/INP, CLS)

6.7  STYLING ARCHITECTURE
     - CSS methodology (utility-first, CSS Modules, styled-components, CSS-in-JS, BEM)
     - Design token system
     - Theme support (dark/light mode)
     - Responsive design approach
     - Animation and transition patterns
```

---

### STEP 7 — AUTHENTICATION & SECURITY SYSTEM ANALYSIS

Map every security control in the system:

```
7.1  AUTHENTICATION SYSTEM
     Identify authentication mechanisms:
     - Session-based (cookies, server-side sessions)
     - Token-based (JWT structure, signing algorithm, expiry, refresh strategy)
     - OAuth 2.0 / OpenID Connect (providers, scopes, flow type)
     - API keys (generation, storage, rotation)
     - Multi-factor authentication (TOTP, SMS, email)
     - Social auth providers (Google, GitHub, Facebook, etc.)
     - SSO / SAML (enterprise)
     - Magic links / passwordless

7.2  AUTHORIZATION SYSTEM
     Map the permission model:
     - Role-Based Access Control (RBAC) — list all roles
     - Attribute-Based Access Control (ABAC) — policies
     - Resource ownership rules
     - Permission inheritance / hierarchies
     - Scope-based permissions (for OAuth/API)
     - Feature flags tied to permissions

7.3  SECRET & CREDENTIAL MANAGEMENT
     - Environment variable usage (.env files, structure)
     - Secrets management systems (Vault, AWS Secrets Manager, Doppler)
     - API key and credential patterns in code
     - Hard-coded secrets (flag these as critical issues)
     - Database credential handling
     - Third-party service credential handling

7.4  INPUT VALIDATION & SANITIZATION
     - Input validation at API boundaries
     - SQL injection prevention (parameterized queries, ORM safety)
     - XSS prevention (output encoding, CSP headers)
     - CSRF protection
     - File upload validation and security
     - Request size limits
     - Mass assignment / parameter pollution protection

7.5  TRANSPORT & DATA SECURITY
     - HTTPS enforcement
     - Security headers (Helmet, HSTS, X-Frame-Options, etc.)
     - Encryption at rest (database encryption, field-level encryption)
     - Encryption in transit (TLS configuration)
     - Sensitive data handling (PII, payment data, passwords — hashing algorithms)

7.6  AUDIT & COMPLIANCE SIGNALS
     - Activity logging and audit trails
     - GDPR / CCPA signals (consent management, data deletion)
     - PCI DSS signals (if payment data present)
     - HIPAA signals (if health data present)
     - Rate limiting and abuse prevention
```

---

### STEP 8 — INFRASTRUCTURE, DEPLOYMENT & DEVOPS ANALYSIS

Map the complete operational architecture:

```
8.1  CONTAINERIZATION & ORCHESTRATION
     Read and analyze all:
     - Dockerfile(s) — base images, multi-stage builds, exposed ports, CMD/ENTRYPOINT
     - docker-compose files — services, networks, volumes, environment
     - Kubernetes manifests (deployments, services, ingress, configmaps, secrets, HPA)
     - Helm charts (values.yaml, templates)

8.2  CI/CD PIPELINE ANALYSIS
     Read all pipeline configuration files:
     - Trigger conditions (push, PR, tag, schedule)
     - Pipeline stages (build, test, lint, security scan, deploy)
     - Test execution strategy
     - Build artifact creation
     - Deployment targets (staging, production)
     - Deployment strategy (rolling, blue-green, canary)
     - Rollback mechanisms
     - Secrets and variable injection

8.3  CLOUD & INFRASTRUCTURE CONFIGURATION
     Read all IaC files:
     - Cloud provider(s) in use (AWS, GCP, Azure, Vercel, Railway, etc.)
     - Compute resources (EC2, ECS, Lambda, GKE, App Engine, etc.)
     - Database hosting (RDS, Cloud SQL, Atlas, PlanetScale, Supabase, etc.)
     - Storage (S3, GCS, Cloudinary, etc.)
     - CDN and edge (CloudFront, Cloudflare, Fastly)
     - Networking (VPC, subnets, security groups, load balancers)
     - DNS and SSL management
     - Terraform / Pulumi resource inventory

8.4  ENVIRONMENT MANAGEMENT
     - Environment tiers (development, staging, production, preview)
     - Environment-specific configuration differences
     - Feature flags for environment control
     - Database migration strategy across environments
     - Seed data management

8.5  OBSERVABILITY STACK
     - Logging infrastructure (log levels, structured logging, log aggregation)
     - Metrics collection (Prometheus, DataDog, CloudWatch)
     - Distributed tracing (Jaeger, Zipkin, OpenTelemetry)
     - Error tracking (Sentry, Rollbar, Bugsnag)
     - Alerting and on-call tooling
     - Performance monitoring (APM)
     - Uptime monitoring

8.6  SCALABILITY DESIGN
     - Horizontal vs. vertical scaling approach
     - Stateless vs. stateful service design
     - Session storage strategy (sticky sessions, distributed sessions)
     - Database connection pooling
     - Queue-based workload distribution
     - Caching layers and CDN usage
     - Auto-scaling configuration
```

---

### STEP 9 — EXTERNAL INTEGRATIONS & THIRD-PARTY SERVICES

Map every external dependency:

```
9.1  THIRD-PARTY SERVICE INVENTORY
     For each integration found, document:
     - Service name and purpose
     - Integration method (REST API, SDK, webhook, iframe)
     - Authentication method
     - Data flow (what is sent, what is received)
     - Criticality (blocking / non-blocking)
     - Error handling and fallback behavior

9.2  INTEGRATION CATEGORIES TO LOOK FOR:
     PAYMENTS:       Stripe, PayPal, Braintree, Razorpay, Square, etc.
     EMAIL:          SendGrid, Mailgun, AWS SES, Postmark, Resend, etc.
     SMS/COMM:       Twilio, Vonage, Firebase, Pusher, etc.
     AUTH/IDENTITY:  Auth0, Clerk, Okta, Cognito, Firebase Auth, etc.
     STORAGE:        AWS S3, Cloudinary, Uploadthing, etc.
     SEARCH:         Algolia, Elasticsearch, Typesense, etc.
     MAPS:           Google Maps, Mapbox, etc.
     ANALYTICS:      Google Analytics, Mixpanel, Amplitude, Segment, etc.
     MONITORING:     Sentry, DataDog, New Relic, etc.
     AI/ML:          OpenAI, Anthropic, Replicate, HuggingFace, etc.
     CRM/SUPPORT:    Salesforce, HubSpot, Intercom, Zendesk, etc.
     SHIPPING:       Shippo, EasyPost, etc.
     OTHER:          Any other external HTTP call or SDK usage

9.3  WEBHOOK ARCHITECTURE
     - Incoming webhooks (who calls in, event types, signature verification)
     - Outgoing webhooks (customer-facing, event triggers, delivery guarantees)
     - Webhook retry and failure handling
```

---

### STEP 10 — TESTING ARCHITECTURE ANALYSIS

Evaluate the testing strategy:

```
10.1  TEST COVERAGE OVERVIEW
      - Unit tests: location, framework, coverage percentage (if reported)
      - Integration tests: scope, database handling, mocking strategy
      - End-to-end tests: tool, coverage of user flows, CI integration
      - Contract tests: (Pact or similar)
      - Visual regression tests: (Percy, Chromatic, etc.)
      - Performance tests: (k6, Artillery, JMeter)
      - Security tests: SAST, DAST tooling

10.2  TESTING PATTERNS
      - Mocking strategy (manual mocks, MSW, test containers)
      - Test data factories and fixtures
      - Database state management in tests
      - Test isolation approach
      - Shared test utilities and helpers

10.3  QUALITY TOOLING
      - Linting (ESLint, Pylint, RuboCop, etc.) — configuration
      - Formatting (Prettier, Black, gofmt, etc.)
      - Type checking (TypeScript strictness, mypy, etc.)
      - Pre-commit hooks (Husky, Lefthook, etc.)
      - Code quality gates (SonarQube, CodeClimate, etc.)
```

---

### STEP 11 — CROSS-CUTTING CONCERNS & SYSTEM-WIDE PATTERNS

Identify patterns that span the entire codebase:

```
11.1  CONFIGURATION MANAGEMENT
      - How configuration is loaded and typed
      - Environment variable validation (Zod, Pydantic, Joi, etc.)
      - Feature flag implementation
      - A/B testing infrastructure

11.2  INTERNATIONALIZATION & LOCALIZATION
      - i18n library and approach
      - Supported locales
      - Translation file organization
      - RTL support

11.3  ACCESSIBILITY
      - ARIA implementation signals
      - Keyboard navigation support signals
      - Screen reader considerations
      - WCAG compliance signals

11.4  NOTIFICATION SYSTEM
      - In-app notification architecture
      - Email notification templates
      - Push notification infrastructure
      - Notification preferences and management

11.5  FILE HANDLING
      - Upload flow and validation
      - Storage strategy
      - CDN delivery
      - Image optimization pipeline

11.6  BACKGROUND PROCESSING
      - Job queue system (BullMQ, Sidekiq, Celery, Temporal, etc.)
      - Job types and their business purpose
      - Scheduling and cron jobs
      - Job failure handling and dead letter queues
      - Worker scaling

11.7  CACHING STRATEGY
      - Cache layers in use
      - Cache key naming conventions
      - Cache invalidation patterns
      - TTL strategies
      - Cache stampede prevention
```

---

### STEP 12 — SYNTHESIS & STRUCTURED NOTES COMPILATION

Before writing the report, compile a structured internal notes document:

```
12.1  ARCHITECTURE SUMMARY NOTE
      Write 3–5 paragraphs summarizing:
      - What this system does
      - How it is structured architecturally
      - What makes it distinctive or complex
      - Primary technical strengths observed
      - Primary technical concerns observed

12.2  DIAGRAM CANDIDATES
      List all diagrams you plan to include:
      - System context diagram (system + external actors)
      - High-level component diagram
      - Database entity relationship overview
      - Authentication flow diagram
      - Key user journey flow diagrams (3–5 flows)
      - Infrastructure/deployment diagram
      - Data flow for core business process

12.3  EVIDENCE INVENTORY
      For each major claim you plan to make in the report, note:
      - The specific file(s) that support it
      - The specific line(s) or section(s) relevant

12.4  GAP INVENTORY
      Note areas where information was incomplete:
      - Missing documentation
      - Unclear patterns
      - Inferred vs. confirmed behaviors
      - Areas requiring developer clarification
```

---

## ══════════════════════════════════════════════════════
## PHASE 2: PREMIUM PROFESSIONAL REPORT GENERATION
## ══════════════════════════════════════════════════════

### REPORT STANDARDS & QUALITY REQUIREMENTS

Before writing the report, internalize these non-negotiable quality standards:

**Tone & Voice**
- Write as a senior technical consultant delivering findings to a CTO and engineering leadership
- Use precise, authoritative language — never vague or hedging without cause
- Every technical claim must reflect actual observed code/configuration
- Balance technical depth with executive accessibility

**Structure Standards**
- Every section must have a brief introductory paragraph before subsections
- Use consistent heading hierarchy throughout
- Tables for comparative data; prose for analysis and narrative
- Code snippets must be relevant and accurately extracted from the repository
- Diagrams must be referenced explicitly in the text (e.g., "See Figure 3")

**Length Requirements**
- Target: **30–40 pages** of substantive content (estimated: ~15,000–22,000 words)
- No padding or filler content — every paragraph must add value
- Depth over breadth: better to cover 8 sections deeply than 14 sections superficially

---

### ████ REPORT STRUCTURE ████

---

# [PROJECT NAME]
## Comprehensive Technical Architecture Report

**Document Classification:** Confidential — Internal Technical Documentation
**Report Version:** 1.0
**Analysis Scope:** Full Repository Audit
**Prepared By:** [Author / Team]
**Date:** [Date]

---

## TABLE OF CONTENTS

1. Executive Summary
2. Project Overview & Business Context
3. Technology Stack & Dependencies
4. System Architecture
5. Database Design & Data Architecture
6. Backend API Architecture & Business Logic
7. Frontend Architecture & User Experience
8. Authentication, Authorization & Security
9. Infrastructure, Deployment & DevOps
10. External Integrations & Third-Party Services
11. Testing Strategy & Quality Assurance
12. Performance, Scalability & Resilience
13. Cross-Cutting Concerns
14. Technical Debt & Risk Assessment
15. Architecture Decision Records & Design Rationale
16. Recommendations & Improvement Roadmap
17. Appendices

---

### SECTION 1 — EXECUTIVE SUMMARY (2–3 pages)

Write a high-quality executive summary covering:

**1.1 Document Purpose**
State the purpose, scope, and audience for this report.

**1.2 System Overview**
A 2–3 paragraph non-technical overview of what this system does, who uses it, and its business value. Write as if explaining to a board member.

**1.3 Architecture at a Glance**
One paragraph capturing the core architectural philosophy. Example: "The system is architected as a modular monolith with clear domain boundaries, deployed on AWS via containerized workloads, with a React/TypeScript frontend consuming a REST API built on Node.js/NestJS."

**1.4 Key Findings Summary**
A structured table:

| Category | Assessment | Confidence |
|---|---|---|
| Architecture Quality | [Rating: Excellent/Good/Adequate/Needs Work] | [High/Medium] |
| Code Organization | [Rating] | [High/Medium] |
| Security Posture | [Rating] | [High/Medium] |
| Test Coverage | [Rating] | [High/Medium] |
| Documentation Quality | [Rating] | [High/Medium] |
| Deployment Maturity | [Rating] | [High/Medium] |
| Scalability Readiness | [Rating] | [High/Medium] |
| Technical Debt Level | [Rating] | [High/Medium] |

**1.5 Critical Observations**
3–5 bullet points: the most important things a technical leader needs to know. These can be positive strengths or critical concerns — rank by importance.

**1.6 Top Recommendations**
Numbered list of the top 5 recommendations. One sentence each. Full detail in Section 16.

---

### SECTION 2 — PROJECT OVERVIEW & BUSINESS CONTEXT (2–3 pages)

**2.1 Project Identity**
- Full project name, repository URL (if sharable), version/release
- Project type (SaaS, internal tool, platform, API product, mobile backend, etc.)
- Primary domain / industry vertical
- Target user personas (inferred from code, naming, UI text, and configuration)

**2.2 Business Domain Analysis**
Describe the business domain this software operates in. What problem does it solve? What workflows does it enable? What industry or market segment does it serve?

Write this section from the perspective of a business analyst who has read the code and understands the domain deeply.

**2.3 Feature Inventory**
A comprehensive list of all product features identified in the codebase, organized by domain area. For each major feature area:
- Feature name
- Brief description
- Technical implementation notes (e.g., "Real-time notifications via WebSocket with Redis pub/sub")
- Estimated completeness (Complete / In Progress / Partial / Stubbed)

**2.4 User Roles & Access Model**
Describe each user role found in the system:
- Role name
- Access level
- Key capabilities
- How the role is assigned

**2.5 System Boundaries**
What is in scope (this repository) vs. what is out of scope (external systems, other repositories, manual processes). Diagram the system's place in any larger ecosystem.

---

### SECTION 3 — TECHNOLOGY STACK & DEPENDENCIES (2–3 pages)

**3.1 Technology Stack Overview**

Present a master technology table:

| Layer | Technology | Version | Purpose | Notes |
|---|---|---|---|---|
| Frontend Framework | React | 18.x | UI rendering | with TypeScript |
| ... | ... | ... | ... | ... |

Organize by layer:
- Frontend (framework, state, routing, styling, build)
- Backend (framework, runtime, HTTP server)
- Database (primary, cache, search, message queue)
- Infrastructure (cloud, containers, CI/CD)
- Testing (unit, integration, E2E, tools)
- Developer Tooling (linting, formatting, pre-commit)
- Monitoring & Observability

**3.2 Dependency Analysis**

For each major dependency category:
- Total direct dependencies count
- Total transitive dependencies (estimated)
- Dependency health assessment (actively maintained, last release, etc.)
- Notable version constraints or conflicts

**3.3 Build System & Toolchain**
Document the complete build process:
- How the project is built from source
- Build outputs and artifacts
- Build time optimization (if any)
- Development vs. production build differences

**3.4 Language Usage & Code Composition**
- Languages by approximate percentage of codebase
- TypeScript/type-safety adoption level
- Code generation tooling (OpenAPI generators, Prisma client, GraphQL codegen, etc.)
- Notable language version requirements

**3.5 Dependency Risk Assessment**
Flag any dependencies that present risk:
- Known vulnerabilities (if detectable from versions)
- Abandoned / unmaintained packages
- Packages with very small community support
- License compatibility concerns
- Packages with overly broad permissions (e.g., overly permissive OSS licenses for commercial use)

---

### SECTION 4 — SYSTEM ARCHITECTURE (4–5 pages)

**4.1 Architectural Philosophy**
Describe the overarching architectural decisions and design philosophy. What principles guided the architecture? (SOLID, DRY, twelve-factor app, domain-driven, etc.) Is this explicit (documented) or inferred?

**4.2 High-Level System Architecture**

Include: **Figure 1: System Architecture Overview**

Describe the major components and how they relate:
- Client applications (web, mobile, desktop)
- API gateway / BFF layer (if applicable)
- Application services
- Background workers
- External integrations
- Data stores
- Infrastructure layer

For each component, write a descriptive paragraph covering its purpose, technology, and how it connects to adjacent components.

**4.3 Component Interaction Model**
How do the major components communicate? Document:
- Synchronous communication paths (REST, GraphQL, gRPC)
- Asynchronous communication paths (queues, events, pub/sub)
- Data contracts between components (API contracts, event schemas)
- Failure modes and circuit breaker patterns

Include: **Figure 2: Component Interaction Diagram**

**4.4 Module & Package Architecture**

For monolithic applications:
Document the internal module structure. How is the code organized? Are there clear bounded contexts? What is the coupling between modules? What is the cohesion within modules?

For microservices:
Provide a service inventory table:

| Service Name | Purpose | Technology | Port | DB | Dependencies |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

**4.5 Data Flow Architecture**
Trace data as it flows through the system for the primary use cases:
- How does a request enter the system?
- How is it authenticated and authorized?
- How does it reach business logic?
- How is data read from and written to persistent stores?
- How is the response assembled and returned?

Include: **Figure 3: Primary Data Flow Diagram**

**4.6 Architectural Strengths**
What does the architecture do well? Be specific, reference actual code patterns.

**4.7 Architectural Concerns**
What are the architectural weaknesses or areas of risk? Be specific and objective, not editorial.

---

### SECTION 5 — DATABASE DESIGN & DATA ARCHITECTURE (3–4 pages)

**5.1 Data Storage Overview**
List all data stores in use:
- Primary relational database
- Cache stores
- Search indexes
- Object/file storage
- Message/event stores
- Any other persistence

For each, document: purpose, technology, hosting, estimated data volume (if determinable), and criticality.

**5.2 Primary Database Schema**

Include: **Figure 4: Entity Relationship Diagram (Core Entities)**

Document each significant table/collection:

**Table: [table_name]**
- Purpose: [What does this table represent?]
- Key fields: [List with types and descriptions]
- Relationships: [FK references, join tables]
- Indexes: [List performance indexes]
- Special considerations: [Soft delete, auditing, partitioning, etc.]

For each major domain area, write a narrative paragraph explaining the data model design decisions and trade-offs.

**5.3 Schema Evolution & Migration Strategy**
- Number of migrations in the history
- Migration tool and approach
- Backward compatibility approach
- Zero-downtime migration strategy (if implemented)
- Data backfill patterns used

**5.4 Data Access Patterns**
- ORM vs. raw query usage ratio
- Most complex queries identified (with explanation)
- N+1 query risk assessment
- Eager loading patterns
- Pagination strategy
- Full-text search implementation

**5.5 Caching Architecture**
- What is cached (API responses, computed values, session data, etc.)
- Cache layer technology and configuration
- Cache invalidation strategy
- Cache-aside vs. write-through patterns
- TTL strategy
- Cache warming approach (if any)

**5.6 Data Integrity & Consistency**
- Transaction scope and patterns
- Database-level constraints vs. application-level validation
- Eventual consistency patterns (if any)
- Data validation library and approach

---

### SECTION 6 — BACKEND API ARCHITECTURE & BUSINESS LOGIC (4–5 pages)

**6.1 API Design Philosophy**
What API paradigm is used? What design principles are followed? (RESTful, resource-oriented, HATEOAS, schema-first, etc.)

**6.2 API Surface Documentation**

Organize endpoints by domain. For each domain:

#### [Domain Name] API

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | /api/v1/resource | Required | List all resources |
| POST | /api/v1/resource | Required | Create resource |
| ... | ... | ... | ... |

Follow each table with a paragraph describing the domain's overall design.

**6.3 Request Lifecycle**

Trace a request from ingress to response:

1. **Request arrives** at [load balancer / reverse proxy / server]
2. **Middleware chain**: [list each middleware in order with its function]
3. **Route matching**: [routing mechanism description]
4. **Authentication**: [how identity is established]
5. **Authorization**: [how access is controlled]
6. **Input validation**: [validation approach and library]
7. **Business logic execution**: [where this happens]
8. **Data access**: [how data is retrieved/mutated]
9. **Response serialization**: [how responses are formatted]
10. **Response delivery**: [any post-processing, caching headers, etc.]

Include: **Figure 5: Request Processing Pipeline**

**6.4 Business Logic Architecture**
Describe how business logic is organized and implemented:
- Service layer structure
- Domain logic encapsulation
- Use case / interactor pattern (if used)
- Business rules and validation
- State machines (if used)
- Event sourcing (if used)

**6.5 Core Business Process Deep-Dives**

For each major business process (e.g., user registration, order creation, payment processing, content publishing):

**Process: [Name]**
- Trigger: [What initiates this process]
- Actors: [Who/what is involved]
- Steps: [Numbered walkthrough of logic]
- Data mutations: [What changes in the database]
- Side effects: [Notifications, events, queue jobs, etc.]
- Error scenarios: [How failures are handled]
- Business rules: [Key constraints and validations]

**6.6 Background Jobs & Async Processing**
For each job type:
- Job name and purpose
- Trigger (scheduled, event-driven, user-initiated)
- Queue and worker configuration
- Retry strategy
- Idempotency considerations
- Monitoring approach

**6.7 Error Handling Strategy**
- Error classification (user errors vs. system errors vs. external failures)
- Error response format
- HTTP status code usage
- Internal error logging
- Error notification/alerting
- Graceful degradation patterns

---

### SECTION 7 — FRONTEND ARCHITECTURE & USER EXPERIENCE (3–4 pages)

**7.1 Frontend Technology Stack**
Document the complete frontend stack:
- Framework and version, with rendering strategy
- TypeScript usage level and configuration strictness
- Build toolchain and configuration
- Development experience tooling (HMR, dev server, etc.)

**7.2 Application Structure**
Explain how the frontend application is organized:
- Directory structure and conventions
- Route organization (list all major routes)
- Feature module organization
- Shared component library overview
- Utility and hook organization

**7.3 Component Architecture**
Describe the component design system:
- Component hierarchy (atoms → molecules → organisms → pages or similar)
- Prop patterns and component contracts
- Component composition patterns
- Storybook or component documentation (if present)

**7.4 State Management Architecture**
Describe the state management approach in depth:
- Global vs. local state split
- Server state management (data fetching, caching)
- UI state management
- Form state management
- State persistence (localStorage, sessionStorage, URL state)
- State synchronization (real-time, polling)

**7.5 User Flow Documentation**

For each major user-facing flow, provide a narrative walkthrough:

**Flow: [Name] (e.g., User Registration & Onboarding)**
- Entry point: [Route or action]
- Step-by-step UI journey: [Describe screens, interactions]
- API calls: [Which endpoints are called, with what data]
- State changes: [What state is updated]
- Success path: [What the user sees on success]
- Error handling: [What the user sees on failure]
- Edge cases: [Notable edge case handling]

Include: **Figure 6–8: Key User Flow Diagrams**

**7.6 Performance Architecture**
- Code splitting strategy and implementation
- Lazy loading: routes, components, images
- Bundle size analysis (if build config is present)
- Critical path rendering optimization
- Image optimization pipeline
- Font loading strategy

**7.7 Accessibility & Internationalization**
- Accessibility implementation signals (ARIA, keyboard nav, semantic HTML)
- i18n implementation (if present)
- RTL support (if applicable)
- Responsive design breakpoints and approach

---

### SECTION 8 — AUTHENTICATION, AUTHORIZATION & SECURITY (3–4 pages)

**8.1 Authentication Architecture**

Provide a complete description of how users and services authenticate:
- Authentication mechanisms in use
- Session/token lifecycle (issuance, validation, refresh, revocation)
- Credential storage (where and how)
- Multi-factor authentication
- OAuth/SSO implementation details

Include: **Figure 9: Authentication Flow Diagram**

**8.2 Authorization Model**

Document the permission system in detail:
- RBAC roles table with descriptions
- Permission matrix (roles × resources × actions)
- Resource ownership rules
- Scope-based authorization (for APIs or service-to-service)
- Admin vs. user access boundaries

**8.3 Security Controls Inventory**

Present a comprehensive security controls table:

| Control | Implementation | Status | Notes |
|---|---|---|---|
| Input validation | Zod schemas at API boundary | ✅ Implemented | Server-side only |
| SQL injection prevention | Prisma parameterized queries | ✅ Implemented | No raw queries found |
| XSS prevention | React DOM encoding | ✅ Implemented | No dangerouslySetInnerHTML |
| CSRF protection | SameSite cookies + CSRF token | ✅ Implemented | — |
| Rate limiting | express-rate-limit, 100/min | ✅ Implemented | Per IP |
| Security headers | Helmet.js | ✅ Implemented | CSP not configured |
| Password hashing | bcrypt (cost factor 12) | ✅ Implemented | — |
| HTTPS enforcement | HSTS header | ✅ Implemented | — |
| ... | ... | ... | ... |

**8.4 Secret & Credential Management**

Describe how secrets are managed:
- Secret categories present (DB credentials, API keys, JWT secrets, etc.)
- Storage mechanism (env vars, secrets manager, etc.)
- Access control for secrets
- Secret rotation practices (if determinable)
- Any hard-coded secrets found (critical issue — must flag)

**8.5 Security Risks & Vulnerabilities Identified**

Present findings in a risk table:

| # | Risk | Severity | Location | Description | Recommendation |
|---|---|---|---|---|---|
| 1 | [Risk] | Critical/High/Medium/Low | [File/component] | [Description] | [Fix] |
| ... | | | | | |

**Note:** Findings must be based on actual code observations, not assumptions. Do not invent vulnerabilities; do not omit real ones.

**8.6 Compliance Posture**

Based on observed code and configuration:
- GDPR readiness signals (consent, data deletion, data portability)
- PCI DSS signals (if payment data present)
- HIPAA signals (if health data present)
- SOC 2 readiness signals (logging, access control, incident response)

---

### SECTION 9 — INFRASTRUCTURE, DEPLOYMENT & DEVOPS (3–4 pages)

**9.1 Infrastructure Overview**
Provide a complete picture of the deployment target:
- Cloud provider(s) and regions
- Compute model (containers, serverless, VMs, PaaS)
- Database hosting
- Storage services
- Networking architecture
- Estimated infrastructure cost profile (high/medium/low, based on observable configuration)

Include: **Figure 10: Infrastructure Architecture Diagram**

**9.2 Containerization Architecture**

If Docker is in use:
- Dockerfile(s) analysis — base image choices, multi-stage build pattern, security (non-root user, minimal surface)
- Docker Compose service topology
- Image size and build optimization assessment

If Kubernetes is in use:
- Cluster architecture (if determinable)
- Deployment configurations
- Service mesh (if present)
- Ingress configuration
- Persistent volume claims
- ConfigMaps and Secrets management
- Horizontal pod autoscaling

**9.3 CI/CD Pipeline Architecture**

For each pipeline file found:
- Pipeline name and trigger conditions
- Stage-by-stage description
- Test gate requirements
- Build and packaging steps
- Deployment targets and strategy
- Rollback capability
- Environment promotion flow

Include: **Figure 11: CI/CD Pipeline Flow**

**9.4 Environment Configuration**

Document all environment tiers:

| Concern | Development | Staging | Production |
|---|---|---|---|
| Database | Local Docker | [Cloud instance type] | [Cloud instance type] |
| Auth | Dev credentials | Staging secrets | Production secrets |
| External services | Mocked / sandbox | Sandbox APIs | Live APIs |
| Logging | Console | Aggregated | Aggregated + alerts |
| ... | ... | ... | ... |

**9.5 Database Operations**
- Migration execution process (manual, automated in CI/CD)
- Zero-downtime migration approach
- Backup strategy (if determinable)
- Disaster recovery configuration

**9.6 Observability Architecture**
Describe the full observability stack:
- Logging: strategy, format (structured/unstructured), aggregation, retention
- Metrics: what is measured, collection mechanism, visualization
- Tracing: distributed tracing implementation
- Error tracking: tool, alerting rules
- Uptime monitoring: health check endpoints

**9.7 Operational Runbook Signals**
What operational procedures can be inferred from the codebase?
- Health check endpoints and their checks
- Graceful shutdown handling
- Feature flag rollout capability
- Maintenance mode capability
- Manual job execution capability

---

### SECTION 10 — EXTERNAL INTEGRATIONS & THIRD-PARTY SERVICES (2–3 pages)

**10.1 Integration Landscape Overview**

Present the integration map:

| Service | Category | Integration Method | Data Sent | Criticality | Failsafe |
|---|---|---|---|---|---|
| Stripe | Payments | REST API + Webhooks | Payment data | Critical | Queued retry |
| SendGrid | Email | REST API | User data, templates | High | Fallback logging |
| ... | ... | ... | ... | ... | ... |

Include: **Figure 12: Integration Architecture Map**

**10.2 Integration Deep-Dives**

For each Critical or High criticality integration:

**Integration: [Service Name]**
- Business purpose
- Technical implementation (SDK vs. direct API, async vs. sync)
- Authentication mechanism
- Request/response patterns (with sanitized examples)
- Webhook handling (if applicable, including signature verification)
- Error handling and retry logic
- Data flow: what leaves the system, what enters
- Failure impact: what happens if this service is unavailable
- Cost model (if determinable from configuration)

**10.3 Webhook Architecture**
If the system handles webhooks (inbound):
- Webhook receiver endpoints
- Signature verification implementation
- Idempotency handling
- Event processing strategy (synchronous vs. queued)
- Webhook logs and replay capability

If the system emits webhooks (outbound):
- Event types exposed to customers
- Delivery guarantees
- Retry mechanism
- Webhook management UI (if present)

---

### SECTION 11 — TESTING STRATEGY & QUALITY ASSURANCE (2 pages)

**11.1 Testing Philosophy**
Describe the overall testing approach and philosophy (if documented) or infer it from the test structure.

**11.2 Test Coverage Analysis**

| Test Type | Framework | Location | Estimated Coverage | Quality Assessment |
|---|---|---|---|---|
| Unit | Jest | src/**/__tests__ | ~65% (estimated) | Good |
| Integration | Supertest | test/integration | Partial | Needs expansion |
| E2E | Playwright | e2e/ | Happy paths only | Limited |
| ... | ... | ... | ... | ... |

**11.3 Test Quality Assessment**
- Are tests testing behavior or implementation?
- Test isolation quality (are tests independent and repeatable?)
- Test data management approach
- Are critical business flows well-covered?
- Notable gaps in test coverage

**11.4 Quality Tooling Configuration**
- TypeScript configuration (tsconfig.json strictness level)
- ESLint ruleset and custom rules
- Prettier configuration
- Pre-commit hook setup
- CI quality gates

**11.5 Code Quality Metrics** (where determinable)
- Cyclomatic complexity signals (very long functions, deeply nested logic)
- Code duplication signals
- Dead code signals
- Dependency direction violations

---

### SECTION 12 — PERFORMANCE, SCALABILITY & RESILIENCE (2–3 pages)

**12.1 Performance Architecture**
Analyze performance-relevant design decisions:
- Database query optimization (indexes, N+1 prevention, caching)
- API response time optimization (pagination, field selection, compression)
- Frontend performance (bundle splitting, lazy loading, SSR/SSG)
- CDN usage for static assets and API caching

**12.2 Scalability Analysis**

Assess scalability across dimensions:

| Dimension | Current Design | Scalability Ceiling | Scaling Path |
|---|---|---|---|
| Web/API tier | Stateless containers | High | Horizontal + LB |
| Database | Single primary | Medium | Read replicas → Sharding |
| Background jobs | N workers | Medium | More worker pods |
| File storage | S3/object store | Very High | Already horizontally scaled |
| ... | ... | ... | ... |

**12.3 Resilience Patterns**
Document fault tolerance mechanisms:
- Retry patterns (exponential backoff, jitter)
- Circuit breakers
- Graceful degradation
- Rate limiting and backpressure
- Health check and readiness probes
- Graceful shutdown
- Database connection resilience

**12.4 Identified Performance Risks**
List specific performance risks observed:
- Missing database indexes (if identified)
- N+1 query patterns (if found)
- Synchronous operations that should be async
- Missing caching on expensive operations
- Large payload sizes without pagination
- Missing CDN for static assets

---

### SECTION 13 — CROSS-CUTTING CONCERNS (1–2 pages)

**13.1 Logging & Observability in Code**
How is logging implemented at the application level?
- Logging library and configuration
- Log levels and their usage
- Structured logging (JSON) vs. plain text
- Correlation ID / request ID propagation
- Sensitive data in logs (PII leakage risk)
- Audit log implementation

**13.2 Configuration Management**
How is application configuration managed?
- Environment variable loading mechanism
- Configuration validation at startup
- Configuration typing and documentation
- Feature flag implementation (if any)
- Multi-environment configuration strategy

**13.3 Internationalization & Localization**
- i18n implementation details
- Translation management approach
- Currency, date, number formatting
- Locale detection strategy

**13.4 Notification Architecture**
The complete notification system:
- Notification types (email, SMS, push, in-app)
- Template management
- User notification preferences
- Notification delivery reliability

**13.5 File Upload & Media Handling**
- Upload flow: client → server → storage
- Accepted file types and size limits
- Malware scanning (if implemented)
- Image processing (resize, optimize, transform)
- CDN delivery

---

### SECTION 14 — TECHNICAL DEBT & RISK ASSESSMENT (2–3 pages)

**14.1 Technical Debt Inventory**

Categorize and document technical debt:

| # | Item | Category | Severity | Effort | Impact |
|---|---|---|---|---|---|
| 1 | [Debt item] | [Code/Architecture/Testing/Docs/Security] | High/Med/Low | High/Med/Low | High/Med/Low |
| ... | | | | | |

Categories:
- **Code Debt**: specific patterns, anti-patterns, workarounds found
- **Architecture Debt**: structural issues, coupling, missing abstractions
- **Testing Debt**: missing test coverage, fragile tests
- **Documentation Debt**: missing or outdated documentation
- **Security Debt**: security controls that are absent or weak
- **Dependency Debt**: outdated or risky dependencies
- **Infrastructure Debt**: deployment or operational shortcomings

**14.2 Risk Register**

| # | Risk | Probability | Impact | Risk Score | Mitigation Strategy |
|---|---|---|---|---|---|
| 1 | [Risk description] | H/M/L | H/M/L | [1–9] | [Strategy] |
| ... | | | | | |

Risk scoring: Probability × Impact (H=3, M=2, L=1). Score 7–9 = Critical, 4–6 = High, 1–3 = Medium.

**14.3 Dependency Risk Assessment**
Flag dependencies with:
- Known CVEs in the pinned version
- No updates in 12+ months (abandonware risk)
- Single maintainer with no succession plan
- License risk (GPL contamination, BUSL, etc.)
- Major version behind with breaking changes

**14.4 Operational Risk Assessment**
- Single points of failure in the architecture
- Backup and recovery gaps
- Monitoring blind spots
- On-call / incident response readiness
- Bus factor (knowledge concentration)

---

### SECTION 15 — ARCHITECTURE DECISION RECORDS & DESIGN RATIONALE (1–2 pages)

**15.1 Reconstructed Architecture Decisions**

For each significant technical choice made (whether explicitly documented or inferred), create an ADR-format entry:

**ADR-[N]: [Decision Title]**
- **Status**: Accepted (implemented) / Unclear (inferred)
- **Context**: What problem or constraint drove this decision?
- **Decision**: What was chosen?
- **Rationale**: Why was this chosen over alternatives?
- **Consequences**: What are the positive and negative implications?
- **Alternatives Considered**: [If inferrable]

Example ADR topics: choice of ORM, choice of auth strategy, monolith vs. microservices, choice of queue system, database choice, API paradigm, frontend framework.

**15.2 Implicit Design Patterns**
Patterns that appear consistently throughout the codebase but may not be explicitly documented:
- Naming conventions
- Error handling conventions
- Code organization conventions
- Testing conventions
- API response conventions

---

### SECTION 16 — RECOMMENDATIONS & IMPROVEMENT ROADMAP (2–3 pages)

**16.1 Recommendation Framework**

Recommendations are organized by:
- **Priority**: P0 (Critical/Immediate), P1 (High/Next Sprint), P2 (Medium/Next Quarter), P3 (Low/Backlog)
- **Category**: Security, Performance, Architecture, Testing, DevOps, Documentation
- **Effort**: Small (1–3 days), Medium (1–2 weeks), Large (1+ month)
- **Impact**: Critical, High, Medium, Low

**16.2 P0 — Immediate Actions Required**

For each P0 item:

**[REC-P0-N]: [Title]**
- **Finding**: [What was observed]
- **Risk**: [What could go wrong]
- **Recommendation**: [Specific, actionable steps]
- **Files/Components Affected**: [List]
- **Effort**: [Estimate]

**16.3 P1 — High Priority Improvements**

Same format as P0, with slightly less urgency.

**16.4 P2 — Strategic Improvements**

For larger-effort, high-impact improvements:
- Architecture evolution recommendations
- Scalability improvements
- Testing coverage expansion
- Developer experience improvements

**16.5 P3 — Backlog Items**

Lower priority but valuable improvements. Presented as a table for brevity.

**16.6 Improvement Roadmap**

Present a phased roadmap:

| Phase | Timeline | Focus Areas | Key Deliverables |
|---|---|---|---|
| Phase 1 | Month 1 | Security & Critical Debt | [List] |
| Phase 2 | Months 2–3 | Architecture & Performance | [List] |
| Phase 3 | Months 4–6 | Testing & Observability | [List] |
| Phase 4 | Months 6–12 | Strategic Evolution | [List] |

---

### SECTION 17 — APPENDICES

**Appendix A: Complete File Structure Map**
Full directory tree of the repository with purpose annotations.

**Appendix B: Complete API Endpoint Reference**
Full table of all endpoints found, organized by service/domain.

**Appendix C: Database Schema Reference**
Complete schema dump or structured table of all database tables/collections and their fields.

**Appendix D: Environment Variable Reference**
All environment variables found across .env.example files and code references, categorized:
| Variable | Category | Description | Required | Default |
|---|---|---|---|---|

**Appendix E: Dependency Manifest**
Complete list of direct dependencies with versions and purpose.

**Appendix F: Security Controls Checklist**
Full OWASP Top 10 / security checklist with observed status for each item.

**Appendix G: Glossary**
Domain-specific and technical terms used throughout the report with definitions.

---

## ══════════════════════════════════════════════════════
## PHASE 3: QUALITY ASSURANCE BEFORE DELIVERY
## ══════════════════════════════════════════════════════

Before delivering the report, perform this final quality pass:

### QA CHECKLIST

**Accuracy**
- [ ] Every technical claim is traceable to actual observed code/configuration
- [ ] No hallucinated features, endpoints, or behaviors
- [ ] Version numbers are accurate to what was found
- [ ] File paths referenced actually exist in the repository
- [ ] Architecture diagrams accurately reflect the code structure

**Completeness**
- [ ] All 17 sections are present and substantive
- [ ] Executive Summary is self-contained and accurate
- [ ] All major features are covered in the Feature Inventory
- [ ] All critical security findings are documented
- [ ] Recommendations are specific and actionable

**Quality**
- [ ] Tone is professional, authoritative, and consistent
- [ ] No paragraph is pure padding or restated information
- [ ] Tables are complete, accurate, and well-formatted
- [ ] Technical depth is appropriate for the intended audience
- [ ] Document flows logically from overview to detail

**Length**
- [ ] Report is substantively 30–40 pages (15,000–22,000 words equivalent)
- [ ] No section is disproportionately thin or bloated
- [ ] Appendices add genuine reference value

---

## ══════════════════════════════════════════════════════
## INVOCATION INSTRUCTIONS
## ══════════════════════════════════════════════════════

To use this prompt, append it with ONE of the following invocation modes:

### MODE A — FULL REPOSITORY PATH
```
[PASTE THIS ENTIRE PROMPT]

REPOSITORY LOCATION: [path to repository root or provide file tree below]

BEGIN ANALYSIS. Execute Phase 1 completely before generating any section of the report.
```

### MODE B — PROVIDE FILES DIRECTLY
```
[PASTE THIS ENTIRE PROMPT]

I will now provide the repository files for analysis. Read ALL provided files completely
before generating the report.

[PASTE OR ATTACH REPOSITORY FILES]

BEGIN ANALYSIS. Execute Phase 1 completely before generating any section of the report.
```

### MODE C — AGENT MODE (Claude with filesystem access)
```
[PASTE THIS ENTIRE PROMPT]

You have access to the repository at: [REPO_ROOT_PATH]

Use your file system tools to systematically execute Phase 1, reading every relevant
file. Maintain an internal analysis document as you go. Only begin Phase 2 after 
Phase 1 is complete.

BEGIN ANALYSIS.
```

---

## FINAL NOTE TO THE AI

You are producing a document that will be read by engineers, architects, product managers, and potentially investors or auditors. It must be:

- **Accurate**: Based on evidence, never invented
- **Comprehensive**: Cover the entire system without critical omissions
- **Professional**: Written with the authority of a senior consultant
- **Useful**: Actionable findings, not just descriptions
- **Premium**: Formatted and structured to the standard of enterprise technical documentation

The quality of this report reflects the quality of the AI's analysis capability. Approach it with the rigor of a technical audit, the narrative skill of an experienced technical writer, and the strategic insight of a solutions architect.

**Begin Phase 1. Do not start writing the report until Phase 1 is complete.**

---

*End of Master Prompt — Repository Analysis & Technical Report Generation v2.0*
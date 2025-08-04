# Conversational Experiment Designer

## Overview
A natural language interface for laboratory scientists to design experiments and generate statistical models using AI through structured design sessions.

## Design Sessions

### Concept
A **Design Session** is a persistent workspace where scientists iteratively develop and refine experimental designs. Each session maintains conversation history, experiment evolution, and generated artifacts.

### Session Lifecycle
1. **Create Session**: Scientist starts with a research question or rough experiment idea
2. **Iterative Design**: Multiple rounds of description → parsing → refinement → validation
3. **Model Generation**: Generate PyMC models and sample data when design is mature
4. **Export/Save**: Export final designs, models, and documentation
5. **Version Control**: Track design evolution and maintain multiple design variants

### Session Components
- **Session Metadata**: ID, title, creation date, last modified, scientist info
- **Conversation History**: Complete dialogue between scientist and AI assistant
- **Experiment Versions**: Snapshots of `ExperimentDescription` at different iteration stages
- **Generated Artifacts**: PyMC models, sample data, visualizations, documentation
- **Notes & Annotations**: Scientist's thoughts, decisions, and rationale
- **Validation History**: Record of design issues identified and resolved

## Core Components

### 1. Natural Language Input

- Large text area for scientists to describe experiments in plain English
- Real-time parsing and validation as they type
- Auto-save and draft functionality

### 2. AI Assistant

- Real-time parsing of experimental descriptions
- Validation of experimental design completeness
- Suggests missing factors and asks clarifying questions
- Provides immediate feedback on design issues

### 3. Interactive Refinement

- AI suggests missing experimental factors
- Asks targeted questions to clarify ambiguous descriptions
- Allows scientists to correct AI interpretations
- Maintains conversation history for context

### 4. Interactive Experiment Design Canvas

**Layout Structure:**

- **Three-panel layout**: Left sidebar (conversation), center canvas (experiment design), right panel (AI assistant)
- **Responsive design** that adapts to different screen sizes
- **Drag-and-drop interface** for factor management and reordering

**Component Cards:**

- **Treatment Factors**: Blue cards with treatment levels, randomization info, and effect size expectations
- **Nuisance Factors**: Gray cards showing sources of variation to control (plate effects, day effects, operator effects)
- **Blocking Factors**: Green cards displaying blocking structure and randomization within blocks
- **Covariates**: Orange cards with measurement details and expected relationships
- **Response Variables**: Purple cards showing outcome measures and measurement protocols

**Card Features:**

- **Expandable details**: Click to show full factor specifications
- **Inline editing**: Double-click to edit factor properties directly
- **AI assistant button**: Each card has a dedicated AI assistant for refinement
- **Validation indicators**: Color-coded status (valid/green, warning/yellow, error/red)
- **Relationship lines**: Visual connections showing factor interactions and nesting

**Canvas Interactions:**

- **Add factor button**: Floating action button to add new factors
- **Factor grouping**: Visual grouping of related factors
- **Zoom and pan**: Navigate large experiment designs
- **Search and filter**: Find specific factors quickly

## User Flow

### Starting a Design Session
1. **Create New Session**:
   - Click "New Design Session" button on dashboard
   - Modal dialog with session title and optional description
   - Choose from template designs or start from scratch
2. **Initial Description**:
   - Large text area with placeholder text and examples
   - Real-time character count and auto-save indicator
   - "Start Design" button appears when minimum content is entered
3. **AI Parsing**:
   - Loading animation with "Analyzing your experiment..." message
   - Progress indicator showing parsing steps (factors, structure, validation)
   - Preview of parsed components before confirmation
4. **Session Workspace**:
   - Three-panel layout loads with parsed experiment displayed as cards
   - Welcome message in AI assistant panel with next steps
   - Auto-save confirmation and session URL for sharing

### Iterative Design Process
5. **Design Canvas**:
   - Experiment displayed as color-coded factor cards in center panel
   - Cards show factor name, type, current values, and validation status
   - Relationship lines connect interacting factors
   - Empty state with "Add your first factor" when canvas is empty
6. **Interactive Refinement**: Scientist can:
   - **Edit components**: Double-click cards or use edit button to open detail panel
   - **AI assistance**: Click AI button on any card for contextual help
   - **Add factors**: Use floating "+" button or AI assistant suggestions
   - **Remove factors**: Delete button with confirmation dialog
   - **Reorder factors**: Drag and drop to change factor arrangement
7. **Real-time Validation**:
   - Validation status shown as colored indicators on each card
   - Error messages appear in AI assistant panel with fix suggestions
   - Overall experiment status in status bar (Ready/Issues/Complete)
8. **Version Snapshots**:
   - Auto-save creates versions at significant changes
   - Version timeline in left sidebar shows evolution
   - Ability to revert to previous versions with diff view

### Model Generation & Execution
9. **Generate Models**:
    - "Generate Model" button appears in toolbar when design is valid
    - Progress indicator showing model generation steps
    - Code preview in modal dialog with syntax highlighting
    - Option to edit generated code before finalizing
10. **Execute Models**:
    - "Run Model" button executes the generated PyMC model in background
    - Real-time progress showing sampling steps and convergence
    - Live updates of sampling diagnostics (ESS, R-hat, trace plots)
    - Automatic stopping when convergence criteria are met
11. **Generate Sample Data**:
    - "Generate Data" button creates PEP723-compatible scripts for data simulation
    - Scripts sample from prior with configurable number of draws
    - Automatic CSV export with realistic experimental structure
    - Data validation showing expected vs. actual distributions
    - Option to download script or run directly in browser
12. **Export & Share**:
    - Export menu with multiple formats (PDF report, JSON session, Python script)
    - PEP723-compatible script generation for reproducible data simulation
    - Customizable export options (include conversation, include data, etc.)
    - Progress bar for large exports
    - Shareable links for collaboration

### Session Management
12. **Save/Load**:
    - Auto-save indicator in status bar (last saved: 2 minutes ago)
    - Manual save button with keyboard shortcut (Ctrl+S)
    - Session recovery if browser crashes or connection lost
    - Offline mode with sync when connection restored
13. **Share/Collaborate**:
    - Share button generates unique URL for session
    - Permission settings (view-only, comment, edit)
    - Real-time collaboration indicators showing who's online
    - Comment system for feedback and suggestions
14. **Template Creation**:
    - "Save as Template" option in session menu
    - Template gallery showing available designs
    - Template categories (screening, optimization, validation)
    - Template rating and usage statistics

## Technical Requirements

### Core Infrastructure
- **Session Management**: Persistent storage for design sessions with metadata
- **Version Control**: Track evolution of `ExperimentDescription` over time
- **Conversation History**: Store complete dialogue between scientist and AI
- **Auto-save**: Real-time persistence of session state and changes

### AI Integration
- **Stateful Conversations**: Maintain context across multiple interactions
- **Integration with existing `experiment_bot`** for parsing
- **Real-time validation** using current validation pipeline
- **Contextual suggestions** based on session history and current state

### User Interface

**Session Dashboard:**
- **Grid layout** of session cards showing title, last modified, status, and preview
- **Quick actions**: Resume, duplicate, export, delete for each session
- **Search and filter**: Find sessions by title, date, or content
- **Create new session**: Prominent "New Design Session" button
- **Recent sessions**: Horizontal scrollable list of recently accessed sessions

**Main Workspace Layout:**
- **Left Sidebar (30% width)**: Conversation history with collapsible sections
  - **Notes section**: Collapsible area for scientist's private notes
  - **Transcript section**: Collapsible area for full conversation history
  - **Summary section**: Always visible high-level session overview
- **Center Canvas (50% width)**: Interactive experiment design area
  - **Toolbar**: Add factor, generate model, export, save buttons
  - **Design canvas**: Drag-and-drop factor management
  - **Status bar**: Validation status, factor count, session info
- **Right Panel (20% width)**: Contextual AI assistant
  - **Current focus**: Shows AI assistant for selected factor or general session
  - **Suggestions**: AI recommendations and validation feedback
  - **Quick actions**: Common operations for current context

**Component Cards Design:**
- **Card dimensions**: 280px × 200px with consistent spacing
- **Color coding**: Blue (treatments), Gray (nuisance), Green (blocking), Orange (covariates), Purple (response)
- **Header**: Factor name and type with status indicator
- **Body**: Key properties and current values
- **Footer**: Action buttons (edit, delete, AI assistant)
- **Hover effects**: Subtle elevation and highlight on mouse over

**AI Assistant Interface:**
- **Chat-like interface** with message bubbles
- **Context awareness**: Shows relevant information about current selection
- **Quick response buttons**: Common actions like "Add factor", "Fix validation error"
- **Voice input support**: Optional speech-to-text for hands-free operation
- **Suggestion chips**: Clickable suggestions for common operations

**Responsive Design:**
- **Desktop**: Full three-panel layout with all features
- **Tablet**: Collapsible sidebars with touch-friendly interactions
- **Mobile**: Single-column layout with bottom navigation
- **Accessibility**: Keyboard navigation, screen reader support, high contrast mode

### Data Management
- **Export to PyMC code generation pipeline**
- **Sample data generation** with realistic experimental structure
- **Session export/import** for sharing and collaboration
- **Template system** for reusable experiment patterns

## Backend API Design

### Technology Stack
- **FastAPI**: High-performance async API framework with automatic OpenAPI documentation
- **HTMX**: Server-side rendering with dynamic updates without JavaScript complexity
- **Pydantic**: Data validation and serialization for all API models
- **Async database operations** for responsive UI updates
- **WebSocket support** for real-time collaboration features

### API Architecture

**Core Principles:**
- **RESTful design** with clear resource-based endpoints
- **HTMX-first approach** - all endpoints return HTML fragments for seamless UI updates
- **Progressive enhancement** - API also supports JSON for programmatic access
- **Real-time updates** via Server-Sent Events (SSE) for live collaboration
- **Stateless design** with session state stored in database

### Session Management Endpoints

```python
# Session CRUD operations
POST   /api/sessions                    # Create new session
GET    /api/sessions                    # List user sessions (dashboard)
GET    /api/sessions/{session_id}       # Get session workspace
PUT    /api/sessions/{session_id}       # Update session metadata
DELETE /api/sessions/{session_id}       # Delete session

# Session content operations
GET    /api/sessions/{session_id}/conversation    # Get conversation history
POST   /api/sessions/{session_id}/conversation    # Add message to conversation
GET    /api/sessions/{session_id}/experiment      # Get current experiment
PUT    /api/sessions/{session_id}/experiment      # Update experiment
GET    /api/sessions/{session_id}/versions        # Get version history
POST   /api/sessions/{session_id}/versions        # Create new version
```

### Experiment Design Endpoints

```python
# Factor management
POST   /api/sessions/{session_id}/factors         # Add new factor
GET    /api/sessions/{session_id}/factors/{factor_id}  # Get factor details
PUT    /api/sessions/{session_id}/factors/{factor_id}  # Update factor
DELETE /api/sessions/{session_id}/factors/{factor_id}  # Remove factor

# AI assistance
POST   /api/sessions/{session_id}/ai/parse        # Parse natural language description
POST   /api/sessions/{session_id}/ai/suggest      # Get AI suggestions
POST   /api/sessions/{session_id}/ai/validate     # Validate current design
POST   /api/sessions/{session_id}/ai/refine       # AI-assisted refinement

# Model generation and execution
POST   /api/sessions/{session_id}/generate/model  # Generate PyMC model
POST   /api/sessions/{session_id}/execute/model   # Execute PyMC model (sampling)
POST   /api/sessions/{session_id}/generate/data   # Generate PEP723 script + sample data
GET    /api/sessions/{session_id}/model/status    # Get model execution status
DELETE /api/sessions/{session_id}/model/stop      # Stop running model execution
```

### HTMX-Specific Endpoints

```python
# HTMX partial updates
GET    /htmx/sessions/{session_id}/canvas         # Get experiment canvas HTML
GET    /htmx/sessions/{session_id}/conversation   # Get conversation HTML
GET    /htmx/sessions/{session_id}/assistant      # Get AI assistant HTML
GET    /htmx/sessions/{session_id}/factor/{factor_id}  # Get factor card HTML

# HTMX actions
POST   /htmx/sessions/{session_id}/add-factor     # Add factor (returns card HTML)
PUT    /htmx/sessions/{session_id}/update-factor  # Update factor (returns updated card)
DELETE /htmx/sessions/{session_id}/remove-factor  # Remove factor (returns canvas update)
POST   /htmx/sessions/{session_id}/ai-chat        # AI chat (returns response HTML)
```

### Real-time Collaboration

```python
# WebSocket endpoints
WS     /ws/sessions/{session_id}                  # Real-time session updates
WS     /ws/sessions/{session_id}/cursor           # Live cursor positions
WS     /ws/sessions/{session_id}/typing           # Typing indicators

# Server-Sent Events
GET    /sse/sessions/{session_id}/updates         # Live session updates
GET    /sse/sessions/{session_id}/validation      # Real-time validation status
```

### Request/Response Examples

**Create Session (HTMX):**
```http
POST /api/sessions
Content-Type: application/x-www-form-urlencoded

title=Protein Expression Study&description=I want to study protein expression...
```

**Response (HTMX):**
```html
<div id="session-workspace" hx-swap-oob="true">
  <div class="three-panel-layout">
    <div class="sidebar">
      <!-- Conversation history -->
    </div>
    <div class="canvas">
      <!-- Experiment cards -->
    </div>
    <div class="assistant">
      <!-- AI assistant -->
    </div>
  </div>
</div>
```

**Add Factor (HTMX):**
```http
POST /htmx/sessions/123/add-factor
Content-Type: application/x-www-form-urlencoded

factor_type=treatment&name=Temperature&levels=25,30,35
```

**Response:**
```html
<div class="factor-card treatment" data-factor-id="456">
  <div class="card-header">
    <h3>Temperature</h3>
    <span class="status valid"></span>
  </div>
  <div class="card-body">
    <p>Levels: 25, 30, 35</p>
  </div>
  <div class="card-actions">
    <button hx-get="/htmx/sessions/123/factor/456/edit">Edit</button>
    <button hx-delete="/htmx/sessions/123/remove-factor/456">Delete</button>
  </div>
</div>
```

### API Models (Pydantic)

```python
class SessionCreate(BaseModel):
    title: str
    description: Optional[str] = None
    template_id: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    title: str
    description: Optional[str]
    created_at: datetime
    last_modified: datetime
    status: SessionStatus
    factor_count: int
    validation_status: ValidationStatus

class FactorCreate(BaseModel):
    factor_type: FactorType
    name: str
    levels: List[str]
    description: Optional[str] = None
    randomization: Optional[RandomizationType] = None

class AIParseRequest(BaseModel):
    description: str
    context: Optional[Dict] = None

class AIParseResponse(BaseModel):
    experiment: ExperimentDescription
    confidence: float
    suggestions: List[str]
    validation_issues: List[ValidationIssue]
```

### Error Handling

```python
# Standard error responses
class APIError(BaseModel):
    error: str
    message: str
    details: Optional[Dict] = None

# Common error codes
400: Bad Request (validation errors)
404: Session/Factor not found
409: Concurrent modification conflict
422: Validation error (Pydantic)
500: Internal server error
```

### Performance Considerations

**Database Optimization:**
- **Connection pooling** for database operations
- **Async queries** to prevent blocking
- **Caching strategy** for frequently accessed session data
- **Indexing** on session_id, user_id, created_at

**HTMX Optimization:**
- **Partial updates** to minimize HTML transfer
- **Debounced requests** for real-time validation
- **Lazy loading** for conversation history
- **Compression** for large HTML responses

**Real-time Performance:**
- **WebSocket connection pooling**
- **Event batching** for multiple updates
- **Client-side buffering** for offline support

## UI to API Mapping

### Dashboard Interface

**Session Grid Cards:**
```html
<!-- Each session card -->
<div class="session-card" data-session-id="123">
  <h3>Protein Expression Study</h3>
  <p>Last modified: 2 hours ago</p>
  <div class="card-actions">
    <button hx-get="/htmx/sessions/123/workspace"
            hx-target="#main-content">Resume</button>
    <button hx-post="/htmx/sessions/123/duplicate"
            hx-target="#session-grid">Duplicate</button>
    <button hx-delete="/htmx/sessions/123"
            hx-confirm="Delete this session?">Delete</button>
  </div>
</div>
```

**API Endpoints:**
- `GET /htmx/sessions/123/workspace` → Loads full session workspace
- `POST /htmx/sessions/123/duplicate` → Creates copy, returns new session card
- `DELETE /htmx/sessions/123` → Removes session, updates grid

### Main Workspace Layout

**Left Sidebar (Conversation History):**
```html
<div class="sidebar">
  <div class="summary-section">
    <!-- Always visible summary -->
    <h4>Session Overview</h4>
    <p>3 treatment factors, 2 nuisance factors</p>
  </div>

  <details class="notes-section">
    <summary>Notes</summary>
    <div hx-get="/htmx/sessions/123/notes"
         hx-trigger="click once">
      <!-- Notes content loaded here -->
    </div>
  </details>

  <details class="transcript-section">
    <summary>Full Transcript</summary>
    <div hx-get="/htmx/sessions/123/conversation"
         hx-trigger="click once">
      <!-- Conversation history loaded here -->
    </div>
  </details>
</div>
```

**API Endpoints:**
- `GET /htmx/sessions/123/notes` → Returns notes HTML fragment
- `GET /htmx/sessions/123/conversation` → Returns conversation history HTML

### Center Canvas (Experiment Design)

**Factor Cards:**
```html
<!-- Treatment Factor Card -->
<div class="factor-card treatment" data-factor-id="456">
  <div class="card-header">
    <h3>Temperature</h3>
    <span class="status valid"></span>
  </div>
  <div class="card-body">
    <p>Levels: 25°C, 30°C, 35°C</p>
    <p>Randomization: Complete</p>
  </div>
  <div class="card-actions">
    <button hx-get="/htmx/sessions/123/factor/456/edit"
            hx-target="#modal-content">Edit</button>
    <button hx-post="/htmx/sessions/123/factor/456/ai-assist"
            hx-target="#assistant-panel">AI Help</button>
    <button hx-delete="/htmx/sessions/123/factor/456"
            hx-confirm="Remove this factor?">Delete</button>
  </div>
</div>
```

**API Endpoints:**
- `GET /htmx/sessions/123/factor/456/edit` → Returns edit modal HTML
- `POST /htmx/sessions/123/factor/456/ai-assist` → Updates AI assistant with factor context
- `DELETE /htmx/sessions/123/factor/456` → Removes factor, updates canvas

**Add Factor Button:**
```html
<button class="add-factor-btn"
        hx-get="/htmx/sessions/123/add-factor-modal"
        hx-target="#modal-content">
  <i class="plus-icon"></i> Add Factor
</button>
```

**API Endpoints:**
- `GET /htmx/sessions/123/add-factor-modal` → Returns add factor form
- `POST /htmx/sessions/123/factors` → Creates factor, returns new card HTML

### Right Panel (AI Assistant)

**AI Chat Interface:**
```html
<div class="assistant-panel">
  <div class="chat-messages" id="chat-messages">
    <!-- Messages loaded here -->
  </div>

  <form hx-post="/htmx/sessions/123/ai-chat"
        hx-target="#chat-messages"
        hx-swap="beforeend">
    <input type="text" name="message" placeholder="Ask about your experiment...">
    <button type="submit">Send</button>
  </form>

  <div class="suggestion-chips">
    <button hx-post="/htmx/sessions/123/ai-suggest"
            hx-target="#assistant-panel"
            hx-vals='{"suggestion": "add_blocking"}'>
      Add blocking factor
    </button>
    <button hx-post="/htmx/sessions/123/ai-suggest"
            hx-target="#assistant-panel"
            hx-vals='{"suggestion": "validate_design"}'>
      Validate design
    </button>
  </div>
</div>
```

**API Endpoints:**
- `POST /htmx/sessions/123/ai-chat` → Processes message, returns response HTML
- `POST /htmx/sessions/123/ai-suggest` → Handles suggestion, returns updated assistant

### Toolbar Actions

**Main Toolbar:**
```html
<div class="toolbar">
  <button hx-post="/htmx/sessions/123/save"
          hx-indicator="#save-indicator">
    <span id="save-indicator" class="htmx-indicator">Saving...</span>
    Save
  </button>

  <button hx-post="/htmx/sessions/123/generate/model"
          hx-target="#modal-content"
          hx-disabled-elt="this">
    Generate Model
  </button>

    <button hx-post="/htmx/sessions/123/execute/model"
          hx-target="#modal-content"
          hx-indicator="#model-indicator">
    <span id="model-indicator" class="htmx-indicator">Running...</span>
    Run Model
  </button>

  <button hx-post="/htmx/sessions/123/generate/data"
          hx-target="#modal-content">
    Generate Data Script
  </button>

  <button hx-get="/htmx/sessions/123/export"
          hx-target="#modal-content">
    Export
  </button>
</div>
```

**API Endpoints:**
- `POST /htmx/sessions/123/save` → Saves session, returns save status
- `POST /htmx/sessions/123/generate/model` → Generates PyMC model, returns code preview
- `POST /htmx/sessions/123/execute/model` → Executes model sampling, returns progress updates
- `POST /htmx/sessions/123/generate/data` → Generates PEP723 script, returns script preview
- `GET /htmx/sessions/123/export` → Returns export options modal

### Real-time Updates

**Live Validation Status:**
```html
<div class="status-bar">
  <span class="validation-status"
        hx-get="/htmx/sessions/123/validation-status"
        hx-trigger="every 5s">
    Valid ✓
  </span>
  <span class="factor-count">3 factors</span>
  <span class="last-saved">Saved 2 minutes ago</span>
</div>
```

**API Endpoints:**
- `GET /htmx/sessions/123/validation-status` → Returns current validation status
- `GET /sse/sessions/123/updates` → Server-Sent Events for real-time updates

### Modal Dialogs

**Edit Factor Modal:**
```html
<div id="modal-content">
  <form hx-put="/htmx/sessions/123/factor/456"
        hx-target="closest .factor-card">
    <label>Factor Name:</label>
    <input type="text" name="name" value="Temperature">

    <label>Levels:</label>
    <input type="text" name="levels" value="25,30,35">

    <button type="submit">Update</button>
    <button type="button" onclick="closeModal()">Cancel</button>
  </form>
</div>
```

**API Endpoints:**
- `PUT /htmx/sessions/123/factor/456` → Updates factor, returns updated card HTML

### Error Handling

**Error Display:**
```html
<div class="error-message"
     hx-swap-oob="true"
     hx-target="#error-container">
  <p>Error: Factor name is required</p>
  <button onclick="this.parentElement.remove()">Dismiss</button>
</div>
```

**API Error Responses:**
```html
<!-- 422 Validation Error -->
<div class="error-message">
  <h4>Validation Error</h4>
  <ul>
    <li>Factor name is required</li>
    <li>At least 2 levels are needed</li>
  </ul>
</div>
```

### Collaboration Features

**Share Session:**
```html
<button hx-get="/htmx/sessions/123/share"
        hx-target="#modal-content">
  Share Session
</button>

<!-- Share modal content -->
<div id="modal-content">
  <h3>Share Session</h3>
  <input type="text" value="https://app.example.com/sessions/123" readonly>
  <select name="permission">
    <option value="view">View only</option>
    <option value="comment">Can comment</option>
    <option value="edit">Can edit</option>
  </select>
  <button hx-post="/htmx/sessions/123/share"
          hx-target="#modal-content">
    Update Permissions
  </button>
</div>
```

**API Endpoints:**
- `GET /htmx/sessions/123/share` → Returns share settings modal
- `POST /htmx/sessions/123/share` → Updates sharing permissions

## Model Execution & PEP723 Script Generation

### Model Execution Workflow

**Background Execution:**
- **Async model execution** using Celery or similar task queue
- **Real-time progress updates** via Server-Sent Events
- **Resource management** with configurable memory and time limits
- **Graceful failure handling** with detailed error reporting

**Execution Status Tracking:**
```python
class ModelExecutionStatus(BaseModel):
    execution_id: str
    status: ExecutionStatus  # PENDING, RUNNING, COMPLETED, FAILED
    progress: float  # 0.0 to 1.0
    current_step: str  # "Initializing", "Sampling", "Convergence check"
    diagnostics: Optional[Dict] = None  # ESS, R-hat, trace plots
    results: Optional[Dict] = None  # Posterior samples, summary stats
    error_message: Optional[str] = None
```

### PEP723-Compatible Script Generation

**Script Structure:**
```python
# Generated PEP723 script example
# /// script
# dependencies = [
#     "pymc>=5.0",
#     "pandas>=2.0",
#     "numpy>=1.24"
# ]
# ///

import pymc as pm
import pandas as pd
import numpy as np

# Experiment: Protein Expression Study
# Generated from conversational experiment designer

def generate_sample_data(n_samples=100, random_seed=42):
    """Generate sample data from the experimental design."""

    with pm.Model() as model:
        # Treatment factor: Temperature
        temp_levels = [25, 30, 35]
        temp_effect = pm.Normal('temp_effect', mu=0, sigma=2)

        # Nuisance factor: Plate effects
        plate_effect = pm.Normal('plate_effect', mu=0, sigma=1)

        # Response variable: Protein expression
        mu = temp_effect + plate_effect
        protein_expression = pm.Normal('protein_expression', mu=mu, sigma=0.5)

        # Sample from prior
        trace = pm.sample_prior_predictive(samples=n_samples, random_seed=random_seed)

    # Convert to DataFrame
    data = pd.DataFrame({
        'temperature': np.random.choice(temp_levels, n_samples),
        'plate_id': np.random.randint(1, 6, n_samples),
        'protein_expression': trace.prior['protein_expression'].values.flatten()
    })

    return data

if __name__ == "__main__":
    # Generate sample data
    data = generate_sample_data(n_samples=100)

    # Save to CSV
    data.to_csv('protein_expression_data.csv', index=False)
    print(f"Generated {len(data)} samples")
    print(data.head())
```

**Script Features:**
- **Self-contained execution** with all dependencies specified
- **Configurable parameters** (sample size, random seed, etc.)
- **Realistic data generation** based on experimental design
- **CSV export** with proper experimental structure
- **Documentation** explaining the experimental design

### API Models for Execution

```python
class ModelExecutionRequest(BaseModel):
    session_id: str
    n_samples: int = 1000
    random_seed: Optional[int] = None
    convergence_criteria: Dict = Field(default_factory=lambda: {
        "ess_threshold": 400,
        "rhat_threshold": 1.1
    })

class PEP723ScriptRequest(BaseModel):
    session_id: str
    n_samples: int = 100
    random_seed: Optional[int] = None
    include_documentation: bool = True
    export_format: str = "csv"  # csv, excel, parquet

class ScriptExecutionResponse(BaseModel):
    script_content: str
    execution_id: Optional[str] = None
    download_url: Optional[str] = None
    preview_data: Optional[pd.DataFrame] = None
```

### Real-time Execution Monitoring

**Progress Updates:**
```html
<div class="execution-progress">
  <div class="progress-bar">
    <div class="progress-fill" style="width: 65%"></div>
  </div>
  <p>Sampling: 650/1000 draws (ESS: 423, R-hat: 1.05)</p>
  <button hx-delete="/htmx/sessions/123/model/stop">Stop Execution</button>
</div>
```

**Diagnostics Display:**
```html
<div class="model-diagnostics">
  <div class="diagnostic-card">
    <h4>Effective Sample Size</h4>
    <p class="ess-value">423</p>
    <span class="status good">✓ Good</span>
  </div>
  <div class="diagnostic-card">
    <h4>R-hat</h4>
    <p class="rhat-value">1.05</p>
    <span class="status good">✓ Converged</span>
  </div>
</div>
```

### Integration with Existing Pipeline

**Code Generation Integration:**
- **Seamless transition** from experiment design to model execution
- **Automatic validation** of generated models before execution
- **Error recovery** with suggestions for fixing model issues
- **Version tracking** of executed models and results

**Data Export Options:**
- **CSV format** for compatibility with analysis tools
- **Excel format** for easy sharing with collaborators
- **Parquet format** for large datasets
- **Python pickle** for preserving data types and metadata

### Storage Architecture

#### Document Store Approach (Recommended)

**Primary Choice: Document Database (MongoDB, CouchDB, DynamoDB, TinyDB)**

**Rationale:**
- **Natural Schema Fit**: Design sessions are inherently document-like with nested conversation histories, experiment versions, and artifacts
- **Schema Evolution**: AI-driven features will require rapid schema changes without migrations
- **Complex Nested Data**: Conversation history and experiment versions nest naturally without complex JOINs
- **AI/ML Workload Alignment**: Variable-length conversations and generated artifacts fit document model
- **Developer Experience**: Direct Pydantic model serialization with no ORM complexity

**Storage Structure:**
```json
{
  "session_id": "uuid",
  "title": "Protein Expression Study Design",
  "conversation_history": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "message_type": "user",
      "content": "I want to study protein expression..."
    }
  ],
  "experiment_versions": [
    {
      "version_id": "v1",
      "timestamp": "2024-01-15T10:35:00Z",
      "experiment": { /* ExperimentDescription object */ },
      "change_description": "Initial parsing"
    }
  ],
  "generated_artifacts": [
    {
      "artifact_type": "pymc_model",
      "content": "import pymc as pm...",
      "experiment_version_id": "v2"
    }
  ]
}
```

#### Backend Abstraction Strategy

**Consistent Storage Interface:**
- Abstract `SessionStore` interface providing uniform access across all database backends
- Pluggable storage drivers (MongoDB, CouchDB, DynamoDB, TinyDB, File-based for development)
- **Consistent API regardless of underlying storage technology** - the same interface works whether using a cloud database like DynamoDB or a local file-based solution like TinyDB
- Configuration-driven backend selection allowing easy switching between storage solutions
- All storage operations abstracted behind the same interface to ensure portability and testing flexibility

**Storage Operations:**
```python
# Abstract interface
class SessionStore:
    async def create_session(session: DesignSession) -> str
    async def get_session(session_id: str) -> DesignSession
    async def update_session(session: DesignSession) -> None
    async def list_sessions(scientist_id: str) -> List[SessionSummary]
    async def delete_session(session_id: str) -> None
    async def search_sessions(query: SearchQuery) -> List[SessionSummary]
```

#### Performance Considerations

**Document Store Optimizations:**

- Index on `scientist_id`, `created_at`, `last_modified` for user dashboards
- Text search indexes on conversation content for session discovery
- Separate collection for session metadata vs. full session content
- Implement session archival for completed/old sessions

**Caching Strategy:**

- Redis cache for active sessions
- Cache experiment versions separately for quick model regeneration
- Invalidation strategy for collaborative editing

### Proposed Schema Extensions

```python
class DesignSession(BaseModel):
    """Represents a complete experimental design session."""

    session_id: str = Field(..., description="Unique session identifier")
    title: Optional[str] = Field(None, description="User-provided session title")
    description: Optional[str] = Field(None, description="Session description or research question")

    created_at: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)
    scientist_id: Optional[str] = Field(None, description="User identifier")

    # Session content
    conversation_history: List[ConversationMessage] = Field(default_factory=list)
    experiment_versions: List[ExperimentVersion] = Field(default_factory=list)
    generated_artifacts: List[GeneratedArtifact] = Field(default_factory=list)
    notes: List[SessionNote] = Field(default_factory=list)

    # Current state
    current_experiment: Optional[ExperimentDescription] = Field(None)
    session_status: SessionStatus = Field(default=SessionStatus.ACTIVE)

class ConversationMessage(BaseModel):
    """Individual message in the design conversation."""

    timestamp: datetime = Field(default_factory=datetime.now)
    message_type: MessageType = Field(...)  # USER, AI_ASSISTANT, SYSTEM
    content: str = Field(...)
    experiment_state_id: Optional[str] = Field(None, description="Associated experiment version")

class ExperimentVersion(BaseModel):
    """Snapshot of experiment at a specific point in time."""

    version_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    experiment: ExperimentDescription = Field(...)
    change_description: Optional[str] = Field(None)
    validation_results: Optional[Dict] = Field(None)

class GeneratedArtifact(BaseModel):
    """Artifacts generated during the design session."""

    artifact_id: str = Field(...)
    artifact_type: ArtifactType = Field(...)  # PYMC_MODEL, SAMPLE_DATA, VISUALIZATION
    timestamp: datetime = Field(default_factory=datetime.now)
    content: str = Field(...)
    experiment_version_id: str = Field(...)
```

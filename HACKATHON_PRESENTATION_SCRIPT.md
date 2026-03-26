# Hackathon Presentation Script

## 30-Second Version

"This project is an AI PCB assistant built for KiCad. We split the system into a KiCad plugin and a local FastAPI backend. The plugin reads the live board and sends requests to the backend for circuit generation, placement optimization, and DFM analysis. On the backend side, we use a layered pipeline: LLM or SKiDL-based generation, schema validation and sanitization, heuristic placement, DFM checks, and finally KiCad schematic export. The key idea is that it runs locally, integrates directly with KiCad, and does more than text generation by actually moving toward manufacturable PCB outputs." 

## 90-Second Version

"Our project is a local AI assistant for PCB design inside KiCad. Architecturally, it has two parts. The first is a KiCad action plugin written in Python with a wx-based UI. The second is a local FastAPI backend that handles the heavier engineering tasks.

When the user interacts with the plugin, it extracts the current board state, including components, nets, positions, and board dimensions, then sends that data to the backend over localhost. For circuit generation, the backend does not rely on a single fragile AI call. It uses a staged pipeline. First, it can try SKiDL generation through an LLM. If that is unavailable or fails, it can use structured JSON generation or deterministic templates. After generation, the result is sanitized and validated with Pydantic models before anything is imported back into KiCad.

From there, the backend can auto-place components using force-directed placement or simulated annealing, then run DFM checks like spacing, boundaries, orientation, power integrity, and thermal sanity. It can also export a native `.kicad_sch` file. So the main value is not just AI text output. It is a local engineering workflow that connects generation, validation, placement, and KiCad-native output in one system."

## 3-5 Minute Full Script

"We built an AI PCB assistant for KiCad that runs as a local-first engineering tool rather than a cloud-only chatbot.

The system has a clear two-part architecture. On the front end, we have a KiCad plugin. This plugin is responsible for the user interface, extracting the active board state from KiCad, sending requests, and applying results back onto the board. On the backend, we have a FastAPI service that exposes endpoints for health, generation, placement, DFM checking, export, and template access.

The reason we split the architecture this way is separation of concerns. KiCad stays focused on CAD interaction, while the backend becomes a reusable local AI and analysis engine. That also makes the design easier to test and easier to extend later.

The most important technical flow is circuit generation. When the user enters a natural-language prompt like 'generate a 555 timer LED blinker', the backend runs a staged pipeline. If available, it first attempts SKiDL-based generation using the LLM. That means the model produces SKiDL code, which we then screen for unsafe patterns before executing in a constrained subprocess. If that succeeds, we parse the generated netlist back into structured circuit data. If that path fails, the backend can fall back to JSON generation or deterministic templates.

That fallback design is important because it turns the project from a single-shot demo into a more reliable system. We are not trusting raw model output blindly. After generation, the circuit goes through sanitization and validation. We normalize references, connections, and defaults, and then validate the structure using Pydantic models. We also apply topology guardrails for common cases like 555 timer wiring and regulator decoupling checks.

Once we have valid circuit data, the backend can auto-place components. Our MVP uses heuristic placement, primarily force-directed placement and simulated annealing. In simple terms, we convert electrical connectivity into a weighted graph and then try to place parts so connected components are closer together while still staying inside board boundaries. The architecture also leaves room for RL-based placement later, but for this MVP we focused on practical heuristics that work now.

After placement, we run DFM analysis. The DFM engine checks component spacing, edge clearance, orientation, power integrity, signal integrity, thermal concerns, floating parts, and net-length-related issues. So the system is not just generating a schematic idea. It is also applying engineering checks that matter for manufacturability.

Finally, the backend can export a native KiCad schematic file using a pure Python S-expression writer. That means the output is not only JSON for internal use. It can become a real `.kicad_sch` artifact for KiCad workflows.

On the plugin side, the last step is import. The plugin takes the generated result, creates or updates footprints, assigns nets to pads, rebuilds connectivity, and refreshes the live board. So the end-to-end story is: prompt to circuit data, circuit data to validation, validation to placement and DFM, and then results back into KiCad.

The biggest differentiator of this project is that it is local-first, KiCad-native, and engineering-aware. We are not just generating text. We are building a pipeline that turns AI output into structured PCB design actions."

## Demo Narration

Use this while clicking through the demo:

"I'm inside KiCad, and this plugin is our entry point. The plugin can read the live board state directly from KiCad, so we are not working from a disconnected mock environment.

Now I'll trigger generation with a prompt. The backend receives this over localhost, scores the request against known templates, and if the model is available it can attempt SKiDL or structured circuit generation. Once it gets a candidate result, it sanitizes and validates the circuit before returning it.

Here we can see the result summary, including the generation method, component count, net count, and any warnings. If generation succeeds, the plugin can import the returned components and connections back into the active board.

Next, I'll run placement optimization. This uses the backend placement engine to compute improved component positions based on net connectivity. Then I'll run DFM analysis, which checks spacing, boundaries, and other manufacturability-related constraints.

So the full workflow is local AI generation, validation, optimization, and KiCad integration, all in one loop."

## Safe Judge Answers

### If they ask, "What is the AI model doing here?"

"The model is used for structured circuit generation. In one path it generates SKiDL code, and in another it produces structured JSON. We then validate and sanitize that output before using it."

### If they ask, "How do you avoid hallucinations?"

"We do not trust raw LLM output directly. We use schema validation, sanitization, topology checks, and deterministic template fallback when needed."

### If they ask, "Is this cloud-based?"

"No, the intended workflow is local-first. The backend runs on localhost and can use local Ollama or a local GGUF model."

### If they ask, "How is placement done?"

"The implemented MVP uses heuristic optimization, mainly force-directed placement and simulated annealing. The architecture also leaves room for RL placement in later versions."

### If they ask, "What makes this more than a chatbot?"

"It is connected to KiCad, operates on structured board and circuit data, performs engineering validation, and can export KiCad-native schematic files."

### If they ask, "What is next after the hackathon?"

"The next steps are improving generation reliability, expanding templates, tightening board import/export, and extending the placement engine and validation stack."

## What Not To Overclaim

Do not say:

- "We have full production RL placement."
- "We have full autorouting in the current MVP."
- "We have complete simulation integrated end to end."

Safer version:

"The architecture is prepared for those capabilities, but the current MVP focuses on generation, placement, DFM, and KiCad integration."

## Closing Line

"The core achievement here is that we turned AI from a text assistant into a local PCB workflow engine that can generate, validate, optimize, and hand results back to KiCad in a usable form."

# Architecture diagrams (C4-PlantUML)

The C4 diagrams in [`../ARCHITECTURE.md`](../ARCHITECTURE.md) are authored as
**C4-PlantUML**. Each `.puml` is **self-contained** (sets `!NEW_C4_STYLE=1`,
includes the public C4-PlantUML stdlib, and declares its own person-outline and
`async`/`sync` relationship tags). The `.puml` files are the **source of truth**;
the rendered SVGs are produced automatically by CI - you do not hand-commit
image files.

| View | Source | Rendered (by CI) |
|------|--------|------------------|
| System Context (L1) | [`01-context.puml`](01-context.puml) | `01-context.svg` |
| Container (L2) | [`02-container.puml`](02-container.puml) | `02-container.svg` |
| Component - FM pipeline (L3) | [`03-component-fm.puml`](03-component-fm.puml) | `03-component-fm.svg` |
| Component - Advisory layer (L3) | [`04-component-advisory.puml`](04-component-advisory.puml) | `04-component-advisory.svg` |
| Deployment | [`05-deployment.puml`](05-deployment.puml) | `05-deployment.svg` |

## Conventions

- **New C4 style** (`!NEW_C4_STYLE=1`) with `SHOW_PERSON_OUTLINE()`.
- **Relationships:** `async` edges render **dashed** (message bus, closed-loop
  API), `sync` edges solid (file / blocking calls).
- **Placement:** `Rel_Neighbor` / `Rel_D` with short edge labels keep the layout
  uncrowded.
- Every diagram ends with `SHOW_LEGEND()`.

## How rendering works (GitHub Actions)

GitHub Markdown renders Mermaid natively **but not PlantUML**, so the diagrams
are compiled to SVG by a workflow and the generated images are committed back by
the CI bot:

1. Edit a `.puml` source and push.
2. [`.github/workflows/render-diagrams.yml`](../../.github/workflows/render-diagrams.yml)
   triggers on changes to `docs/diagrams/**.puml`.
3. It renders each `.puml` to SVG (PlantUML + Graphviz) and the
   `github-actions[bot]` commits the updated `*.svg` with `[skip ci]`.
4. `ARCHITECTURE.md` embeds the SVGs with `![](diagrams/01-context.svg)`.

> **Repo setting:** Settings -> Actions -> General -> *Workflow permissions* must
> be **Read and write** so the bot can commit the rendered images.

The diagrams use the **public** C4-PlantUML includes
(`!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/...`),
which the CI runner resolves over the network - no private include server.

## Rendering locally (optional)

```bash
# requires Java + Graphviz + the PlantUML jar (https://plantuml.com/download)
java -jar plantuml.jar -tsvg docs/diagrams/*.puml
# output is named from the @startuml id; CI renames it to match the source file.
```

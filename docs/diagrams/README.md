# Architecture diagrams (C4-PlantUML)

The C4 diagrams in [`../ARCHITECTURE.md`](../ARCHITECTURE.md) are authored as
**C4-PlantUML**. The `.puml` files are the **source of truth**; the rendered
SVGs are produced automatically by CI - you do not hand-commit image files.

| View | Source | Rendered (by CI) |
|------|--------|------------------|
| System Context (L1) | [`01-context.puml`](01-context.puml) | `01-context.svg` |
| Container (L2) | [`02-container.puml`](02-container.puml) | `02-container.svg` |
| Shared style/theme | [`style.puml`](style.puml) | - |

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

## Style

[`style.puml`](style.puml) is the shared theme included by every diagram: blue
systems, orange containers, grey external systems, flat (no shadow), orthogonal
routing.

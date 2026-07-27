#!/usr/bin/env node

/**
 * Deliver the canonical portable report with a narrowly scoped fix for the
 * packaged report header's 100vw/classic-scrollbar overflow on Windows.
 */

import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const scriptRoot = "C:/Users/lhall/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts";
const { deliverPortableArtifact } = await import(pathToFileURL(`${scriptRoot}/deliver_portable_artifact.mjs`));
const { buildPortableArtifact } = await import(pathToFileURL(`${scriptRoot}/build_portable_artifact.mjs`));

function buildWithoutViewportOverflow(input, options) {
  const html = buildPortableArtifact(input, options);
  const faulty = ".portable-page-header{position:sticky;top:0;z-index:60;display:flex;align-items:center;justify-content:space-between;width:100vw;height:48px;min-height:48px;margin-right:calc(50% - 50vw);margin-left:calc(50% - 50vw);";
  const fixed = ".portable-page-header{position:sticky;top:0;z-index:60;display:flex;align-items:center;justify-content:space-between;width:100%;height:48px;min-height:48px;margin-right:0;margin-left:0;";
  if (!html.includes(faulty)) {
    throw new Error("Expected packaged report-header CSS was not found; refusing an unreviewed rewrite.");
  }
  const patched = html.replaceAll(faulty, fixed);
  const guard = "<style id=\"classic-scrollbar-overflow-guard\">html,body,#data-analytics-portable-reader,#data-analytics-portable-reader-root,.dashboard-shell{max-width:100%;overflow-x:hidden}.analytics-top-bar{width:100%!important;margin-right:0!important;margin-left:0!important}</style>";
  if (!patched.includes("</head>")) {
    throw new Error("Expected portable report head was not found; refusing an unreviewed rewrite.");
  }
  return patched.replace("</head>", `${guard}</head>`);
}

const root = resolve("artifacts/analysis/esmc_layer51_synthesis_20260726");
try {
  const result = await deliverPortableArtifact(
    {
      inputPath: resolve(root, "artifact.json"),
      outputPath: resolve(root, "report.html"),
      readyTimeoutMs: 15_000,
      actionTimeoutMs: 5_000,
      timeoutMs: 30_000,
      screenshotPath: resolve(root, "report_failure.png"),
    },
    { build: buildWithoutViewportOverflow },
  );
  process.stdout.write(`${JSON.stringify(result)}\n`);
} catch (error) {
  process.stderr.write(`${JSON.stringify(error?.deliveryResult ?? { ok: false, error: error?.message ?? String(error) })}\n`);
  process.exitCode = 1;
}

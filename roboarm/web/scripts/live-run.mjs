import { access, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { chromium } from "playwright-core";

const baseUrl = process.env.ROBOARM_URL ?? "http://127.0.0.1:3000";
const chromePath =
  process.env.CHROME_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const artifactDirectory = path.resolve(
  process.cwd(),
  "../artifacts/browser/gkm-live-run",
);

await access(chromePath);
await mkdir(artifactDirectory, { recursive: true });

const browser = await chromium.launch({
  executablePath: chromePath,
  headless: true,
  args: ["--enable-webgl", "--ignore-gpu-blocklist"],
});

const pageErrors = [];
const consoleErrors = [];
const consoleWarnings = [];
const desktopViewport = { width: 1520, height: 1080 };
const narrowViewport = { width: 900, height: 1080 };
const page = await browser.newPage({
  viewport: desktopViewport,
  deviceScaleFactor: 1,
  colorScheme: "dark",
});
page.on("pageerror", (error) => pageErrors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
  if (message.type() === "warning") consoleWarnings.push(message.text());
});

async function screenshot(filename) {
  await page.waitForTimeout(300);
  await page.screenshot({
    path: path.join(artifactDirectory, filename),
    fullPage: true,
  });
}

try {
  await page.goto(baseUrl, { waitUntil: "networkidle", timeout: 30_000 });
  await page.getByTestId("rgb-camera-canvas").waitFor({ state: "visible" });
  await page.getByTestId("rgb-observation").waitFor({ state: "visible" });
  const failedTabs = page.locator(
    '[role="tab"][data-disposition="failed"]',
  );
  const promotedTabs = page.locator(
    '[role="tab"][data-disposition="promoted"]',
  );
  const failedCount = await failedTabs.count();
  const promotedCount = await promotedTabs.count();
  if (failedCount < 1 || promotedCount < 2) {
    throw new Error(
      `expected at least one failed and two successful replays; got ${failedCount}/${promotedCount}`,
    );
  }

  const screenshots = [];
  for (let index = 0; index < Math.min(3, failedCount); index += 1) {
    await failedTabs.nth(index).click();
    const initialName = `${String(screenshots.length + 1).padStart(2, "0")}-failure-${index + 1}-initial.png`;
    await screenshot(initialName);
    screenshots.push(initialName);
    await page.getByTestId("start-run").click();
    await page.locator('main[data-status="failed"]').waitFor({
      state: "visible",
      timeout: 60_000,
    });
    await page.getByTestId("failure-overlay").waitFor({ state: "visible" });
    const outcomeName = `${String(screenshots.length + 1).padStart(2, "0")}-failure-${index + 1}-outcome.png`;
    await screenshot(outcomeName);
    screenshots.push(outcomeName);
  }

  for (let index = 0; index < Math.min(2, promotedCount); index += 1) {
    await promotedTabs.nth(index).click();
    const initialName = `${String(screenshots.length + 1).padStart(2, "0")}-success-${index + 1}-initial.png`;
    await screenshot(initialName);
    screenshots.push(initialName);
    const actionCount = await page
      .getByTestId("action-timeline")
      .locator(".timeline-row")
      .count()
      .then((count) => count - 1);
    await page.getByTestId("start-run").click();
    if (index === promotedCount - 1 || index === 1) {
      await page.waitForFunction(
        (midpoint) =>
          Number(document.querySelector("main")?.getAttribute("data-turn")) >=
          midpoint,
        Math.max(1, Math.floor(actionCount / 2)),
        { timeout: 60_000 },
      );
      const midpointName = `${String(screenshots.length + 1).padStart(2, "0")}-success-${index + 1}-mid-task.png`;
      await screenshot(midpointName);
      screenshots.push(midpointName);
    }
    await page.locator('main[data-status="success"]').waitFor({
      state: "visible",
      timeout: 60_000,
    });
    await page.getByTestId("success-overlay").waitFor({ state: "visible" });
    const outcomeName = `${String(screenshots.length + 1).padStart(2, "0")}-success-${index + 1}-outcome.png`;
    await screenshot(outcomeName);
    screenshots.push(outcomeName);
  }

  await page.setViewportSize(narrowViewport);
  const narrowName = `${String(screenshots.length + 1).padStart(2, "0")}-success-2-outcome-narrow.png`;
  await screenshot(narrowName);
  screenshots.push(narrowName);

  const evidence = await page.evaluate(() => {
    const root = document.querySelector("main");
    const rgbCanvas = document.querySelector(
      '[data-testid="rgb-camera-canvas"]',
    );
    const paletteCanvas = document.querySelector(
      '[data-testid="rgb-observation"]',
    );
    const attemptTabs = Array.from(
      document.querySelectorAll('[role="tab"]'),
    ).map((tab) => ({
      text: tab.textContent?.replace(/\s+/g, " ").trim(),
      selected: tab.getAttribute("aria-selected") === "true",
      disposition: tab.getAttribute("data-disposition"),
      replayStage: tab.getAttribute("data-replay-stage"),
    }));
    let webgl = null;
    if (rgbCanvas instanceof HTMLCanvasElement) {
      const context =
        rgbCanvas.getContext("webgl2") ?? rgbCanvas.getContext("webgl");
      if (context !== null) {
        webgl = {
          version: context.getParameter(context.VERSION),
          renderer: context.getParameter(context.RENDERER),
          drawingBuffer: [
            context.drawingBufferWidth,
            context.drawingBufferHeight,
          ],
        };
      }
    }
    return {
      campaign: root?.getAttribute("data-campaign"),
      disposition: root?.getAttribute("data-disposition"),
      status: root?.getAttribute("data-status"),
      turn: Number(root?.getAttribute("data-turn")),
      success: root?.getAttribute("data-success") === "true",
      rgbCanvas:
        rgbCanvas instanceof HTMLCanvasElement
          ? [rgbCanvas.width, rgbCanvas.height]
          : null,
      paletteCanvas:
        paletteCanvas instanceof HTMLCanvasElement
          ? [paletteCanvas.width, paletteCanvas.height]
          : null,
      attemptTabs,
      failureReplayCount: attemptTabs.filter(
        (attempt) => attempt.disposition === "failed",
      ).length,
      successReplayCount: attemptTabs.filter(
        (attempt) => attempt.disposition === "promoted",
      ).length,
      webgl,
    };
  });

  const report = {
    schemaVersion: 2,
    evidenceKind: "gkm-campaign-replay",
    url: baseUrl,
    capturedAt: new Date().toISOString(),
    browser: await browser.version(),
    viewport: { ...desktopViewport, deviceScaleFactor: 1 },
    narrowViewport: { ...narrowViewport, deviceScaleFactor: 1 },
    screenshots,
    evidence,
    pageErrors,
    consoleErrors,
    consoleWarnings,
  };
  await writeFile(
    path.join(artifactDirectory, "live-run-evidence.json"),
    `${JSON.stringify(report, null, 2)}\n`,
    "utf8",
  );
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);

  if (
    evidence.campaign === null ||
    evidence.disposition !== "promoted" ||
    evidence.status !== "success" ||
    evidence.success !== true ||
    evidence.paletteCanvas?.[0] !== 128 ||
    evidence.paletteCanvas?.[1] !== 72 ||
    evidence.attemptTabs.length < 3 ||
    evidence.failureReplayCount < 1 ||
    evidence.successReplayCount < 2 ||
    evidence.webgl === null ||
    pageErrors.length > 0 ||
    consoleErrors.length > 0 ||
    consoleWarnings.length > 0
  ) {
    process.exitCode = 1;
  }
} catch (error) {
  const diagnostic = {
    error: error instanceof Error ? error.message : String(error),
    url: page.url(),
    body: await page.locator("body").innerText().catch(() => ""),
    pageErrors,
    consoleErrors,
    consoleWarnings,
  };
  await page
    .screenshot({
      path: path.join(artifactDirectory, "diagnostic-failure.png"),
      fullPage: true,
    })
    .catch(() => undefined);
  await writeFile(
    path.join(artifactDirectory, "diagnostic-failure.json"),
    `${JSON.stringify(diagnostic, null, 2)}\n`,
    "utf8",
  );
  process.stderr.write(`${JSON.stringify(diagnostic, null, 2)}\n`);
  throw error;
} finally {
  await browser.close();
}

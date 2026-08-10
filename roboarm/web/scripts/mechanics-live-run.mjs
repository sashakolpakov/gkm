import { access, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { chromium } from "playwright-core";

const baseUrl =
  process.env.ROBOARM_URL ?? "http://127.0.0.1:3000/mechanics-test";
const chromePath =
  process.env.CHROME_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const artifactDirectory = path.resolve(
  process.cwd(),
  "../artifacts/browser/mechanics-live-run",
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
const page = await browser.newPage({
  viewport: { width: 1520, height: 1080 },
  deviceScaleFactor: 1,
  colorScheme: "dark",
});
page.on("pageerror", (error) => pageErrors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
  if (message.type() === "warning") consoleWarnings.push(message.text());
});

async function screenshot(filename) {
  await page.waitForTimeout(350);
  await page.screenshot({
    path: path.join(artifactDirectory, filename),
    fullPage: true,
  });
}

try {
  await page.goto(baseUrl, { waitUntil: "networkidle", timeout: 30_000 });
  await page.getByTestId("rgb-camera-canvas").waitFor({ state: "visible" });
  await page.getByTestId("rgb-observation").waitFor({ state: "visible" });
  const rejectedAttempt = page.locator(
    '[role="tab"][data-disposition="expected-rejection"]',
  );
  const completedAttempt = page.locator(
    '[role="tab"][data-disposition="completed"]',
  );
  await rejectedAttempt.waitFor({ state: "visible" });
  await completedAttempt.waitFor({ state: "visible" });
  await page
    .getByText("NOT MACHINE LEARNING EVIDENCE", { exact: true })
    .waitFor();
  await screenshot("01-desktop-collision-initial.png");

  await rejectedAttempt.click();
  await page.getByTestId("start-run").click();
  await page.locator('main[data-status="failed"]').waitFor({
    state: "visible",
    timeout: 30_000,
  });
  await page.getByTestId("failure-overlay").waitFor({ state: "visible" });
  await screenshot("02-desktop-collision-rejected.png");

  await completedAttempt.click();
  await screenshot("03-desktop-completion-initial.png");
  const completedActionCount = await page
    .getByTestId("action-timeline")
    .locator(".timeline-row")
    .count()
    .then((count) => count - 1);
  await page.getByTestId("start-run").click();
  await page.waitForFunction(
    (midpoint) =>
      Number(document.querySelector("main")?.getAttribute("data-turn")) >=
      midpoint,
    Math.max(1, Math.floor(completedActionCount / 2)),
    { timeout: 30_000 },
  );
  await screenshot("04-desktop-completion-mid-task.png");
  await page.locator('main[data-status="success"]').waitFor({
    state: "visible",
    timeout: 30_000,
  });
  await page.getByTestId("success-overlay").waitFor({ state: "visible" });
  await screenshot("05-desktop-completion-success.png");

  const cameraBounds = await page
    .getByTestId("rgb-camera-canvas")
    .boundingBox();
  if (cameraBounds === null) {
    throw new Error("RGB camera canvas has no interactive bounds");
  }
  const orbitStart = {
    x: cameraBounds.x + cameraBounds.width * 0.55,
    y: cameraBounds.y + cameraBounds.height * 0.55,
  };
  await page.mouse.move(orbitStart.x, orbitStart.y);
  await page.mouse.down();
  await page.mouse.move(orbitStart.x + 210, orbitStart.y, { steps: 18 });
  await page.mouse.up();
  await page.waitForTimeout(450);
  await screenshot("06-desktop-orbit-success.png");

  await page.setViewportSize({ width: 430, height: 932 });
  await screenshot("07-narrow-completion-success.png");

  const evidence = await page.evaluate(() => {
    const root = document.querySelector("main");
    const rgbCanvas = document.querySelector(
      '[data-testid="rgb-camera-canvas"]',
    );
    const paletteCanvas = document.querySelector(
      '[data-testid="rgb-observation"]',
    );
    const warning = Array.from(document.querySelectorAll("*")).some(
      (element) =>
        element.textContent?.trim() === "NOT MACHINE LEARNING EVIDENCE",
    );
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
      fixtureSet: root?.getAttribute("data-campaign"),
      evidenceKind: root?.getAttribute("data-evidence-kind"),
      disposition: root?.getAttribute("data-disposition"),
      status: root?.getAttribute("data-status"),
      turn: Number(root?.getAttribute("data-turn")),
      success: root?.getAttribute("data-success") === "true",
      testOnlyWarningVisible: warning,
      rgbCanvas:
        rgbCanvas instanceof HTMLCanvasElement
          ? [rgbCanvas.width, rgbCanvas.height]
          : null,
      paletteCanvas:
        paletteCanvas instanceof HTMLCanvasElement
          ? [paletteCanvas.width, paletteCanvas.height]
          : null,
      webgl,
    };
  });

  const report = {
    schemaVersion: 1,
    evidenceKind: "developer-mechanics-test",
    scientificClaim: "none; scripted regression only",
    url: baseUrl,
    capturedAt: new Date().toISOString(),
    browser: await browser.version(),
    viewport: {
      desktop: { width: 1520, height: 1080, deviceScaleFactor: 1 },
      narrow: { width: 430, height: 932, deviceScaleFactor: 1 },
    },
    screenshots: [
      "01-desktop-collision-initial.png",
      "02-desktop-collision-rejected.png",
      "03-desktop-completion-initial.png",
      "04-desktop-completion-mid-task.png",
      "05-desktop-completion-success.png",
      "06-desktop-orbit-success.png",
      "07-narrow-completion-success.png",
    ],
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
    evidence.fixtureSet !== "canonical-mechanics-fixture" ||
    evidence.evidenceKind !== "developer-mechanics-test" ||
    evidence.disposition !== "completed" ||
    evidence.status !== "success" ||
    evidence.success !== true ||
    evidence.testOnlyWarningVisible !== true ||
    evidence.paletteCanvas?.[0] !== 128 ||
    evidence.paletteCanvas?.[1] !== 72 ||
    evidence.webgl === null ||
    pageErrors.length > 0 ||
    consoleErrors.length > 0 ||
    consoleWarnings.length > 0
  ) {
    process.exitCode = 1;
  }
} finally {
  await browser.close();
}

import { Show, createEffect, createSignal, onSettled } from "solid-js";
import { Badge, Button } from "../ui";
import {
  type DownloadProgress,
  MODEL,
  chooseLocalFolder,
  clearModelCache,
  forgetLocalFolder,
  fsAccessSupported,
  grantPendingFolder,
  isCached,
  loadFromHuggingFace,
  loadFromLocal,
  loadedModel,
  localFolder,
  pendingFolder,
  resumeLocalFolder,
  unloadModel,
} from "../../lib/phonon";

function fmtBytes(n: number): string {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Loads Phonon-2 either from Hugging Face (cached in the browser afterwards) or from a folder on disk. */
export function ModelPicker() {
  const [busy, setBusy] = createSignal(false);
  const [phase, setPhase] = createSignal("");
  const [progress, setProgress] = createSignal<DownloadProgress | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [cached, setCached] = createSignal(false);

  onSettled(() => {
    void isCached().then(setCached);
    resumeLocalFolder();
  });

  async function run(task: () => Promise<void>) {
    setBusy(true);
    setError(null);
    setProgress(null);
    try {
      await task();
      setCached(await isCached());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setPhase("");
      setProgress(null);
    }
  }

  const fromHf = () => run(() => loadFromHuggingFace(setProgress, setPhase));
  const fromLocal = () => {
    const folder = localFolder();
    if (folder) void run(() => loadFromLocal(folder, setPhase));
  };

  async function pickFolder() {
    setError(null);
    try {
      const folder = await chooseLocalFolder();
      if (folder && !folder.files) setError(`Phonon-2 was not found in ${folder.name}/.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function grant() {
    setError(null);
    try {
      if (!(await grantPendingFolder())) setError("Access wasn't granted.");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  // A folder that was remembered and still readable is the cheaper source: use it without being asked.
  let autoLoaded = false;
  createEffect(
    () => localFolder(),
    (folder) => {
      if (folder?.files && !autoLoaded && !loadedModel() && !busy()) {
        autoLoaded = true;
        void run(() => loadFromLocal(folder, setPhase));
      }
    },
  );

  const pct = () => {
    const p = progress();
    return p?.total ? Math.round((p.loaded / p.total) * 100) : null;
  };

  return (
    <div class="rounded-xl border border-line bg-surface p-4">
      <div class="flex flex-wrap items-center gap-2">
        <p class="text-sm font-medium text-ink">Model</p>
        <Show when={loadedModel()} fallback={<Badge>Not loaded</Badge>}>
          {(m) => (
            <Badge tone="accent">
              Loaded from {m().source === "hf" ? "Hugging Face" : "local folder"} in {(m().ms / 1000).toFixed(1)} s
            </Badge>
          )}
        </Show>
        <Show when={loadedModel()}>
          <button
            type="button"
            onClick={unloadModel}
            class="ml-auto text-xs text-muted underline decoration-dotted hover:text-ink"
          >
            unload
          </button>
        </Show>
      </div>

      <div class="mt-4 grid gap-4 md:grid-cols-2">
        {/* Hugging Face */}
        <div class="md:border-r md:border-line md:pr-4">
          <p class="text-sm font-medium text-ink">Hugging Face</p>
          <p class="mt-2 text-sm text-muted">
            {MODEL.label}: the five-value quantised Parakeet-TDT-0.6B-v3, ~{MODEL.approxSizeMb} MB download.
            {cached() ? " Already cached in this browser." : ""}
          </p>
          <p class="mt-1 font-mono text-xs text-faint">
            {MODEL.hfRepo}/{MODEL.file}
          </p>
          <div class="mt-4 flex items-center gap-3">
            <Button variant="primary" disabled={busy()} onClick={fromHf}>
              {busy() ? "Loading…" : loadedModel()?.source === "hf" ? "Reload" : cached() ? "Load" : "Download & load"}
            </Button>
            <Show when={cached()}>
              <button
                type="button"
                disabled={busy()}
                onClick={async () => {
                  await clearModelCache();
                  setCached(false);
                }}
                class="text-xs text-muted underline decoration-dotted hover:text-ink disabled:opacity-40"
              >
                clear downloaded cache
              </button>
            </Show>
          </div>
        </div>

        {/* Local */}
        <div>
          <p class="text-sm font-medium text-ink">Local folder</p>
          <Show
            when={fsAccessSupported()}
            fallback={
              <p class="mt-2 text-sm text-danger">
                This browser doesn't support the File System Access API needed to read a local folder. Try Chrome
                or Edge.
              </p>
            }
          >
            <Show
              when={localFolder()}
              fallback={
                <div class="mt-2">
                  <Show
                    when={pendingFolder()}
                    fallback={
                      <p class="text-sm text-muted">
                        Pick a folder holding <code class="font-mono text-xs">{MODEL.file}</code>, or the unpacked{" "}
                        <code class="font-mono text-xs">{MODEL.unpackDir}/</code> (
                        <code class="font-mono text-xs">model.fermion</code> +{" "}
                        <code class="font-mono text-xs">config.json</code>), or a checkout of the Hugging Face repo
                        that contains either.
                      </p>
                    }
                  >
                    {(handle) => (
                      <>
                        <p class="mb-3 rounded-lg border border-accent-line bg-accent-soft px-3 py-2 text-sm text-ink">
                          Resuming <span class="font-mono text-xs">{handle().name}/</span> from last time: grant access to
                          reload it without picking it again.
                        </p>
                        <Button variant="primary" disabled={busy()} onClick={grant}>
                          Grant access
                        </Button>
                      </>
                    )}
                  </Show>
                  <Button class="mt-3" variant={pendingFolder() ? "secondary" : "primary"} disabled={busy()} onClick={pickFolder}>
                    {pendingFolder() ? "Choose a different folder…" : "Choose folder…"}
                  </Button>
                </div>
              }
            >
              {(folder) => (
                <div class="mt-2">
                  <p class="font-mono text-xs text-faint">{folder().name}/</p>
                  <p class="mt-2 text-sm text-muted">
                    <Show when={folder().files} fallback={<span class="text-danger">Phonon-2 not found in this folder.</span>}>
                      {(files) => <>Found <span class="font-mono text-xs">{files().where}</span>.</>}
                    </Show>
                  </p>
                  <div class="mt-4 flex items-center gap-3">
                    <Button variant="primary" disabled={busy() || !folder().files} onClick={fromLocal}>
                      {busy() ? "Loading…" : loadedModel()?.source === "local" ? "Reload" : "Load"}
                    </Button>
                    <button
                      type="button"
                      disabled={busy()}
                      onClick={pickFolder}
                      class="text-xs text-muted underline decoration-dotted hover:text-ink disabled:opacity-40"
                    >
                      change folder
                    </button>
                    <button
                      type="button"
                      disabled={busy()}
                      onClick={() => void forgetLocalFolder()}
                      class="text-xs text-muted underline decoration-dotted hover:text-ink disabled:opacity-40"
                    >
                      forget
                    </button>
                  </div>
                </div>
              )}
            </Show>
          </Show>
        </div>
      </div>

      <Show when={busy()}>
        <div class="mt-4">
          <div class="flex justify-between text-xs text-faint">
            <span>{phase()}…</span>
            <Show when={progress()}>{(p) => <span>{fmtBytes(p().loaded)}{p().total ? ` / ${fmtBytes(p().total!)}` : ""}</span>}</Show>
          </div>
          <div class="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
            <div
              class={`h-full rounded-full bg-accent transition-all duration-150 ${pct() === null ? "animate-pulse" : ""}`}
              style={{ width: `${pct() ?? 100}%` }}
            />
          </div>
        </div>
      </Show>

      <Show when={error()}>
        <p class="mt-3 text-sm text-danger">{error()}</p>
      </Show>

      <p class="mt-4 text-xs leading-relaxed text-faint">
        The weights expand to f32 in memory, so the tab needs about 3 GB free: use a desktop Chrome, Edge or
        Firefox. Everything runs on one CPU thread inside a Web Worker; nothing is uploaded.
      </p>
    </div>
  );
}

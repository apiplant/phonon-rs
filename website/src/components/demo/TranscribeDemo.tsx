import { For, Show, createSignal, onCleanup } from "solid-js";
import { DemoLayout } from "./DemoLayout";
import { ModelPicker } from "./ModelPicker";
import { Badge, Button } from "../ui";
import { type Audio, type Transcript, decodeAudio, loadedModel, transcribe } from "../../lib/phonon";

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

/** Opacity from the word's confidence: sure words are solid, shaky ones fade. */
function confidenceStyle(c: number): string {
  const t = Math.max(0, Math.min(1, (c - 0.5) / 0.5));
  return `opacity:${(0.45 + 0.55 * t).toFixed(2)}`;
}

export function TranscribeDemo() {
  const [audio, setAudio] = createSignal<Audio | null>(null);
  const [audioName, setAudioName] = createSignal("");
  const [audioUrl, setAudioUrl] = createSignal<string | null>(null);
  const [dragging, setDragging] = createSignal(false);
  const [recording, setRecording] = createSignal(false);
  const [running, setRunning] = createSignal(false);
  const [progress, setProgress] = createSignal<[number, number] | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [result, setResult] = createSignal<{ transcript: Transcript; ms: number; seconds: number } | null>(null);
  const [copied, setCopied] = createSignal(false);

  let recorder: MediaRecorder | undefined;
  let stream: MediaStream | undefined;

  onCleanup(() => {
    stream?.getTracks().forEach((t) => t.stop());
    const url = audioUrl();
    if (url) URL.revokeObjectURL(url);
  });

  async function useBlob(blob: Blob, name: string) {
    setError(null);
    setResult(null);
    try {
      const decoded = await decodeAudio(await blob.arrayBuffer());
      const old = audioUrl();
      if (old) URL.revokeObjectURL(old);
      setAudioUrl(URL.createObjectURL(blob));
      setAudio(decoded);
      setAudioName(name);
    } catch (e) {
      setAudio(null);
      setError(`Couldn't decode ${name}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  function onFiles(files: FileList | null | undefined) {
    const file = files?.[0];
    if (file) void useBlob(file, file.name);
  }

  async function toggleRecording() {
    if (recording()) {
      recorder?.stop();
      return;
    }
    setError(null);
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      const chunks: Blob[] = [];
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = () => {
        stream?.getTracks().forEach((t) => t.stop());
        setRecording(false);
        void useBlob(new Blob(chunks, { type: recorder?.mimeType }), "recording");
      };
      recorder.start();
      setRecording(true);
    } catch (e) {
      setError(`Microphone unavailable: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function run() {
    const a = audio();
    if (!a || !loadedModel()) return;
    setRunning(true);
    setError(null);
    setResult(null);
    setProgress([0, Math.max(1, Math.ceil(a.duration / 30))]);
    try {
      const { transcript, ms } = await transcribe(a, (done, total) => setProgress([done, total]));
      setResult({ transcript, ms, seconds: a.duration });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
      setProgress(null);
    }
  }

  async function copy() {
    const r = result();
    if (!r) return;
    try {
      await navigator.clipboard.writeText(r.transcript.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard denied: the text is selectable */
    }
  }

  return (
    <DemoLayout
      title="Transcribe in your browser"
      description="Drop in a recording or speak into your microphone. Phonon-2 runs here, as WebAssembly in a Web Worker, so the audio never leaves your machine."
    >
      <ModelPicker />

      <div
        class={`mt-6 rounded-xl border-2 border-dashed p-6 text-center transition-colors ${
          dragging() ? "border-accent bg-accent-soft" : "border-line bg-surface"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          onFiles(e.dataTransfer?.files);
        }}
      >
        <p class="text-sm text-muted">Drop an audio or video file here, or</p>
        <div class="mt-3 flex flex-wrap items-center justify-center gap-3">
          <label class="inline-flex cursor-pointer items-center rounded-lg border border-line bg-surface-2 px-3.5 py-2 text-sm text-ink transition-colors hover:border-line-strong hover:bg-surface-3">
            Choose a file…
            <input type="file" accept="audio/*,video/*" class="sr-only" onChange={(e) => onFiles(e.currentTarget.files)} />
          </label>
          <Button variant={recording() ? "primary" : "secondary"} onClick={toggleRecording} disabled={running()}>
            {recording() ? (
              <>
                <span class="h-2 w-2 animate-pulse rounded-full bg-danger" /> Stop recording
              </>
            ) : (
              "● Record from microphone"
            )}
          </Button>
        </div>
        <p class="mt-3 text-xs text-faint">wav, mp3, flac, ogg, m4a, webm: anything your browser can decode. English.</p>
      </div>

      <Show when={audio()}>
        {(a) => (
          <div class="mt-4 rounded-xl border border-line bg-surface p-4">
            <div class="flex flex-wrap items-center gap-2">
              <p class="min-w-0 truncate text-sm font-medium text-ink">{audioName()}</p>
              <Badge>{fmtTime(a().duration)}</Badge>
            </div>
            <Show when={audioUrl()}>{(url) => <audio class="mt-3 w-full" controls src={url()} />}</Show>
            <div class="mt-4 flex flex-wrap items-center gap-3">
              <Button variant="primary" disabled={!loadedModel() || running()} onClick={run}>
                {running() ? "Transcribing…" : "Transcribe"}
              </Button>
              <Show when={!loadedModel()}>
                <span class="text-xs text-faint">Load the model first.</span>
              </Show>
            </div>
            <Show when={progress()}>
              {(p) => (
                <div class="mt-4">
                  <div class="flex justify-between text-xs text-faint">
                    <span>
                      chunk {Math.min(p()[0] + 1, p()[1])} of {p()[1]}
                    </span>
                    <span>single-threaded: expect a few seconds per second of audio</span>
                  </div>
                  <div class="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
                    <div class="h-full rounded-full bg-accent transition-all duration-300" style={{ width: `${(p()[0] / p()[1]) * 100}%` }} />
                  </div>
                </div>
              )}
            </Show>
          </div>
        )}
      </Show>

      <Show when={error()}>
        <p class="mt-4 text-sm text-danger">{error()}</p>
      </Show>

      <Show when={result()}>
        {(r) => (
          <div class="mt-6 rounded-xl border border-line bg-surface p-5">
            <div class="flex flex-wrap items-center gap-2">
              <h2 class="text-base font-semibold text-ink">Transcript</h2>
              <Badge tone="accent">
                {r().seconds.toFixed(1)} s of audio in {(r().ms / 1000).toFixed(1)} s ({(r().seconds / (r().ms / 1000)).toFixed(1)}x real time)
              </Badge>
              <Badge>confidence {(r().transcript.confidence * 100).toFixed(1)}%</Badge>
            </div>

            <Show when={r().transcript.words.length > 0} fallback={<p class="mt-4 text-sm text-muted">No speech detected.</p>}>
              <p class="mt-4 text-lg leading-relaxed text-ink">
                <For each={r().transcript.words}>
                  {(w) => (
                    <span
                      title={`${fmtTime(w.start)} – ${fmtTime(w.end)} · ${(w.confidence * 100).toFixed(1)}%`}
                      style={confidenceStyle(w.confidence)}
                      class="mr-1 inline-block rounded px-0.5 hover:bg-accent-soft"
                    >
                      {w.word}
                    </span>
                  )}
                </For>
              </p>
              <p class="mt-2 text-xs text-faint">Hover a word for its time and confidence; shaky words are dimmer.</p>
            </Show>

            <div class="mt-4 flex flex-wrap gap-2">
              <Button size="sm" onClick={copy}>
                {copied() ? "Copied" : "Copy text"}
              </Button>
              <Button size="sm" onClick={() => download("transcript.srt", r().transcript.srt, "application/x-subrip")}>
                Download SRT
              </Button>
              <Button size="sm" onClick={() => download("transcript.json", JSON.stringify(r().transcript, null, 2), "application/json")}>
                Download JSON
              </Button>
            </div>
          </div>
        )}
      </Show>
    </DemoLayout>
  );
}

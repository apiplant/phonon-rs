import { type ParentProps } from "solid-js";

export function DemoLayout(props: ParentProps<{ title: string; description: string }>) {
  return (
    <div class="mx-auto w-full max-w-5xl px-5 py-10">
      <p class="font-mono text-xs text-accent">100% client-side · runs in your browser via WebAssembly</p>
      <h1 class="mt-2 text-3xl font-semibold tracking-tight text-ink">{props.title}</h1>
      <p class="mt-2 max-w-2xl leading-relaxed text-muted">{props.description}</p>
      <div class="mt-8">{props.children}</div>
    </div>
  );
}

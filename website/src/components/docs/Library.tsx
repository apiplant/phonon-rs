import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsLibrary() {
  return (
    <DocsLayout>
      <H1>As a library</H1>
      <Lead>
        The <IC>phonon-rs</IC> crate (library name <IC>phonon</IC>) is the engine behind both binaries: model loading, the mel front end,
        batched greedy TDT decoding, and word timestamps.
      </Lead>

      <Section>
        <H2>Add the dependency</H2>
        <P>
          Everything beyond inference is behind features: <IC>mic</IC> (cpal; <IC>phonon::mic</IC>),{" "}
          <IC>cli</IC> (the <IC>phonon</IC> binary) and <IC>dictate</IC> (the <IC>phonon-dictate</IC> binary,
          Linux only). <IC>dictate</IC> implies <IC>cli</IC>, which implies <IC>mic</IC>; the defaults enable{" "}
          <IC>dictate</IC>. For inference only:
        </P>
        <CopyBlock command={`cargo add phonon-rs --no-default-features
cargo add phonon-rs --no-default-features --features cuda   # CUDA is opt-in and passed down to candle`} />
      </Section>

      <Section>
        <H2>Transcribe a file</H2>
        <Pre caption="src/main.rs" lang="rust">{`use phonon::engine::{DeviceArg, Engine, Precision};
use phonon::{audio, text};

// None = default precision: f32 on CPU, f16 on CUDA.
let engine = Engine::load(Some("phonon-2.bps.tar.zst".into()), DeviceArg::Auto, None)?;

let samples = audio::load("talk.mp3".as_ref())?;   // any symphonia format -> 16 kHz mono
let words = engine.transcribe(&samples, 30.0)?;    // chunks of up to 30 s, cut at pauses

println!("{}", text::join(&words));
println!("{}", text::srt(&words));
println!("confidence {:.3}", text::confidence(&words));`}</Pre>
        <P>
          Each <IC>Word</IC> has <IC>word</IC>, <IC>start</IC>, <IC>end</IC> (seconds) and{" "}
          <IC>confidence</IC> (the geometric mean of its tokens' probabilities).
        </P>
      </Section>

      <Section>
        <H2>Batches and threads</H2>
        <Pre caption="src/main.rs" lang="rust">{`let chunks = audio::chunk(0, &samples, 30.0);
let words_per_chunk = engine.run(&chunks, 16)?;    // batches of 16, in input order

phonon::engine::init_threads(Some(8));              // rayon + candle pools; default: physical cores`}</Pre>
      </Section>

      <Section>
        <H2>In the browser</H2>
        <P>
          Built for <IC>wasm32-unknown-unknown</IC>, the crate exposes <IC>WasmPhonon</IC>:{" "}
          <IC>loadArchive(bytes)</IC> or <IC>loadModel(container, configJson)</IC>, then{" "}
          <IC>transcribe(samples, sampleRate, chunkSecs, onProgress)</IC>, which returns JSON with the text, the
          words, SRT cues and the confidence. The browser decodes audio itself (Web Audio), so symphonia is not
          part of that build.
        </P>
        <CopyBlock command={`cargo install wasm-pack
wasm-pack build --target web --release --out-dir website/src/wasm-pkg -- --no-default-features`} />
      </Section>
    </DocsLayout>
  );
}

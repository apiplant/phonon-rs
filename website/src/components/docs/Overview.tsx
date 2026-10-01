import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Section, FlagTable } from "./Prose";
import { LinkButton, Mono } from "../ui";
import { CopyBlock } from "../Code";

export function DocsOverview() {
  return (
    <DocsLayout>
      <H1>Documentation</H1>
      <Lead>
        phonon-rs is a Rust (<Mono>candle</Mono>) port of the <Mono>ParakeetForTDT</Mono> graph, built to
        run{" "}
        <a
          href="https://huggingface.co/FermionResearch/Phonon-2"
          target="_blank"
          rel="noreferrer noopener"
          class="text-accent hover:text-accent-dim"
        >
          Phonon-2
        </a>
        , the five-value quantised Parakeet-TDT-0.6B-v3. It ships as a library crate, a transcriber, a Linux
        dictation daemon, and a WebAssembly build that powers the{" "}
        <a href="/demo" class="text-accent hover:text-accent-dim">
          in-browser demo
        </a>
        .
      </Lead>

      <Section>
        <H2>The model</H2>
        <P>
          Phonon-2 is a 164 MB download (177 MB unpacked). Its encoder holds each weight at one of five learned
          levels in about 2.1 bits; the runtime expands that container to f32 when it loads. English only. The
          weights are published by FermionResearch under CC-BY-4.0.
        </P>
        <P>
          The runtime reads <IC>model.fermion</IC> directly: the unpacked directory, the file itself, or the
          release archive <IC>phonon-2.bps.tar.zst</IC>.
        </P>
        <CopyBlock command={`# unpack next to the project, or just pass the .tar.zst to --model
huggingface-cli download FermionResearch/Phonon-2 phonon-2.bps.tar.zst --local-dir .
mkdir -p model_phonon2_c4c_int6 && tar --zstd -xf phonon-2.bps.tar.zst -C model_phonon2_c4c_int6`} />
      </Section>

      <Section>
        <H2>The binaries</H2>
        <UL>
          <LI>
            <a href="/docs/cli" class="text-accent hover:text-accent-dim">
              <IC>phonon</IC>
            </a>{" "}
            transcribes audio files or the microphone to text, JSON or SRT.
          </LI>
          <LI>
            <a href="/docs/dictate" class="text-accent hover:text-accent-dim">
              <IC>phonon-dictate</IC>
            </a>{" "}
            is a push-to-talk dictation daemon for Linux.
          </LI>
        </UL>
      </Section>

      <Section>
        <H2>Model resolution</H2>
        <P>Both binaries look for the model in this order:</P>
        <UL>
          <LI>
            <IC>--model PATH</IC>
          </LI>
          <LI>
            the <IC>PHONON_MODEL</IC> environment variable
          </LI>
          <LI>
            <IC>model_phonon2_c4c_int6</IC> or <IC>phonon-2.bps.tar.zst</IC> in the working directory, its
            parent, or next to the binary
          </LI>
          <LI>
            otherwise the archive is downloaded from Hugging Face on first use (164 MB) into{" "}
            <IC>$XDG_CACHE_HOME/phonon-rs/Phonon-2/</IC> (<IC>~/.cache/phonon-rs/Phonon-2/</IC> by default) and
            reused afterwards. It is fetched to a <IC>.part</IC> file and renamed when complete, so an interrupted
            download never leaves an archive that looks done.
          </LI>
        </UL>
      </Section>

      <Section>
        <H2>Speed</H2>
        <P>One hour of LibriSpeech cut into 30 s chunks, batches of 16, on a Ryzen 9 7950X and an RTX 4090.</P>
        <FlagTable
          rows={[
            { flag: "phonon, CUDA f16", meaning: "2.8 s for the hour, 1,304x real time" },
            { flag: "PyTorch, CUDA f16", meaning: "2.6 s, 1,372x" },
            { flag: "phonon, CPU f32, 16 threads", meaning: "75 s, 48x" },
            { flag: "PyTorch, CPU f32, 16 threads", meaning: "59 s, 61x" },
          ]}
        />
        <P>
          Startup is 0.3 to 0.7 s against about 30 s for the Python reference. On the 73 test utterances, candle
          produces text identical to PyTorch fp32 on CPU f32, CUDA f32 and CUDA f16, for a 4.26 % WER each.
        </P>
      </Section>

      <Section>
        <H2>Browser build</H2>
        <P>
          The same engine compiles to WebAssembly (single-threaded, f32). The demo downloads{" "}
          <IC>phonon-2.bps.tar.zst</IC> from Hugging Face, or reads it from a local folder, and runs in a Web
          Worker. It needs a desktop browser and about 3 GB of free memory, because the weights expand to f32.
        </P>
        <div class="mt-4">
          <LinkButton href="/demo" variant="primary">
            Open the demo
          </LinkButton>
        </div>
      </Section>
    </DocsLayout>
  );
}

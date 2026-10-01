import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsCli() {
  return (
    <DocsLayout>
      <H1>phonon</H1>
      <Lead>
        Transcribe audio files or the microphone. Files can be in any format symphonia decodes (wav, flac,
        mp3, ogg/vorbis, aac/m4a); they are mixed to mono and resampled to 16 kHz.
      </Lead>

      <Section>
        <H2>Examples</H2>
        <CopyBlock
          command={`phonon recording.wav                         # CUDA if built with it and a GPU is present, else CPU
phonon talk.mp3 --device cpu --format srt > talk.srt
phonon a.flac b.m4a --format json            # words with start/end times
phonon hour.flac --bench                     # timings + real-time factor on stderr

phonon --mic                                 # live from the default input; Ctrl-C to stop
phonon --list-mics
phonon --mic --mic-device wave --format json # one JSON line per utterance`}
        />
      </Section>

      <Section>
        <H2>Options</H2>
        <FlagTable
          rows={[
            { flag: "-m, --model PATH", meaning: <>The model: a directory with <IC>model.fermion</IC> + <IC>config.json</IC>, the file, or <IC>phonon-2.bps.tar.zst</IC>. Also <IC>PHONON_MODEL</IC>.</> },
            { flag: "-d, --device auto|cpu|cuda", meaning: "Where to run. auto picks CUDA when the build has it and a GPU is present." },
            { flag: "--dtype f32|f16|bf16", meaning: "Encoder precision. Defaults to f32 on CPU and f16 on CUDA. The prediction network and joint always run in f32." },
            { flag: "--format text|json|srt", meaning: "Output format. json carries words with start/end times and confidences." },
            { flag: "--chunk-secs 30", meaning: "Long audio is cut at the quietest 20 ms inside the last quarter of each window. In --mic mode this is also the longest single utterance." },
            { flag: "--batch-size 16", meaning: "Chunks per encoder batch." },
            { flag: "--threads N", meaning: "Defaults to the physical core count, because SMT siblings slow the GEMMs down." },
            { flag: "--bench", meaning: "Print timings and the real-time factor on stderr." },
          ]}
        />
      </Section>

      <Section>
        <H2>Microphone mode</H2>
        <P>
          <IC>--mic</IC> captures the chosen input and splits it into utterances with an energy VAD. The
          speech threshold is 4x a running noise-floor estimate, or <IC>--vad-threshold</IC> if you set one.
        </P>
        <UL>
          <LI>While you speak, the current utterance is re-transcribed about every half second and shown dimmed on stderr. On a slow CPU the refresh backs off.</LI>
          <LI>After a pause of <IC>--silence-ms</IC> (700 ms by default) the final text goes to stdout: plain text, one JSON line per utterance, or SRT cues.</LI>
          <LI>Ctrl-C flushes the utterance in progress and exits.</LI>
        </UL>
        <FlagTable
          rows={[
            { flag: "--mic", meaning: "Transcribe the microphone live." },
            { flag: "--mic-device NAME", meaning: "Any part of a name from --list-mics. Defaults to the system default input." },
            { flag: "--list-mics", meaning: "List audio input devices (* marks the default) and exit." },
            { flag: "--silence-ms 700", meaning: "Pause that ends an utterance." },
            { flag: "--vad-threshold X", meaning: "Fixed RMS level (of the 200 Hz high-passed signal) counted as speech. Default: adaptive." },
            { flag: "--min-confidence 0.9", meaning: <>Drop one- or two-word utterances below this confidence (noise often decodes as "Yeah." / "Okay."). 0 keeps everything.</> },
          ]}
        />
        <P>
          On Linux the microphone is opened through PulseAudio's protocol, which PipeWire also serves via
          pipewire-pulse, so <IC>--mic-device</IC> picks the sound server's named sources. Plain ALSA is the
          fallback when no sound server runs.
        </P>
      </Section>
    </DocsLayout>
  );
}

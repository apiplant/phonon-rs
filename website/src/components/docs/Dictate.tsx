import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Pre, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsDictate() {
  return (
    <DocsLayout>
      <H1>phonon-dictate</H1>
      <Lead>
        Push-button dictation for Linux (Wayland or X11). Press a hotkey, speak, press it again, and the
        transcript is typed into whatever window has focus. The model stays loaded, so a sentence comes back
        in about 50 ms on a GPU.
      </Lead>
      <P>
        macOS packages contain only <IC>phonon</IC>: the daemon reads hotkeys from evdev and types through
        uinput, which are Linux interfaces.
      </P>

      <Section>
        <H2>Setup</H2>
        <CopyBlock
          command={`phonon-dictate setup          # press your combo twice; saved to ~/.config/phonon/dictate.json
phonon-dictate setup --hold   # same, but push-to-talk (record while held)
phonon-dictate                # run the daemon (keep it running: autostart / systemd user unit)`}
        />
        <P>Instead of a saved hotkey you can:</P>
        <UL>
          <LI>
            pass one on the command line: <IC>--key f9</IC>, <IC>--key ctrl+alt+d</IC>,{" "}
            <IC>--key super+space</IC>, <IC>--hold</IC>
          </LI>
          <LI>
            skip the evdev hotkey and bind <IC>phonon-dictate toggle</IC> in your desktop's shortcut settings
            (KDE System Settings, or your compositor's config). <IC>start</IC> and <IC>stop</IC> also exist, and
            the daemon listens on <IC>$XDG_RUNTIME_DIR/phonon-dictate.sock</IC>. Run the daemon with{" "}
            <IC>--no-hotkey</IC> in that case.
          </LI>
        </UL>
      </Section>

      <Section>
        <H2>How it works</H2>
        <UL>
          <LI>
            <strong class="font-medium text-ink">Hotkeys</strong> are read from <IC>/dev/input/event*</IC>{" "}
            (evdev), so they work under any compositor. Keyboards plugged in later are picked up too. The hotkey
            is not grabbed, so the focused app also sees it: pick a combo that does nothing else (F13 to F24,
            Pause, an unused Super+letter, a mouse side button).
          </LI>
          <LI>
            <strong class="font-medium text-ink">Text</strong> is typed through a virtual keyboard on{" "}
            <IC>/dev/uinput</IC>, using the US layout. Characters it cannot type are pasted with{" "}
            <IC>wl-copy</IC> then Ctrl+V. <IC>--output paste</IC> always pastes (use it with a non-US layout),{" "}
            <IC>--output clipboard</IC> only copies, <IC>--output stdout</IC> prints.
          </LI>
          <LI>
            <strong class="font-medium text-ink">Both need the input group</strong>:{" "}
            <IC>sudo usermod -aG input $USER</IC>, then log in again.
          </LI>
          <LI>
            Recordings without enough speech (under 0.25 s voiced), or decoding to one or two low-confidence
            words, are discarded. Knocks and breaths do not type "Yeah.".
          </LI>
        </UL>
        <FlagTable
          rows={[
            { flag: "--mic-device NAME", meaning: "Input device." },
            { flag: "--quiet", meaning: "No notifications." },
            { flag: "--no-space", meaning: "No trailing space after the typed text." },
            { flag: "--max-secs 300", meaning: "Longest single recording." },
          ]}
        />
      </Section>

      <Section>
        <H2>Run it at login</H2>
        <Pre caption="~/.config/systemd/user/phonon-dictate.service" lang="text">{`[Unit]
Description=Phonon dictation
After=graphical-session.target

[Service]
ExecStart=/usr/bin/phonon-dictate --model /path/to/model_phonon2_c4c_int6
Restart=on-failure

[Install]
WantedBy=graphical-session.target`}</Pre>
        <CopyBlock command="systemctl --user enable --now phonon-dictate" />
      </Section>
    </DocsLayout>
  );
}

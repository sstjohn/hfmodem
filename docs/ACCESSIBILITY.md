<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Accessibility

The hfmodem website targets [WCAG 2.2 Level AA](https://www.w3.org/TR/WCAG22/) and follows [W3C Web Accessibility Initiative guidance](https://www.w3.org/WAI/). The aim is to work with existing assistive technologies, including screen readers, speech input, keyboards and switches used by people with physical or motor disabilities. Audio is optional throughout the recorded examples.

## Reading and navigation

Every published page has a main landmark and a Skip to content link. Links, native buttons, selects and disclosures work with a keyboard. Tab moves forward, Shift-Tab moves back, and Enter or Space activates the focused control. A visible focus outline shows where you are. Wide code samples, tables and diagrams have keyboard-scrollable regions; surrounding prose should reflow with browser zoom.

Diagrams start paused. Use their Play controls to animate them, or read their text and numerical data views without starting motion. Operating-system reduced-motion and forced-color preferences are respected. Color supplements sender, mode and result labels.

## Recorded exchanges

The [curated logbook](https://hfmodem.ssj.lol/examples/logbook/) contains six examples with separate send and receive records for VARA, PACTOR-3 and ARDOP. The session summary describes mail outcomes and evidence limits. In an inspector, use the section links to reach playback, frame inspection, event filters or signal measurements directly.

Playback begins only when requested. Listen is optional. The recordings contain modem tones and radio noise rather than speech. Identified events describe timing, sender, modulation and decoding results; gaps indicate unidentified activity, not necessarily silence. Packet signal measurements have a text view and downloadable numerical data. The decoded-byte view can contain incomplete or compressed binary data; it is not a complete transcript or necessarily the original mail text.

Entering a changing packet panel pauses playback so its controls remain usable for reading, voice input and switch scanning. Use Play to resume. Frames and observations are shown in pages of up to 25 controls, with search and direction/result filters. The exchange ladder shows the latest 40 packets in the chosen view; the event pages provide the full list. Previous event and Next event inspect packets without clicking a plot. Playback position provides a native keyboard-accessible slider.

Keyboard preferences in the footer let you disable letter-key shortcuts. A Shortcut sequence field also accepts a known sequence as text, with no typing deadline. This works with dictation and slow switch input without requiring timed keypresses.

## Verification and remaining barriers

Accessibility is a release requirement, not a claim of universal access. Automated axe checks cover every generated HTML route in light and dark themes against WCAG 2.2 AA rules. Build checks cover landmarks, local navigation and labels; browser checks cover keyboard operation, focus stability, motion controls and narrow-screen reflow. These checks cannot establish conformance on their own.

Live screen-reader testing, actual speech recognition and sip-and-puff hardware testing remain outstanding in this release review. Keyboard and browser accessibility-tree checks are useful but do not replace those tests. Screen-reader and switch users may encounter differences between browsers and assistive technologies. Dense numerical signal data and compressed decoder payloads can still be difficult to interpret. The operating desktop is a static, simulated demo. Its recorded logbook uses these same six public exchanges. Automated checks cover its desks, dialogs and embedded inspectors; live assistive-technology testing remains outstanding there too. The demo cannot operate a radio.

Report a barrier through the [project issue tracker](https://github.com/sstjohn/hfmodem/issues), including the page, intended action, browser and assistive technology when practical. A plain description of what failed is sufficient.

## Assistive-technology test checklist

Use the homepage, a protocol guide, the logbook and one inspector from each protocol. Repeat with the browser’s light/dark themes, 200% text enlargement and 400% zoom. Wide signal data and tables may scroll within their own regions; prose and controls should remain visible without horizontal page scrolling.

- With a screen reader, find the main landmark, move through headings, open a text/data disclosure, inspect a packet and read its direction, decode result and evidence limits. Confirm that Play and Pause are named correctly and that animation does not continuously interrupt speech. Filter the event list to zero matches and confirm a single useful status announcement.
- With keyboard-only access or a switch configured to send Tab, Shift-Tab, Enter and Space, reach every control and return from every region. Use section jumps, event pages, Previous/Next event and the seek slider without clicking a plot. Pause, read packet details, resume, then pause again. Focus must stay visible and the selected control must remain usable.
- With voice control, activate controls by their visible names, select a mode, dictate into Find and run a known Shortcut sequence. Disable letter-key shortcuts and verify that dictation and navigation do not invoke them. Do not depend on canvas coordinates or drag gestures.
- With audio off, follow the session outcome, sender labels, timing, packet results and signal measurements. Check that unknown activity, missing recordings and incomplete decoded traffic are explained without requiring sound.

Record browser and assistive-technology versions, the action and observed result. Mark tests that have not been run; a failing task remains a release barrier until fixed or clearly disclosed.

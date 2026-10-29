# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Narrow Frotz source edits for structured pipe/file events.

Applied only to the pinned source by the offline recipe. Upstream C remains
GPL-2.0-or-later. These modifications do not provide filesystem confinement.
"""
from pathlib import Path


def apply(root: Path) -> None:
    path = root / "src/dumb/dumb_input.c"
    source = path.read_text()
    source = source.replace('    dumb_getline(s);\n', '''    if (getenv("HF_BBS_HOOK")) {
      fprintf(stderr, "HF_BBS %s\\n", type == INPUT_CHAR ? "key" : "line");
      fflush(stderr);
    }
    dumb_getline(s);
    if (getenv("HF_BBS_HOOK")) { s[strlen(s)-1] = ZC_RETURN; return FALSE; }
''', 1)
    marker = '  /* If we\'re restoring a game before the interpreter starts,'
    hook = '''  /* hfmodem managed file requests: no caller-selected host paths. */
  if (getenv("HF_BBS_HOOK")) {
    if (flag != FILE_SAVE && flag != FILE_RESTORE) return FALSE;
    fflush(stdout);
    fprintf(stderr, "HF_BBS %s-file\\n", flag == FILE_SAVE ? "save" : "restore");
    fflush(stderr);
    dumb_getline(buf);
    buf[strcspn(buf, "\\r\\n")] = '\\0';
    if (strcmp(buf, flag == FILE_SAVE ? "pending.qzl" : "restore.qzl")) return FALSE;
    strcpy(file_name, buf);
    return TRUE;
  }

'''
    if marker not in source:
        raise ValueError("pinned Frotz file hook location changed")
    source = source.replace(marker, hook + marker, 1)
    path.write_text(source)
    path = root / "src/common/fastmem.c"
    source = path.read_text()
    # Actual VM save/restore result, never inferred from display words.
    for name, event in (("z_save", "saved"), ("z_restore", "restored")):
        start = source.index("void " + name + " (void)")
        end = source.index("}/* " + name + " */", start)
        section = source[start:end]
        marker = "finished:\n"
        hook = '''finished:
    if (getenv("HF_BBS_HOOK")) {
        fflush(stdout);
        fprintf(stderr, "HF_BBS %s\\n", success > 0 ? "%s" : "%s-failed");
        fflush(stderr);
    }
''' % ("%s", event, "save" if name == "z_save" else "restore")
        if marker not in section:
            raise ValueError("pinned Frotz result hook location changed")
        source = source[:start] + section.replace(marker, hook, 1) + source[end:]
    path.write_text(source)

# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline proof CLI; never invoked by the modem runtime."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
RUNTIME_SIZES = (280, 348, 352, 704, 1056, 1160, 1408, 1656, 4140, 8280, 12420, 16560, 20700)
# Complete session prefixes, not independently reseeded calls. Budgets count
# ACCEPTED swaps. The first 2484/S38 call was an unkept tuning trial.
RECIPES = {
    280: (5, ((280, 15, 5600),)),
    348: (68, ((348, 16, 6960),)),
    352: (81, ((352, 17, 7040),)),
    704: (30, ((704, 24, 14080),)),
    1056: (82, ((1056, 28, 21120),)),
    1408: (79, ((1408, 33, 28160),)),
    2484: (99, ((2484, 38, 49680), (2484, 40, 49680))),
    1656: (99, ((2484, 38, 49680), (2484, 40, 49680), (1656, 32, 33120))),
    4140: (88, ((4140, 40, 82800),)),
    8280: (51, ((8280, 55, 165600),)),
    12420: (51, ((8280, 55, 165600), (12420, 65, 248400))),
    16560: (91, ((16560, 70, 331200),)),
    20700: (51, ((8280, 55, 165600), (12420, 65, 248400), (20700, 85, 414000))),
}


def emit(record):
    print(json.dumps(record, sort_keys=True), flush=True)


def compile_c(cc, source, binary):
    command = shlex.split(cc) + ["-O3", "-std=c11", "-Wall", "-Wextra", "-o", str(binary), str(source), "-lm"]
    subprocess.run(command, check=True)
    return command


def run_generator(binary, method, seed, calls, cap):
    command = [str(binary), method, str(seed), str(cap)]
    command += [str(v) for call in calls for v in call]
    start = time.monotonic()
    result = subprocess.run(command, capture_output=True, check=False)
    records = []
    for line in result.stderr.decode("utf-8", errors="replace").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"diagnostic": line})
    return result, records, time.monotonic() - start


def values(raw, n):
    if len(raw) != 2 * n:
        raise ValueError(f"expected {2*n} uint16LE bytes, got {len(raw)}")
    data = struct.unpack(f"<{n}H", raw)
    if sorted(data) != list(range(n)):
        raise ValueError("not a permutation of range(N)")
    return data


def reference_bytes(n, reference):
    if reference:
        raw = Path(reference).read_bytes()
        values(raw, n)
        return raw, "explicit_uint16le_reference", None
    if n not in RUNTIME_SIZES:
        raise ValueError("research-only 2484 requires --reference UINT16LE_FILE for verification")
    from .. import p4c2
    data = tuple(int(x) for x in p4c2.permutation(n))
    raw = struct.pack(f"<{n}H", *data)
    values(raw, n)
    expected = p4c2.TABLE_SHA256[n]
    if sha256(raw).hexdigest() != expected:
        raise ValueError("runtime permutation differs from its declared SHA-256")
    return raw, "p4c2.permutation", expected


def verify_one(binary, n, args):
    reference, origin, expected = reference_bytes(n, args.reference)
    if n == 1160:
        emit({"n": n, "status": "unresolved_recipe", "cml_proof": False,
              "runtime_sha256": sha256(reference).hexdigest(), "reference_kind": origin,
              "reason": "Compact runtime representation exists; no recovered seed/history is claimed."})
        return 3
    seed, calls = RECIPES[n]
    result, records, seconds = run_generator(binary, args.method, seed, calls, args.max_proposals)
    row = {"n": n, "seed": seed, "calls": calls, "method": args.method,
           "max_proposals_per_stage": args.max_proposals, "stages": records,
           "seconds": seconds, "reference_kind": origin, "reference_sha256": sha256(reference).hexdigest(),
           "source_sha256": sha256((HERE / "cml_generate.c").read_bytes()).hexdigest(),
           "binary_sha256": sha256(binary.read_bytes()).hexdigest()}
    if result.returncode:
        row.update(status="incomplete_cap" if result.returncode == 3 else "generator_error", cml_proof=False)
        emit(row)
        return 3 if result.returncode == 3 else 1
    generated = values(result.stdout, n)
    compared = values(reference, n)
    differences = [i for i, (a, b) in enumerate(zip(generated, compared)) if a != b]
    got_sha = sha256(result.stdout).hexdigest()
    checks = {"all_entries": not differences, "canonical_bytes": result.stdout == reference,
              "canonical_sha256": got_sha == sha256(reference).hexdigest(),
              "all_stages_completed": len(records) == len(calls) and all(
                  r.get("status") == "complete" and r.get("n") == call[0]
                  and r.get("s") == call[1] and r.get("swaps") == call[2]
                  for r, call in zip(records, calls))}
    if expected is not None:
        checks["declared_runtime_sha256"] = got_sha == expected
    passed = all(checks.values())
    row.update(status="verified" if passed else "mismatch", cml_proof=passed,
               generated_sha256=got_sha, checks=checks,
               mismatching_entries=len(differences), first_mismatches=differences[:16])
    emit(row)
    return 0 if passed else 1


def self_test(binary, cc, directory):
    # Independent literal upstream algorithm; only explicit seed and budget are
    # substituted in this temporary control. Fail if the expected text drifts.
    upstream = (HERE / "upstream" / "srandom.h").read_text()
    seed_line = "idum = -(tmptr->tm_mon + tmptr->tm_mday + tmptr->tm_hour + tmptr->tm_min + tmptr->tm_sec);"
    budget_line = "max_swaps= 10*N;"
    if upstream.count(seed_line) != 1 or upstream.count(budget_line) != 1:
        raise ValueError("upstream seed/budget substitution anchors changed")
    upstream = upstream.replace(seed_line, "idum = -proof_seed;").replace(budget_line, "max_swaps = proof_swaps;")
    wrapper = r'''
int main(int argc,char **argv) {
    if(argc<5 || (argc-2)%3) return 2;
    proof_seed=atol(argv[1]); int *a=NULL,n=0;
    for(int q=2;q<argc;q+=3) {
        n=atoi(argv[q]); proof_swaps=atoi(argv[q+2]);
        free(a); a=malloc(sizeof(int)*n); if(!a) return 2;
        CreateSRandomInterleaver(n,atoi(argv[q+1]),a);
    }
    for(int i=0;i<n;i++) { putchar(a[i]&255); putchar((a[i]>>8)&255); }
    free(a); return 0;
}
'''
    source = directory / "upstream_control.c"
    source.write_text("static long proof_seed; static int proof_swaps;\n" + upstream + wrapper)
    original = directory / "upstream_control"
    compile_c(cc, source, original)
    # N9/S2 crosses a literal lower guard boundary; chaining checks no reseed.
    cases = [(1, ((9, 2, 4),)), (17, ((37, 3, 80),)),
             (99, ((37, 3, 80), (31, 3, 40))), (5, ((1, 1, 0),))]
    for seed, calls in cases:
        outputs = []
        stages = []
        for method in ("scalar", "cached", "audit"):
            result, record, _ = run_generator(binary, method, seed, calls, 1_000_000)
            if result.returncode:
                raise ValueError(f"tiny {method} control did not complete: {record}")
            outputs.append(result.stdout)
            stages.append(record)
        command = [str(original), str(seed)] + [str(v) for call in calls for v in call]
        literal = subprocess.run(command, capture_output=True, check=True).stdout
        if any(out != literal for out in outputs) or any(x != stages[0] for x in stages):
            raise ValueError("scalar/cache/audit/upstream control disagreement")
        values(literal, calls[-1][0])
        emit({"status": "self_test_pass", "seed": seed, "calls": calls,
              "sha256": sha256(literal).hexdigest(), "stages": stages[0],
              "compared": ["upstream", "scalar", "cached", "audit"]})
    # Uncompletable N1/S1 is bounded explicitly: no table output on cap.
    capped, rows, _ = run_generator(binary, "audit", 1, ((1, 1, 1),), 8)
    if capped.returncode != 3 or capped.stdout or rows[-1].get("status") != "incomplete_cap":
        raise ValueError("cap was not reported as an incomplete run")
    emit({"status": "self_test_pass", "case": "cap_emits_no_table"})


def call_argument(value):
    try:
        result = tuple(int(v) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use N,S,SWAPS") from exc
    if len(result) != 3 or not 1 <= result[1] <= result[0] <= 65535 or result[2] < 0:
        raise argparse.ArgumentTypeError("require 1 <= S <= N <= 65535 and SWAPS >= 0")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="cc", help="C compiler command (no -ffast-math)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="recipe inventory without compilation or generation")
    sub.add_parser("self-test", help="tiny independent upstream/scalar/cache controls")
    verify = sub.add_parser("verify", help="regenerate selected recovered recipes; may take hours")
    verify.add_argument("sizes", nargs="*", type=int, choices=(*RUNTIME_SIZES, 2484))
    verify.add_argument("--all-solved", action="store_true", help="all 12 solved runtime sizes, potentially very slow")
    verify.add_argument("--reference", help="explicit canonical uint16LE file (one size only)")
    generate = sub.add_parser("generate", help="arbitrary explicit CML seed/history; not a recipe discovery claim")
    generate.add_argument("--seed", required=True, type=int)
    generate.add_argument("--call", required=True, action="append", type=call_argument, help="ordered N,S,SWAPS; repeat for each call")
    generate.add_argument("--output", required=True, help="new output uint16LE file; existing files refused")
    for p in (verify, generate):
        p.add_argument("--method", choices=("scalar", "cached", "audit"), default="cached")
        p.add_argument("--max-proposals", type=int, default=0, help="per-stage cap; 0 means no cap; caps never prove a table")
    args = parser.parse_args(argv)
    if args.command == "list":
        for n in (*RUNTIME_SIZES, 2484):
            recipe = RECIPES.get(n)
            emit({"n": n, "runtime": n in RUNTIME_SIZES, "recipe_recovered": recipe is not None,
                  "seed": recipe[0] if recipe else None, "calls": recipe[1] if recipe else None,
                  "status": "recipe_available_not_executed" if recipe else "unresolved_recipe"})
        return 0
    if args.command == "verify":
        if args.all_solved and args.sizes:
            parser.error("choose explicit sizes or --all-solved")
        sizes = [n for n in RUNTIME_SIZES if n in RECIPES] if args.all_solved else args.sizes
        if not sizes:
            parser.error("select sizes or --all-solved")
        if args.reference and sizes != [2484]:
            parser.error("--reference is only for the research-only 2484 selector")
        sizes = list(dict.fromkeys(sizes))
    if getattr(args, "max_proposals", 0) < 0:
        parser.error("--max-proposals must be nonnegative")
    if args.command == "generate" and not 0 <= args.seed < 2147483399:
        parser.error("seed must be between0 and2147483398")
    if args.command == "generate" and (Path(args.output).exists() or Path(args.output).is_symlink()):
        parser.error("output already exists; choose a new file")
    try:
        if args.command == "verify" and sizes == [1160]:
            return verify_one(None, 1160, args)  # No recipe, so no compiler needed.
        with tempfile.TemporaryDirectory(prefix="c2proof-") as temporary:
            directory = Path(temporary)
            binary = directory / "cml_generate"
            compile_c(args.cc, HERE / "cml_generate.c", binary)
            if args.command == "self-test":
                self_test(binary, args.cc, directory)
                return 0
            if args.command == "verify":
                codes = [verify_one(binary, n, args) for n in sizes]
                return 1 if 1 in codes else (3 if 3 in codes else 0)
            result, records, seconds = run_generator(binary, args.method, args.seed, args.call, args.max_proposals)
            if result.returncode:
                emit({"status": "incomplete_cap" if result.returncode == 3 else "generator_error",
                      "cml_proof": False, "stages": records})
                return 3 if result.returncode == 3 else 1
            values(result.stdout, args.call[-1][0])
            with open(args.output, "xb") as output:
                output.write(result.stdout)
            emit({"status": "generated_not_compared", "cml_proof": False, "seed": args.seed,
                  "calls": args.call, "stages": records, "seconds": seconds,
                  "sha256": sha256(result.stdout).hexdigest()})
            return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        emit({"status": "error", "cml_proof": False, "reason": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

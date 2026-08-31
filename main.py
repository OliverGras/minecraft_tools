#!/usr/bin/env python3
"""
find_oversized_blockentity.py

Scans Minecraft .mca region file(s) and reports block entities (tile
entities) in them -- either every one sorted largest-first (to find what's
tripping the client's 2 MiB / 2,097,152 byte NbtAccounter decode cap), or
only ones matching a specific block ID (e.g. to find every AE2 drive or
Sophisticated Storage backpack in the world/a region/a chunk).

IMPORTANT -- WHICH SIZE MATTERS:
    NbtAccounter does not measure the serialized size of a tag. It charges
    an estimated JVM heap cost per tag as it decodes (48 bytes per compound
    + 36 per entry + 28 + 2/char per key, 36 + 2/char per string, 37 + 4
    per list element, and so on). A block entity full of many small nested
    tags -- thousands of in-transit items in a pipe, an inventory of item
    stacks with components -- therefore costs 3-7x MORE than its size on
    disk. A 900 KB block entity on disk can be 6 MB to the accounter and
    blow the cap.

    So this tool reports BOTH numbers on every line and ranks by the
    accounted estimate:
        acct = estimated NbtAccounter cost -- what the 2 MiB cap applies to
        disk = serialized bytes
    The accounted number is an estimate: the constants are read off the
    1.20.2+/1.21.x NbtAccounter, so treat anything within ~10% of the cap
    as a suspect.

Requires: pip install nbtlib --break-system-packages

USAGE:
    python3 find_oversized_blockentity.py <path> [chunkX chunkZ] [--over-limit] [--block ID] [--exact] [--min-bytes N]

    <path> can be:
      - a single region file, e.g. world/region/r.-1.0.mca
      - a region/ folder, to scan every .mca file in it, e.g. world/region

    chunkX / chunkZ (optional): ABSOLUTE chunk coordinates to restrict to
    one chunk. Only valid when <path> is a single region file.

    --over-limit / -o (optional): report EVERY block entity at or over the
      2,097,152 byte cap, in every chunk of every scanned file, instead of
      only the largest one per chunk. Combine with --min-bytes to use a
      different threshold (e.g. --over-limit --min-bytes 1500000 to also
      catch ones that are merely close to the cap). Can be combined with
      --block to restrict the search to certain block types.

    --block ID (optional, repeatable): only report block entities whose id
      contains this text (case-insensitive substring match by default).
      e.g. --block ae2:drive  --block sophisticatedstorage

    --exact: match --block against the full id exactly instead of substring.

    --serialized: rank and filter by on-disk serialized size instead of the
      accounted estimate. Only useful for inspecting disk usage -- the cap
      is not compared against this number.

    --breakdown [N]: after scanning, show where the accounted bytes live
      inside the N largest block entities (default 1) -- the biggest child
      tags plus a per-tag-type census. This is what tells you which field
      is bloated and needs trimming.

    --min-bytes N (optional): only report block entities at or above N
      bytes. Defaults to 0 (show everything) when --block is used, or
      omitted entirely in whole-region "largest per chunk" summary mode.

EXAMPLES:
    # Find the single oversized block entity in one chunk
    python3 find_oversized_blockentity.py world/region/r.-1.0.mca -15 25

    # Scan a whole region file, largest block entity per chunk
    python3 find_oversized_blockentity.py world/region/r.-1.0.mca

    # Find every AE2 drive anywhere in the world, with sizes
    python3 find_oversized_blockentity.py world/region --block ae2:drive

    # Find every Sophisticated Storage backpack over 500KB anywhere
    python3 find_oversized_blockentity.py world/region --block sophisticatedbackpacks --min-bytes 500000

    # List EVERY block entity in the world that is over the 2 MiB cap
    python3 find_oversized_blockentity.py world/region --over-limit

    # Same, but also catch ones merely approaching the cap
    python3 find_oversized_blockentity.py world/region --over-limit --min-bytes 1500000

    # Find the offender and show what inside it is actually big
    python3 find_oversized_blockentity.py world/region/r.-1.0.mca -15 26 --breakdown

This is READ-ONLY. It does not modify your world. Once you've found the
offending block entity's coordinates, remove/edit it with MCA Selector
or NBTExplorer.
"""

import sys
import os
import re
import struct
import zlib
import gzip
import io
import argparse

try:
    import nbtlib
    from nbtlib import tag as nbttag
    from nbtlib import Compound, List
except ImportError:
    print("Missing dependency. Install it with:")
    print("    pip install nbtlib --break-system-packages")
    sys.exit(1)

TWO_MIB = 2 * 1024 * 1024  # 2,097,152 -- the vanilla client NbtAccounter cap
NEAR_CAP = 2_000_000       # anything above this is worth flagging as suspect

# NbtAccounter does NOT count serialized bytes -- it charges an ESTIMATED
# JVM heap cost per tag as it decodes. A block entity holding thousands of
# small nested tags (pipe contents, item stacks) can sit at a few hundred KB
# on disk yet blow past the 2 MiB cap while being read. These are the
# per-tag charges from net.minecraft.nbt.NbtAccounter / the TagType.load
# implementations (1.20.2+ / 1.21.x).
COMPOUND_BASE = 48      # CompoundTag.load
COMPOUND_ENTRY = 36     # per HashMap entry stored
COMPOUND_KEY_BASE = 28  # NbtAccounter.readUTF: 28 + 2 per char, for the key
LIST_BASE = 37          # ListTag.load
LIST_ELEMENT_REF = 4    # per element reference in the backing ArrayList
STRING_BASE = 36        # StringTag.load: 36 + 2 per char
ARRAY_BASE = 24         # Byte/Int/LongArrayTag: 24 + width * length
PRIMITIVE_COST = {      # Byte/Short/Int/Long/Float/DoubleTag.load
    "Byte": 9, "Short": 10, "Int": 12, "Long": 16, "Float": 12, "Double": 16,
}
ARRAY_WIDTH = {"ByteArray": 1, "IntArray": 4, "LongArray": 8}


def parse_region_filename(path):
    """Extract region X,Z from a filename like r.-1.1.mca."""
    m = re.search(r"r\.(-?\d+)\.(-?\d+)\.mca$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def read_chunk_payloads(path):
    """
    Yields (chunk_x_in_region, chunk_z_in_region, raw_nbt_bytes) for every
    generated chunk in the region file, decompressing as needed.
    """
    with open(path, "rb") as f:
        header = f.read(8192)
        if len(header) < 8192:
            raise ValueError("File too small to be a valid region file")

        file_size = os.path.getsize(path)

        for cz in range(32):
            for cx in range(32):
                idx = (cx + cz * 32) * 4
                entry = header[idx:idx + 4]
                offset_sectors = (entry[0] << 16) | (entry[1] << 8) | entry[2]
                sector_count = entry[3]
                if offset_sectors == 0 and sector_count == 0:
                    continue  # chunk not generated

                byte_offset = offset_sectors * 4096
                if byte_offset + 5 > file_size:
                    # corrupt/truncated entry, skip
                    continue

                f.seek(byte_offset)
                length_bytes = f.read(4)
                if len(length_bytes) < 4:
                    continue
                (length,) = struct.unpack(">I", length_bytes)
                if length == 0:
                    continue
                compression_type = f.read(1)[0]
                payload = f.read(length - 1)

                if compression_type == 1:
                    raw = gzip.decompress(payload)
                elif compression_type == 2:
                    raw = zlib.decompress(payload)
                elif compression_type == 3:
                    raw = payload  # uncompressed
                else:
                    print(f"  chunk ({cx},{cz}): unsupported compression type {compression_type}, skipping")
                    continue

                yield cx, cz, raw


def find_block_entities(raw_nbt_bytes, keep_tags=False):
    """
    Parses one chunk's decompressed NBT and returns a list of dicts:
    {id, x, y, z, bytes, accounted} for every block entity found, largest
    (by accounted size) first.
    Handles both the modern (1.18+) root-level "block_entities" list and
    the older "Level.TileEntities" layout.
    """
    buf = io.BytesIO(raw_nbt_bytes)
    nbt_file = nbtlib.File.parse(buf)
    root = nbt_file  # nbtlib.File behaves like the root Compound

    # Locate the list of block entities regardless of MC version layout
    if "block_entities" in root:
        be_list = root["block_entities"]
    elif "Level" in root and "TileEntities" in root["Level"]:
        be_list = root["Level"]["TileEntities"]
    else:
        return []

    results = []
    for be in be_list:
        # Serialize just this one compound to measure its own byte size
        tmp = nbtlib.File(be, root_name="")
        out = io.BytesIO()
        tmp.write(out)
        size = len(out.getvalue())

        entry = {
            "id": str(be.get("id", "?")),
            "x": int(be.get("x", 0)),
            "y": int(be.get("y", 0)),
            "z": int(be.get("z", 0)),
            "bytes": size,               # serialized, on disk
            "accounted": accounted_size(be),  # what NbtAccounter charges
        }
        if keep_tags:
            entry["tag"] = be
        results.append(entry)

    results.sort(key=lambda e: -e["accounted"])
    return results


def utf16_len(s):
    """Java String.length() counts UTF-16 code units, not code points."""
    return len(s) if s.isascii() else len(s.encode("utf-16-le")) // 2


def accounted_size(t):
    """
    Estimate what Minecraft's NbtAccounter will charge for this tag while
    decoding it -- i.e. the number the 2,097,152 byte cap is compared
    against. This is what actually causes the disconnect; the serialized
    size on disk is typically 2-4x smaller.
    """
    name = type(t).__name__

    if isinstance(t, nbttag.Compound):
        total = COMPOUND_BASE
        for key, value in t.items():
            total += COMPOUND_KEY_BASE + 2 * utf16_len(key)
            total += COMPOUND_ENTRY
            total += accounted_size(value)
        return total

    if isinstance(t, nbttag.String):
        return STRING_BASE + 2 * utf16_len(t)

    if isinstance(t, nbttag.List):
        total = LIST_BASE + LIST_ELEMENT_REF * len(t)
        for item in t:
            total += accounted_size(item)
        return total

    if name in ARRAY_WIDTH:
        return ARRAY_BASE + ARRAY_WIDTH[name] * len(t)

    return PRIMITIVE_COST.get(name, 8)


def tag_census(t, out=None):
    """Count tags by type and their accounted cost, for --breakdown."""
    if out is None:
        out = {}

    name = type(t).__name__
    entry = out.setdefault(name, {"count": 0, "accounted": 0})
    entry["count"] += 1

    if isinstance(t, nbttag.Compound):
        entry["accounted"] += COMPOUND_BASE
        for key, value in t.items():
            entry["accounted"] += COMPOUND_KEY_BASE + 2 * utf16_len(key) + COMPOUND_ENTRY
            tag_census(value, out)
    elif isinstance(t, nbttag.List):
        entry["accounted"] += LIST_BASE + LIST_ELEMENT_REF * len(t)
        for item in t:
            tag_census(item, out)
    else:
        entry["accounted"] += accounted_size(t)

    return out


def print_breakdown(be, max_depth=3):
    """Show where a block entity's accounted bytes actually live."""
    root = be.get("tag")
    if root is None:
        return

    print(f"  BREAKDOWN: {be['id']} at ({be['x']}, {be['y']}, {be['z']}) "
          f"-- {be['accounted']:,} accounted bytes")

    def walk(t, path, depth):
        if depth > max_depth:
            return
        if isinstance(t, nbttag.Compound):
            children = [(str(k), v) for k, v in t.items()]
        elif isinstance(t, nbttag.List):
            children = [(f"[{i}]", v) for i, v in enumerate(t)]
        else:
            return

        sized = sorted(((n, v, accounted_size(v)) for n, v in children),
                       key=lambda e: -e[2])
        for child_name, value, size in sized[:6]:
            if size < 1000:
                continue
            kind = type(value).__name__
            count = f" x{len(value)}" if isinstance(value, (nbttag.Compound, nbttag.List)) else ""
            print(f"    {'  ' * depth}{path}{child_name:<28} {kind}{count:<8} "
                  f"{size:>12,} bytes")
            walk(value, "", depth + 1)

    walk(root, "", 1)

    census = tag_census(root)
    print("    tag census (type: count, accounted bytes):")
    for name, entry in sorted(census.items(), key=lambda e: -e[1]["accounted"]):
        print(f"      {name:<12} {entry['count']:>10,} tags  {entry['accounted']:>14,} bytes")
    print()


def matches_block_filter(entity_id, block_filters, exact):
    if not block_filters:
        return True
    eid = entity_id.lower()
    for f in block_filters:
        fl = f.lower()
        if exact:
            if eid == fl:
                return True
        else:
            if fl in eid:
                return True
    return False


def size_label(be, metric):
    """Both numbers on every line -- the one that matters is 'acct'."""
    flag = ""
    if be["accounted"] >= TWO_MIB:
        flag = "  <<< OVER THE 2 MiB CAP"
    elif be["accounted"] > NEAR_CAP:
        flag = "  <<< close to the cap"
    return (f"acct {be['accounted']:>11,} | disk {be['bytes']:>10,} bytes{flag}")


def format_entity_line(be, metric, indent="    "):
    return (f"{indent}{be['id']:<40} pos({be['x']:>6},{be['y']:>4},{be['z']:>6})  "
            f"{size_label(be, metric)}")


def format_chunk_line(be, abs_x, abs_z, metric):
    return (f"  Chunk ({abs_x:>5},{abs_z:>5})  {be['id']:<40} "
            f"pos({be['x']:>6},{be['y']:>4},{be['z']:>6})  {size_label(be, metric)}")


def remember(be, abs_x, abs_z, worst_overall, metric, keep):
    record = dict(be, chunk_x=abs_x, chunk_z=abs_z)
    if worst_overall[0] is None or record[metric] > worst_overall[0][metric]:
        worst_overall[0] = record
    if not keep:
        record.pop("tag", None)
    return record


def scan_region_file(path, target, opts, worst_overall, oversized, candidates):
    region_coords = parse_region_filename(path)
    if region_coords is None:
        print(f"Warning: couldn't parse region X,Z from filename '{os.path.basename(path)}', assuming region (0,0).")
        region_x, region_z = 0, 0
    else:
        region_x, region_z = region_coords

    print(f"Region file: {os.path.basename(path)} (region {region_x}, {region_z})")

    metric = opts.metric
    scanned = 0
    any_match_this_file = False

    for cx, cz, raw in read_chunk_payloads(path):
        abs_x = region_x * 32 + cx
        abs_z = region_z * 32 + cz

        if target is not None and (abs_x, abs_z) != target:
            continue

        scanned += 1
        try:
            entities = find_block_entities(raw, keep_tags=bool(opts.breakdown))
        except Exception as e:
            print(f"  Chunk ({abs_x}, {abs_z}): failed to parse NBT ({e}), skipping")
            continue

        if opts.breakdown:
            for be in entities:
                if matches_block_filter(be["id"], opts.block, opts.exact):
                    candidates.append(dict(be, chunk_x=abs_x, chunk_z=abs_z))
            if len(candidates) > 4 * opts.breakdown + 20:
                # only the biggest are ever printed; don't hold the rest in RAM
                candidates.sort(key=lambda e: -e["accounted"])
                del candidates[opts.breakdown:]

        if opts.over_limit:
            # Over-limit mode: list EVERY block entity at/over the threshold,
            # not just the biggest one in each chunk.
            hits = [e for e in entities
                    if e[metric] >= opts.min_bytes
                    and matches_block_filter(e["id"], opts.block, opts.exact)]
            for be in hits:
                any_match_this_file = True
                print(format_chunk_line(be, abs_x, abs_z, metric))
                oversized.append(remember(be, abs_x, abs_z, worst_overall, metric, False))
            continue

        if opts.block:
            # Block-ID search mode: list every matching block entity, any size
            matches = [e for e in entities
                       if matches_block_filter(e["id"], opts.block, opts.exact)
                       and e[metric] >= opts.min_bytes]
            for be in matches:
                any_match_this_file = True
                print(format_chunk_line(be, abs_x, abs_z, metric))
                remember(be, abs_x, abs_z, worst_overall, metric, False)
            continue

        # Default mode: largest-block-entity-per-chunk / full listing for one target chunk
        if not entities:
            if target is not None:
                print(f"  Chunk ({abs_x}, {abs_z}): no block entities found.")
            continue

        top = entities[0]

        if target is not None:
            print(f"  Block entities in chunk ({abs_x}, {abs_z}), sorted largest first:")
            for be in entities:
                if be[metric] < opts.min_bytes:
                    continue
                print(format_entity_line(be, metric))
        else:
            print(f"  Chunk ({abs_x:>5},{abs_z:>5}) largest: {top['id']:<40} "
                  f"pos({top['x']:>6},{top['y']:>4},{top['z']:>6})  {size_label(top, metric)}")

        remember(top, abs_x, abs_z, worst_overall, metric, False)

    if scanned == 0 and target is not None:
        print(f"  (chunk {target[0]}, {target[1]} not found in this region file)")
    if (opts.block or opts.over_limit) and not any_match_this_file:
        if opts.over_limit:
            print(f"  (nothing at or over {opts.min_bytes:,} {opts.metric} bytes in this file)")
        else:
            print("  (no matching block entities in this file)")


def main():
    ap = argparse.ArgumentParser(
        description="Find oversized or specific block entities in Minecraft region file(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", help="Path to a .mca region file, or a region/ folder to scan every .mca in it")
    ap.add_argument("chunk_x", nargs="?", type=int, help="Absolute chunk X (optional, single-file only)")
    ap.add_argument("chunk_z", nargs="?", type=int, help="Absolute chunk Z (optional, single-file only)")
    ap.add_argument("--over-limit", "-o", action="store_true",
                     help="Report EVERY block entity at or over the 2 MiB decode cap in every "
                          "chunk, instead of only the largest one per chunk. Use --min-bytes to "
                          "override the threshold; combine with --block to narrow by block type")
    ap.add_argument("--block", "-b", action="append", default=[],
                     help="Only report block entities whose id contains this text (repeatable). "
                          "e.g. --block ae2:drive")
    ap.add_argument("--exact", action="store_true",
                     help="Match --block against the full id exactly instead of substring")
    ap.add_argument("--min-bytes", type=int, default=None,
                     help="Only report block entities at or above this many bytes "
                          "(default: 0, or the 2 MiB cap when --over-limit is used)")
    ap.add_argument("--serialized", action="store_true",
                     help="Rank and filter by on-disk serialized size instead of the NbtAccounter "
                          "estimate. The estimate is what the 2 MiB cap is actually compared "
                          "against, so only use this to inspect disk usage")
    ap.add_argument("--breakdown", nargs="?", type=int, const=1, default=0, metavar="N",
                     help="After scanning, show where the accounted bytes live inside the N "
                          "largest block entities (default 1), plus a tag-type census. This is "
                          "what tells you which field to trim")
    args = ap.parse_args()

    # The NbtAccounter estimate is the metric the 2 MiB cap applies to.
    args.metric = "bytes" if args.serialized else "accounted"

    # --over-limit defaults its threshold to the client's decode cap
    if args.min_bytes is None and args.over_limit:
        args.min_bytes = TWO_MIB
    elif args.min_bytes is None:
        args.min_bytes = 0

    if not os.path.exists(args.path):
        print(f"Path not found: {args.path}")
        sys.exit(1)

    if os.path.isdir(args.path):
        if args.chunk_x is not None:
            print("chunkX/chunkZ can only be used with a single region file, not a folder. Ignoring.")
        region_files = sorted(
            os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith(".mca")
        )
        if not region_files:
            print(f"No .mca files found in {args.path}")
            sys.exit(1)
        target = None
    else:
        region_files = [args.path]
        target = None
        if args.chunk_x is not None and args.chunk_z is not None:
            target = (args.chunk_x, args.chunk_z)

    print("-" * 70)
    print("'acct' = estimated NbtAccounter cost, which is what the "
          f"{TWO_MIB:,} byte cap")
    print("applies to. 'disk' = serialized size. acct is normally 2-4x disk.")
    if args.over_limit:
        print(f"Listing every block entity at or over {args.min_bytes:,} {args.metric} bytes.")
    print("-" * 70)

    worst_overall = [None]  # boxed so scan_region_file can mutate it
    oversized = []          # every over-threshold hit, --over-limit mode only
    candidates = []         # tag refs retained for --breakdown
    for rf in region_files:
        scan_region_file(rf, target, args, worst_overall, oversized, candidates)
        print()

    print("-" * 70)

    if args.over_limit:
        if not oversized:
            print(f"No block entities at or over {args.min_bytes:,} {args.metric} bytes were found.")
            if args.min_bytes == TWO_MIB:
                print("Nothing here would trip the client's NbtAccounter cap -- if players are")
                print("still being disconnected, the culprit is in a region file outside this scan")
                print("(another dimension?), or try --min-bytes to catch ones just under the cap.")
        else:
            noun = "BLOCK ENTITY" if len(oversized) == 1 else "BLOCK ENTITIES"
            print(f"{len(oversized)} {noun} AT OR OVER {args.min_bytes:,} "
                  f"{args.metric.upper()} BYTES, largest first:")
            for be in sorted(oversized, key=lambda e: -e[args.metric]):
                print(format_chunk_line(be, be["chunk_x"], be["chunk_z"], args.metric))
            print()
            print("Each of these must be removed or trimmed (MCA Selector / NBTExplorer) before")
            print("the affected chunks will stream to clients again.")
    else:
        worst = worst_overall[0]
        if worst:
            label = "LARGEST MATCHING BLOCK ENTITY" if args.block else "LARGEST BLOCK ENTITY FOUND OVERALL"
            print(f"{label}:")
            print(f"  Chunk ({worst['chunk_x']}, {worst['chunk_z']}) - "
                  f"{worst['id']} at ({worst['x']}, {worst['y']}, {worst['z']})")
            print(f"  {worst['accounted']:,} accounted bytes / {worst['bytes']:,} bytes on disk")
            if worst["accounted"] >= TWO_MIB:
                print(f"  This is OVER the {TWO_MIB:,} byte cap -- this is the block entity")
                print("  causing the NbtAccounterException disconnects.")
            elif worst["accounted"] > NEAR_CAP:
                print(f"  This is just under the {TWO_MIB:,} byte cap. The estimate is")
                print("  approximate, so this is still a strong suspect.")
        else:
            print("No block entities found matching the given criteria.")

    if args.breakdown and candidates:
        print()
        print("-" * 70)
        for be in sorted(candidates, key=lambda e: -e["accounted"])[:args.breakdown]:
            print(f"Chunk ({be['chunk_x']}, {be['chunk_z']}):")
            print_breakdown(be)


if __name__ == "__main__":
    main()

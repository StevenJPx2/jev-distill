import { readFileSync } from "node:fs";

import { labelsOf, type TaskContract } from "./contract.js";
import { DistillError } from "./errors.js";
import { Decoder, parseJson, sha256Hex, unitInterval } from "./json.js";
import { contentHash, type TaskState, validateState } from "./state.js";

export type SplitSide = "train" | "holdout";

export interface DatasetRow {
  readonly id: string;
  readonly state: TaskState;
  readonly contentHash: string;
  readonly group: string | null;
  /** Human reference label, used only for evaluation, never for training. */
  readonly gold: string | null;
  readonly split: SplitSide;
  /** Leakage component: rows sharing content or group always share a side. */
  readonly component: string;
}

export interface DatasetStats {
  readonly lines: number;
  readonly unique: number;
  readonly duplicates_collapsed: number;
  readonly gold_conflicts_dropped: number;
  readonly components: number;
  readonly train: number;
  readonly holdout: number;
  readonly holdout_with_gold: number;
}

export interface Dataset {
  readonly rows: readonly DatasetRow[];
  readonly stats: DatasetStats;
  /** Hash of (id, content) pairs; identifies the exact dataset an artifact saw. */
  readonly fingerprint: string;
}

interface RawRow {
  readonly id: string;
  readonly explicitId: boolean;
  readonly state: TaskState;
  readonly contentHash: string;
  readonly group: string | null;
  readonly gold: string | null;
}

const d = new Decoder("invalid_dataset");
const ID = /^[A-Za-z0-9._:-]{1,128}$/;

export function readJsonLines(path: string, maxLineBytes: number, maxRows: number): { line: number; value: unknown }[] {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    throw new DistillError("io", `cannot read ${path}: ${(error as Error).message}`);
  }

  const out: { line: number; value: unknown }[] = [];
  const lines = text.split("\n");

  for (const [index, raw] of lines.entries()) {
    const line = raw.trim();
    if (line === "") continue;
    if (Buffer.byteLength(line) > maxLineBytes) d.fail(`${path}:${index + 1}`, `line exceeds max_line_bytes (${maxLineBytes})`);
    if (out.length >= maxRows) d.fail(path, `more than max_rows (${maxRows}) rows`);

    out.push({ line: index + 1, value: parseJson(line, "invalid_dataset", `${path}:${index + 1}`) });
  }

  return out;
}

export function loadDataset(path: string, contract: TaskContract): Dataset {
  const lines = readJsonLines(path, contract.dataset.max_line_bytes, contract.dataset.max_rows);
  const labels = labelsOf(contract);
  const raws = lines.map(({ line, value }) => decodeRow(value, `${path}:${line}`, contract, labels));

  return buildDataset(raws, contract);
}

function decodeRow(value: unknown, where: string, contract: TaskContract, labels: readonly string[]): RawRow {
  const o = d.object(value, where, ["id", "state", "group", "gold"]);
  const state = validateState(contract.state.fields, o.state, `${where} state`);
  const hash = contentHash(state);
  const gold = o.gold === undefined || o.gold === null ? null : d.oneOf(o.gold, `${where}.gold`, labels);
  const group = o.group === undefined || o.group === null ? null : d.string(o.group, `${where}.group`, { max: 256 });
  const explicitId = o.id !== undefined;

  return {
    id: explicitId ? d.string(o.id, `${where}.id`, { pattern: ID }) : `c_${hash.slice(0, 20)}`,
    explicitId,
    state,
    contentHash: hash,
    group,
    gold,
  };
}

function buildDataset(raws: readonly RawRow[], contract: TaskContract): Dataset {
  const seenIds = new Map<string, RawRow>();
  for (const row of raws) {
    const prior = seenIds.get(row.id);
    if (prior && (row.explicitId || prior.contentHash !== row.contentHash)) d.fail(`row ${row.id}`, "duplicate id");
    if (!prior) seenIds.set(row.id, row);
  }

  const components = leakageComponents(raws);
  const unique = new Map<string, { row: RawRow; golds: Set<string> }>();

  for (const row of raws) {
    const entry = unique.get(row.contentHash) ?? { row, golds: new Set<string>() };
    if (row.gold !== null) entry.golds.add(row.gold);
    unique.set(row.contentHash, entry);
  }

  let goldConflicts = 0;
  const rows: DatasetRow[] = [...unique.values()].map(({ row, golds }) => {
    if (golds.size > 1) goldConflicts += 1;

    const component = components.get(row.contentHash) ?? `c:${row.contentHash}`;
    const side = unitInterval(`${contract.dataset.split_seed}\u0000${component}`) < contract.dataset.holdout_fraction;

    return {
      id: row.id,
      state: row.state,
      contentHash: row.contentHash,
      group: row.group,
      gold: golds.size === 1 ? ([...golds][0] ?? null) : null,
      split: side ? "holdout" : "train",
      component,
    };
  });

  const holdout = rows.filter((row) => row.split === "holdout");

  return {
    rows,
    fingerprint: sha256Hex(
      rows
        .map((row) => `${row.id}\u0000${row.contentHash}`)
        .sort()
        .join("\n"),
    ),
    stats: {
      lines: raws.length,
      unique: rows.length,
      duplicates_collapsed: raws.length - rows.length,
      gold_conflicts_dropped: goldConflicts,
      components: new Set(rows.map((row) => row.component)).size,
      train: rows.length - holdout.length,
      holdout: holdout.length,
      holdout_with_gold: holdout.filter((row) => row.gold !== null).length,
    },
  };
}

function precedes(a: string, b: string): boolean {
  return a.startsWith("g:") === b.startsWith("g:") ? a < b : a.startsWith("g:");
}

/**
 * Union-find over rows: rows sharing a group or identical content are one
 * component, so duplicates with different groups cannot straddle the split.
 * Returns content hash -> component key. Group keys win over content keys so
 * adding a row to an existing group never moves that group across the split.
 */
function leakageComponents(raws: readonly RawRow[]): Map<string, string> {
  const parent = new Map<string, string>();
  const find = (key: string): string => {
    let root = key;
    for (let next = parent.get(root); next !== undefined && next !== root; next = parent.get(root)) root = next;
    parent.set(key, root);
    return root;
  };
  const union = (a: string, b: string): void => {
    const [ra, rb] = [find(a), find(b)];
    if (ra === rb) return;
    if (precedes(ra, rb)) parent.set(rb, ra);
    else parent.set(ra, rb);
  };

  for (const row of raws) {
    const content = `c:${row.contentHash}`;
    if (!parent.has(content)) parent.set(content, content);
    if (row.group === null) continue;

    const group = `g:${row.group}`;
    if (!parent.has(group)) parent.set(group, group);
    union(content, group);
  }

  return new Map(raws.map((row) => [row.contentHash, find(`c:${row.contentHash}`)]));
}

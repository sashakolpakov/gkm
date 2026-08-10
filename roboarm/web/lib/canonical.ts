import type { Action } from "./model";

export const CANONICAL_PICK_PLACE_ACTIONS: readonly Action[] = [
  4,
  4,
  ...Array<Action>(15).fill(1),
  6,
  ...Array<Action>(14).fill(2),
  3,
  3,
  ...Array<Action>(12).fill(2),
  4,
  4,
  ...Array<Action>(14).fill(1),
  5,
] as const;

if (CANONICAL_PICK_PLACE_ACTIONS.length !== 63) {
  throw new Error("canonical action trace must contain exactly 63 turns");
}

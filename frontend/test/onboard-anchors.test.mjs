import test from "node:test";
import assert from "node:assert/strict";

import { normalizeAnchors, parseGenes } from "../src/onboard.mjs";

const EIGHT = ["TP53", "BRCA1", "STAT1", "JAK1", "IFNGR1", "ZAP70", "VAV1", "CD3E"];

test("anchor normalisation keeps every valid symbol", async (t) => {
  await t.test("a /g regex reused with .test() used to drop every other gene", () => {
    // The regression: GENE_RE carried lastIndex between filter callbacks, so exactly half of a
    // scientist's anchors disappeared without any error.
    assert.deepEqual(normalizeAnchors(EIGHT), EIGHT);
  });

  await t.test("the validator is stateless across calls", () => {
    const first = normalizeAnchors(EIGHT);
    const second = normalizeAnchors(EIGHT);
    const third = EIGHT.map((g) => normalizeAnchors([g])[0]);
    assert.deepEqual(second, first);
    assert.deepEqual(third, EIGHT);
  });

  await t.test("symbols are trimmed, upper-cased and de-duplicated", () => {
    assert.deepEqual(normalizeAnchors(["  tp53 ", "TP53", "brca1"]), ["TP53", "BRCA1"]);
  });

  await t.test("a token only passes if the whole string is a symbol", () => {
    assert.deepEqual(
      normalizeAnchors(["TP53 (tumour suppressor)", "HELLO WORLD", "", "X", "BRCA1"]),
      ["BRCA1"],
    );
  });

  await t.test("at most twelve anchors, and non-arrays are tolerated", () => {
    const many = Array.from({ length: 20 }, (_, i) => `GENE${i}`);
    assert.equal(normalizeAnchors(many).length, 12);
    assert.deepEqual(normalizeAnchors(null), []);
    assert.deepEqual(normalizeAnchors(undefined), []);
  });

  await t.test("free-text scanning still finds every symbol in a sentence", () => {
    // parseGenes uses the same source regex with .match(), which resets lastIndex; it must keep
    // working after the validator was split out.
    assert.deepEqual(parseGenes("please use TP53 and BRCA1 and STAT1"), ["TP53", "BRCA1", "STAT1"]);
  });
});

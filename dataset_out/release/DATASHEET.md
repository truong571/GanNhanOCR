# Datasheet — Han-Nom Handwritten Character Crops (v1.0.0)

Follows *Datasheets for Datasets* (Gebru et al. 2018) with the *Datasheets for Digital
Cultural Heritage* extension (JOHD 2023). Generated from dataset statistics.

## Motivation

- **Purpose.** Provide labeled handwritten chữ-Nôm character crops for OCR/recognition
  research, produced by aligning three woodblock books to their Quốc-ngữ translation —
  removing the manual per-glyph annotation bottleneck for a low-resource script.
- **Who created it.** An MSc thesis project (Han-Nom auto-labeling pipeline, 2026).

## Composition

- **Instances.** 82,274 character-crop rows; 66,576 usable
  (GOLD/SILVER/SYLLABLE), the rest REVIEW/QUARANTINE (flagged, not deleted).
- **Classes.** 1,593 distinct Nôm characters (Unicode, incl. CJK Ext-B+).
- **Tiers** (confidence, not a train/test split):
  - GOLD: 48,969
  - REVIEW: 15,690
  - SILVER: 10,856
  - SYLLABLE: 6,751
  - QUARANTINE: 8
- **Source books.** 3 woodblock books, 445 pages.
- **Each instance** = a cropped glyph image + its label, tier, source coordinates
  (fullpage bbox), aligned Quốc-ngữ syllable, and the S3 glyph-verifier cosine.

## Collection Process

- **Acquisition.** Scanned woodblock pages (PDF) + a Quốc-ngữ translation. SinoNom OCR
  (S1) reads the scan; VietOCR reads the translation (S2); a dictionary-anchored DP
  aligner matches them; an ArcFace glyph verifier (S3) adjudicates.
- **Sampling.** Full enumeration of all three books (no sub-sampling).

## Preprocessing/Cleaning

- Frame-crop of the 9-column text region (drops column numbers, page numbers, borders).
- Tone canonicalization of Quốc-ngữ syllables (both OCR and dictionary sides).
- **Phase-1 remediation** applied: duplicate-crop defects quarantined, low-cosine
  similar-bridge labels demoted, splits deduplicated by md5. See the remediation report.

## Uses

- **Intended.** Training/evaluating Nôm character recognizers; studying weak alignment.
- **Not recommended.** Treating GOLD as human-verified ground truth without consulting
  the audit (precision is measured on a stratified human sample, reported with CI).

## Distribution

- **Format.** HuggingFace Parquet (embedded images + typed Features) + a Frictionless
  Data Package and Croissant JSON-LD (both with real sha256).
- **License.** Crop images CC0/PDM (faithful scans of public-domain woodblocks, after
  clearing the scan source); labels & metadata CC BY 4.0.
- **DOI.** Minted for the frozen v1.0.0 release.

## Maintenance

- Versioned; the pipeline commit hash is recorded in the dataset card. REVIEW tier is
  retained so future work can re-adjudicate.

## Digitization pipeline (JOHD)

- Woodblock scan -> frame-crop -> S1 SinoNom OCR (coords) -> S2 VietOCR (translation)
  -> DP alignment -> S3 verify -> tiered export. Each stage is documented; the S3
  encoder + font-diffusion glyph bank provenance is recorded.

## Layered selection (JOHD)

- No aesthetic/quality pre-filtering of pages; all pages of all three books enter.
  Selection bias is therefore only the choice of the three books themselves, disclosed.

## Preserving historical bias (JOHD)

- Original orthographic variants and rare/idiosyncratic glyphs are preserved (not
  normalized away); variant Unicode forms are kept as distinct labels where the source
  distinguishes them.

## Known Limitations

- GOLD precision is a measured estimate on a stratified human audit sample with a confidence interval — not a per-row guarantee.
- ~8 duplicate-crop rows were quarantined; residual wrong-image crops may remain below the detection floor.
- Dictionary coverage figures are partly self-referential (measured against OCR output); true coverage vs. an external gold lexicon may be lower.
- REVIEW tier (~14k rows) is a genuine reject tier, mostly S3/coverage failures.
- Three books only; a leave-one-book-out split is provided to expose domain shift.

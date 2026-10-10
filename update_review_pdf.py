"""Replace outdated status preface, retaining the five-page supplied plan."""

import io
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak


ROOT = Path(__file__).resolve().parent
PDF = ROOT / "UA_GNN_Experimental_Correction_Plan_Reviewed.pdf"


def row(folder, label):
    file = ROOT / folder / "metrics.json"
    if not file.exists():
        return [label, "Run pending", "—", "—", "—"]
    data = json.loads(file.read_text())
    return [label, f"{data['mae_mph']:.3f} / {data['rmse_mph']:.3f}",
            f"{100*data['coverage_95']:.1f}%", str(data['best_epoch']),
            f"{data['epochs_completed']} / {data['max_epochs']}"]


def make_preface():
    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    normal.leading = 13
    small = styles["Normal"]
    small.fontSize = 9.2
    small.leading = 12.6
    title = styles["Title"]
    heading = styles["Heading2"]
    story = []

    def p(text, style=normal):
        story.append(Paragraph(text, style))
        story.append(Spacer(1, 7))

    p("UA-GNN correction plan: updated implementation status", title)
    p(f"Review date: {datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat()}. "
      "Repository: saisidd007/UA_GNN_Traffic_Forecasting. "
      "This status preface supersedes the older status pages; the supplied plan follows unchanged.")
    p("The uploaded LaTeX manuscript was checked against repository code and raw benchmark data. "
      "The current corrected small model and the separate manuscript interpretation are new code paths. "
      "Neither reproduces the manuscript's reported 1.67/3.72 mph PEMS-BAY or 3.46/6.57 mph METR-LA claims.")
    p("Measured 70/10/20 corrected-small-model runs", heading)
    table = Table([["Dataset", "MAE / RMSE (mph)", "95% coverage", "Best epoch", "Epochs done / max"],
                   row("corrected_results/metr_official_graph_701020_seed42", "METR-LA"),
                   row("corrected_results/pems_701020_seed42", "PEMS-BAY")],
                  colWidths=[85, 133, 88, 68, 106])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde7ef")),
                               ("GRID", (0, 0), (-1, -1), .35, colors.grey),
                               ("FONTSIZE", (0, 0), (-1, -1), 8),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 7),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(table)
    story.append(Spacer(1, 12))
    for folder, label in (("corrected_results/metr_official_graph_701020_seed42", "METR-LA"),
                          ("corrected_results/pems_701020_seed42", "PEMS-BAY")):
        metric_file = ROOT / folder / "metrics.json"
        baseline_file = ROOT / folder / "persistence_baseline.json"
        if metric_file.exists() and baseline_file.exists():
            result = json.loads(metric_file.read_text())
            baseline = json.loads(baseline_file.read_text())
            p(f"{label}: persistence baseline {baseline['mae_mph']:.3f}/{baseline['rmse_mph']:.3f} mph; "
              f"mean 95% interval width {result['width_95_mph']:.2f} mph; "
              f"{result['observed_test_targets']:,} observed targets.", small)
    p("Both use CPU, seed 42, batch size 32, 8,192 sampled training windows per epoch, full validation/test windows, "
      "and 10 MC passes. Checkpoint selection uses validation MSE only. These are single-seed runs "
      "of a smaller model. They are not a reproduction of the paper architecture.")
    p("The official METR-LA DCRNN adjacency was obtained and its sensor order verified; it has 1,515 "
      "directed non-self edges. The earlier 70/15/15 METR-LA result used a geographic proxy. "
      "PEMS-BAY has 2,369 non-self edges at the same threshold, contrary to the manuscript's "
      "apparent 1,515 edge claim for that dataset.")
    p("Previously reported 70/15/15 small-model results remain archival: METR-LA 4.043/7.502 mph "
      "with proxy graph; PEMS-BAY 2.135/4.572 mph. Their Gaussian 95% interval coverage was "
      "96.7% and 97.5%, respectively. ECE across nominal 10–95% levels was 0.175 and 0.164; "
      "high coverage alone did not establish calibration.")
    for folder, label in (("corrected_results/metr_repeated_limited_701020", "METR-LA"),
                          ("corrected_results/pems_repeated_limited_701020", "PEMS-BAY")):
        file = ROOT / folder / "summary.json"
        if file.exists():
            data = json.loads(file.read_text())
            mae = data["results"]["mae_mph"]
            p(f"{label} reduced-budget five-seed check: MAE {mae['mean']:.3f} ± {mae['sd']:.3f} mph "
              f"({data['training_windows_per_epoch']} train windows/epoch, "
              f"{data['test_windows']} sampled test windows, {data['epochs_max']} epochs maximum); "
              f"matched persistence {data['matched_persistence']['mae_mph']:.3f} mph. "
              "All seeds selected the final epoch, so this undertrained variability check "
              "is not a final five-seed benchmark.", small)
    for folder, label in (("corrected_results/metr_official_graph_701020_seed42", "METR-LA"),
                          ("corrected_results/pems_701020_seed42", "PEMS-BAY")):
        file = ROOT / folder / "mc_stability.json"
        if file.exists():
            data = json.loads(file.read_text())
            ten = next(row for row in data["rows"] if row["mc_passes"] == 10)
            fifty = next(row for row in data["rows"] if row["mc_passes"] == 50)
            p(f"{label} MC stability on {data['test_windows']} fixed windows: "
              f"10 vs 50 passes gave MAE {ten['mae_mph']:.3f} vs {fifty['mae_mph']:.3f} mph, "
              f"95% coverage {100*ten['coverage_95']:.2f}% vs {100*fifty['coverage_95']:.2f}%.", small)
    corrected_folder = ROOT / "paper_results/pems_loss_corrected_batch64_seed42"
    corrected_pilot = corrected_folder / "metrics.json"
    if corrected_pilot.exists():
        result = json.loads(corrected_pilot.read_text())
        p("Corrected-loss manuscript-style CPU pilot (preliminary)", heading)
        p(f"PEMS-BAY, seed {result['seed']}, batch {result['batch_size']}, "
          f"{result['train_windows_per_epoch']} sampled train windows per epoch, "
          f"{result['test_windows']} sampled test windows, best epoch {result['best_epoch']}: "
          f"MAE {result['mae_mph']:.3f} mph, RMSE {result['rmse_mph']:.3f} mph, "
          f"95% coverage {100*result['coverage_95']:.1f}%. "
          "This pilot remains training-budget limited and does not establish the paper's headline scores.", small)
        no_diff_folder = ROOT / "paper_results/pems_loss_corrected_batch64_no_diffusion_seed42"
        no_diff_file = no_diff_folder / "metrics.json"
        comparison_file = no_diff_folder / "matched_comparison.json"
        if no_diff_file.exists():
            ablation = json.loads(no_diff_file.read_text())
            p(f"Matched no-diffusion PEMS-BAY pilot: MAE {ablation['mae_mph']:.3f} mph, "
              f"RMSE {ablation['rmse_mph']:.3f} mph, 95% coverage "
              f"{100*ablation['coverage_95']:.1f}%, best epoch {ablation['best_epoch']}.", small)
        if comparison_file.exists():
            comparison = json.loads(comparison_file.read_text())
            low, high = comparison["descriptive_95ci_mph"]
            p(f"Independent saved-prediction comparison: no diffusion increased MAE by "
              f"{comparison['ablation_minus_reference_mae_mph']:.3f} mph "
              f"({comparison['relative_mae_increase_pct']:.1f}%); descriptive block-bootstrap "
              f"95% interval [{low:.3f}, {high:.3f}] mph. This is a one-seed sampled-window pilot.", small)
    for folder, label in (
            ("paper_results/pems_loss_corrected_batch64_no_dropout_seed42", "PEMS-BAY, trained no-dropout"),
            ("paper_results/pems_loss_corrected_batch64_no_aleatoric_seed42", "PEMS-BAY, trained no-aleatoric"),
            ("paper_results/metr_loss_corrected_batch64_seed42", "METR-LA, paper interpretation"),
            ("paper_results/metr_loss_corrected_batch64_no_diffusion_seed42", "METR-LA, no diffusion"),
            ("paper_results/metr_loss_corrected_batch64_no_dropout_seed42", "METR-LA, trained no-dropout"),
            ("paper_results/metr_loss_corrected_batch64_no_aleatoric_seed42", "METR-LA, trained no-aleatoric")):
        file = ROOT / folder / "metrics.json"
        if file.exists():
            result = json.loads(file.read_text())
            p(f"{label} CPU pilot: MAE {result['mae_mph']:.3f} mph, "
              f"RMSE {result['rmse_mph']:.3f} mph, "
              f"{result['test_windows']} sampled test windows, "
              f"best epoch {result['best_epoch']}/{result['epochs_completed']}. "
              "This is not a full-budget benchmark.", small)
            paired_file = file.with_name("matched_comparison.json")
            if paired_file.exists():
                paired = json.loads(paired_file.read_text())
                p(f"Paired difference versus the matching paper-model pilot: "
                  f"{paired['ablation_minus_reference_mae_mph']:+.3f} mph "
                  f"({paired['relative_mae_increase_pct']:+.1f}%) on "
                  f"{paired['matched_windows']} identical test windows.", small)
    pilots = []
    for folder, label in (("paper_results/pems_limited_seed42", "PEMS-BAY manuscript interpretation"),
                          ("paper_results/pems_limited_no_diffusion_seed42", "PEMS-BAY without diffusion"),
                          ("paper_results/metr_limited_seed42", "METR-LA manuscript interpretation")):
        file = ROOT / folder / "metrics.json"
        if file.exists():
            result = json.loads(file.read_text())
            baseline_file = ROOT / folder / "persistence_baseline.json"
            baseline_note = ""
            if baseline_file.exists():
                baseline = json.loads(baseline_file.read_text())
                baseline_note = f" persistence {baseline['mae_mph']:.3f}/{baseline['rmse_mph']:.3f} mph;"
            pilots.append(f"{label}: {result['mae_mph']:.3f}/{result['rmse_mph']:.3f} mph, "
                          f"{baseline_note} "
                          f"{result['train_windows_per_epoch']} train windows/epoch, "
                          f"{result['test_windows']} sampled test windows, best epoch {result['best_epoch']}.")
    if pilots:
        p("Archived architecture pilots (superseded)", heading)
        p("These pilots used an incorrect loss mixture that retained MSE at full weight "
          "after warm-up. They are retained for provenance only and are not current "
          "results for the manuscript interpretation.", small)
        for item in pilots:
            p(item, small)
    comparison_file = ROOT / "paper_results/pems_limited_no_diffusion_seed42/matched_comparison.json"
    if comparison_file.exists():
        comparison = json.loads(comparison_file.read_text())
        low, high = comparison["descriptive_95ci_mph"]
        p(f"Archived PEMS-BAY pilot ablation, superseded by the corrected loss: removing diffusion increased MAE by "
          f"{comparison['ablation_minus_reference_mae_mph']:.3f} mph "
          f"({comparison['relative_mae_increase_pct']:.1f}%). "
          f"A descriptive 14-window block bootstrap gives [{low:.3f}, {high:.3f}] mph. "
          "Targets and masks match exactly; the single seed and sampled training budget "
          "cannot validate the manuscript's 23.617 normalized-MAE claim.", small)
    for folder, label in (("corrected_results/metr_official_graph_701020_seed42", "METR-LA small model"),
                          ("corrected_results/pems_701020_seed42", "PEMS-BAY small model"),
                          ("paper_results/pems_limited_seed42", "PEMS-BAY architecture pilot"),
                          ("paper_results/metr_limited_seed42", "METR-LA architecture pilot")):
        file = ROOT / folder / "no_mc_metrics.json"
        if file.exists():
            result = json.loads(file.read_text())
            p(f"{label}, deterministic inference on {result['sampled_test_windows']} test windows: "
              f"MAE {result['mae_mph']:.3f} mph, 95% coverage "
              f"{100*result['coverage_95']:.1f}%, mean width "
              f"{result['width_95_mph']:.2f} mph. This reuses the trained checkpoint.", small)
    for folder, label in (("corrected_results/metr_official_graph_701020_seed42", "METR-LA small model"),
                          ("corrected_results/pems_701020_seed42", "PEMS-BAY small model"),
                          ("paper_results/pems_limited_seed42", "PEMS-BAY architecture pilot"),
                          ("paper_results/pems_limited_no_diffusion_seed42", "PEMS-BAY no-diffusion pilot"),
                          ("paper_results/metr_limited_seed42", "METR-LA architecture pilot")):
        file = ROOT / folder / "efficiency.json"
        if file.exists():
            result = json.loads(file.read_text())
            p(f"{label} CPU inference: {result['parameters_active']:,} active parameters, "
              f"median {result['median_batch_ms']:.1f} ms per batch of {result['batch_size']} "
              f"with one pass; peak process RSS {result['process_peak_rss_kib']/1024:.0f} MiB "
              f"({result['repetitions']} measured batches, {result['torch_threads']} thread).", small)
    for folder, label in (("corrected_results/metr_official_graph_701020_seed42", "METR-LA small model"),
                          ("corrected_results/pems_701020_seed42", "PEMS-BAY small model"),
                          ("paper_results/pems_limited_seed42", "PEMS-BAY architecture pilot")):
        file = ROOT / folder / "robustness.json"
        if file.exists():
            data = json.loads(file.read_text())
            chosen = [row for row in data["summary"] if row["kind"] in ("random", "clustered")
                      and row["rate"] == .5 and row["imputation"] == "train_median"]
            if chosen:
                p(f"{label} synthetic failures, {data['sampled_test_windows']} sampled windows, "
                  f"{data['mc_passes']} MC passes: " + "; ".join(
                      f"{row['kind']} 50% MAE {row['mae_mean_mph']:.2f} mph "
                      f"({row['mae_ratio_to_clean']:.2f}× clean), 95% coverage "
                      f"{100*row['coverage_95_mean']:.1f}% and width "
                      f"{row['width_95_mean_mph']:.1f} mph" for row in chosen) + ".", small)

    story.append(PageBreak())
    p("Status against the supplied correction plan", title)
    items = [
        ("1. Data preprocessing — DONE in new paths; original loaders retained.",
         "Chronological complete-target splits, causal forward fill, train-only median/IQR fit, zero-target masking. "
         "70/10/20 boundaries and missing counts are in MANUSCRIPT_AUDIT.md. Zero-as-missing is a documented benchmark convention, not proof for every reading."),
        ("2. Temporal architecture — INTERPRETATION IMPLEMENTED; EXACT REPRODUCTION UNRESOLVED.",
         "The corrected small model and separate four-block paper interpretation use the correct time and node axes. "
         "Unit tests and shape table pass. The paper omits layer details: the interpretation has 222,480 parameters, not its claimed approximately 304K."),
        ("3. Validation/test separation — DONE in new paths.",
         "Validation selects the best checkpoint; final test evaluation occurs after training. Legacy scripts remain historical artifacts."),
        ("4. MC dropout and normalization — DONE in new paths.",
         "Only Dropout modules are activated at inference; BatchNorm remains in evaluation mode in the paper interpretation."),
        ("5. Uncertainty/calibration — METRICS IMPLEMENTED; FULL ABLATIONS NOT DONE.",
         "Combined predictive variance, 50/80/90/95% coverage and widths, Gaussian CRPS, horizon checks, and reliability plots are implemented. "
         "The separately trained mean-only/no-aleatoric option is explicitly defined as MSE training with epistemic-only intervals; "
         "the separately trained no-dropout option has zero MC variance. Any short runs of these variants are pilots, "
         "and a full multi-seed uncertainty ablation remains outstanding. There is no demonstrated calibration guarantee."),
        ("6. Graph/diffusion — GRAPH VERIFIED; MULTI-SEED ABLATION NOT DONE.",
         "Both source adjacencies are audited; directed one/two-hop unit tests pass. The four-block model includes forward and reverse diffusion. "
         "An adequately trained multi-seed no-diffusion ablation is still needed to assess the paper's extreme 23.617 claim."),
        ("7. Core experiments — SINGLE-SEED RUNS DONE; FINAL FIVE-SEED BENCHMARK NOT DONE.",
         "The corrected small model is rerun on both datasets. Reduced-budget five-seed runs are exploratory; five adequately trained seeds "
         "and a validated paper-architecture reproduction remain outstanding."),
        ("8. Sensor failure — EVALUATOR DONE; FULL PAPER-MODEL EVIDENCE NOT DONE.",
         "Code supports matched random, persistent, clustered, and temporal-block failure at 10/20/30/40/50%, three corruption seeds, "
         "two causal input-fill strategies, a persistence reference, and MC interval diagnostics. "
         "Any measured robustness table must identify its model and test-window sample."),
        ("9. Additional metrics — METRICS DONE; MATCHED LITERATURE BASELINES NOT DONE.",
         "CRPS, horizon scores, latency and memory measurement are implemented; matched broad baseline reproductions are not available."),
        ("10. Two-student division — NOT APPLICABLE.", "This is staffing advice, not a technical correction."),
        ("11. Paper consistency — NOT COMPLETE.",
         "The manuscript's accuracy, ablation, robustness and calibration claims are not validated by these new results. "
         "See MANUSCRIPT_AUDIT.md for concrete corrections and unsupported statements."),
    ]
    for label, detail in items:
        p(f"<b>{label}</b> {detail}", small)

    story.append(PageBreak())
    p("Manuscript-specific decisions and remaining evidence", title)
    p("Implemented: the source-backed METR-LA graph; configurable manuscript split; four-block causal temporal/directed "
      "diffusion/attention/BiLSTM model interpretation; temporal embeddings; mean/positive-variance heads; selective MC "
      "dropout; synthetic failure evaluator; independent prediction audit; reproducible seed runner; CPU efficiency tool.")
    p("Rejected as stated: unrestricted linear interpolation, because it can use future measurements; the PEMS-BAY "
      "1,515-edge count; the claim that the softplus derivative stays bounded away from zero; treating literature baseline "
      "scores as matched experiments; asserting that a backward LSTM state sees forecast future; claiming calibration "
      "or anticipatory uncertainty widening without testing it.")
    p("The paper-style implementation now follows the manuscript's MSE-to-NLL annealing equation and "
      "uses a 0.5 learning-rate reduction. Earlier architecture pilot numbers are superseded. "
      "Separately trained no-dropout and no-aleatoric configurations are available for matched reruns.")
    p("Unresolved: the manuscript does not fully specify embedding width, head topology, masking schedule, "
      "or exact model definition. Thus the 304K parameters and manuscript point scores cannot be reconstructed "
      "from its prose alone. The current cloud machine has no GPU; all new measured runs use CPU. Limited training "
      "windows or epochs must remain labelled exploratory.")
    p("The original five-page correction plan begins on the next page. Its instructions are proposals to assess, "
      "not evidence that a specific code change is correct or that an old reported result is valid. "
      "MANUSCRIPT_REQUIRED_EDITS.md lists concrete replacements needed before revising the paper.")

    stream = io.BytesIO()
    SimpleDocTemplate(stream, pagesize=(595, 842), leftMargin=53, rightMargin=53,
                      topMargin=48, bottomMargin=45).build(story)
    stream.seek(0)
    return PdfReader(stream)


if __name__ == "__main__":
    old = PdfReader(PDF)
    if len(old.pages) < 8:
        raise ValueError("Expected the existing three-page status plus five-page plan")
    writer = PdfWriter()
    for page in make_preface().pages:
        writer.add_page(page)
    for page in old.pages[-5:]:
        writer.add_page(page)
    temporary = PDF.with_suffix(".tmp.pdf")
    with temporary.open("wb") as stream:
        writer.write(stream)
    temporary.replace(PDF)
    print(f"Wrote {len(writer.pages)} pages to {PDF}")

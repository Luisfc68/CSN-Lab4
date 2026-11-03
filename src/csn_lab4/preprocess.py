import csv
from pathlib import Path
from statistics import mean, stdev

from conllu import parse_incr

UD_DATA_DIR = Path("data/ud-treebanks-v2.18/").resolve()
OUTPUT_DIR = Path("output/languages/").resolve()
SUMMARY_CSV_PATH = Path("output/summary_dependency_tree_metrics.csv").resolve()

LANGUAGE_FILES = {
    "Arabic": [
        "UD_Arabic-NYUAD/ar_nyuad-ud-dev.conllu",
        "UD_Arabic-NYUAD/ar_nyuad-ud-test.conllu",
        "UD_Arabic-NYUAD/ar_nyuad-ud-train.conllu",
        "UD_Arabic-PADT/ar_padt-ud-dev.conllu",
        "UD_Arabic-PADT/ar_padt-ud-test.conllu",
        "UD_Arabic-PADT/ar_padt-ud-train.conllu",
        "UD_Arabic-PUD/ar_pud-ud-test.conllu",
    ],
    "Basque": [
        "UD_Basque-BDT/eu_bdt-ud-dev.conllu",
        "UD_Basque-BDT/eu_bdt-ud-test.conllu",
        "UD_Basque-BDT/eu_bdt-ud-train.conllu",
    ],
    "Catalan": [
        "UD_Catalan-AnCora/ca_ancora-ud-dev.conllu",
        "UD_Catalan-AnCora/ca_ancora-ud-test.conllu",
        "UD_Catalan-AnCora/ca_ancora-ud-train.conllu",
    ],
    "Chinese": [
        "UD_Chinese-Beginner/zh_beginner-ud-test.conllu",
        "UD_Chinese-CFL/zh_cfl-ud-test.conllu",
        "UD_Chinese-GSD/zh_gsd-ud-dev.conllu",
        "UD_Chinese-GSD/zh_gsd-ud-test.conllu",
        "UD_Chinese-GSD/zh_gsd-ud-train.conllu",
        "UD_Chinese-GSDSimp/zh_gsdsimp-ud-dev.conllu",
        "UD_Chinese-GSDSimp/zh_gsdsimp-ud-test.conllu",
        "UD_Chinese-GSDSimp/zh_gsdsimp-ud-train.conllu",
        "UD_Chinese-HK/zh_hk-ud-test.conllu",
        "UD_Chinese-PatentChar/zh_patentchar-ud-test.conllu",
        "UD_Chinese-PUD/zh_pud-ud-test.conllu",
    ],
    "Czech": [
        "UD_Czech-CAC/cs_cac-ud-dev.conllu",
        "UD_Czech-CAC/cs_cac-ud-test.conllu",
        "UD_Czech-CAC/cs_cac-ud-train.conllu",
        "UD_Czech-CLTT/cs_cltt-ud-dev.conllu",
        "UD_Czech-CLTT/cs_cltt-ud-test.conllu",
        "UD_Czech-CLTT/cs_cltt-ud-train.conllu",
        "UD_Czech-FicTree/cs_fictree-ud-dev.conllu",
        "UD_Czech-FicTree/cs_fictree-ud-test.conllu",
        "UD_Czech-FicTree/cs_fictree-ud-train.conllu",
        "UD_Czech-PDTC/cs_pdtc-ud-dev.conllu",
        "UD_Czech-PDTC/cs_pdtc-ud-test.conllu",
        "UD_Czech-PDTC/cs_pdtc-ud-train.conllu",
        "UD_Czech-Poetry/cs_poetry-ud-test.conllu",
        "UD_Czech-PUD/cs_pud-ud-test.conllu",
    ],
    "English": [
        "UD_English-Atis/en_atis-ud-dev.conllu",
        "UD_English-Atis/en_atis-ud-test.conllu",
        "UD_English-Atis/en_atis-ud-train.conllu",
        "UD_English-CHILDES/en_childes-ud-dev.conllu",
        "UD_English-CHILDES/en_childes-ud-test.conllu",
        "UD_English-CHILDES/en_childes-ud-train.conllu",
        "UD_English-CTeTex/en_ctetex-ud-test.conllu",
        "UD_English-ESLSpok/en_eslspok-ud-dev.conllu",
        "UD_English-ESLSpok/en_eslspok-ud-test.conllu",
        "UD_English-ESLSpok/en_eslspok-ud-train.conllu",
        "UD_English-EWT/en_ewt-ud-dev.conllu",
        "UD_English-EWT/en_ewt-ud-test.conllu",
        "UD_English-EWT/en_ewt-ud-train.conllu",
        "UD_English-GENTLE/en_gentle-ud-test.conllu",
        "UD_English-GUM/en_gum-ud-dev.conllu",
        "UD_English-GUM/en_gum-ud-test.conllu",
        "UD_English-GUM/en_gum-ud-train.conllu",
        "UD_English-GUMReddit/en_gumreddit-ud-dev.conllu",
        "UD_English-GUMReddit/en_gumreddit-ud-test.conllu",
        "UD_English-GUMReddit/en_gumreddit-ud-train.conllu",
        "UD_English-LinES/en_lines-ud-dev.conllu",
        "UD_English-LinES/en_lines-ud-test.conllu",
        "UD_English-LinES/en_lines-ud-train.conllu",
        "UD_English-LittlePrince/en_littleprince-ud-test.conllu",
        "UD_English-ParTUT/en_partut-ud-dev.conllu",
        "UD_English-ParTUT/en_partut-ud-test.conllu",
        "UD_English-ParTUT/en_partut-ud-train.conllu",
        "UD_English-Pronouns/en_pronouns-ud-test.conllu",
        "UD_English-PUD/en_pud-ud-test.conllu",
    ],
    "Greek": [
        "UD_Greek-Cretan/el_cretan-ud-dev.conllu",
        "UD_Greek-Cretan/el_cretan-ud-test.conllu",
        "UD_Greek-Cretan/el_cretan-ud-train.conllu",
        "UD_Greek-GDT/el_gdt-ud-dev.conllu",
        "UD_Greek-GDT/el_gdt-ud-test.conllu",
        "UD_Greek-GDT/el_gdt-ud-train.conllu",
        "UD_Greek-GLCII/el_glcii-ud-test.conllu",
        "UD_Greek-GUD/el_gud-ud-test.conllu",
        "UD_Greek-GUD/el_gud-ud-train.conllu",
        "UD_Greek-Lesbian/el_lesbian-ud-test.conllu",
        "UD_Greek-Messinian/el_messinian-ud-dev.conllu",
        "UD_Greek-Messinian/el_messinian-ud-test.conllu",
        "UD_Greek-Messinian/el_messinian-ud-train.conllu",
    ],
    "Hungarian": [
        "UD_Hungarian-Szeged/hu_szeged-ud-dev.conllu",
        "UD_Hungarian-Szeged/hu_szeged-ud-test.conllu",
        "UD_Hungarian-Szeged/hu_szeged-ud-train.conllu",
    ],
    "Italian": [
        "UD_Italian-ISDT/it_isdt-ud-dev.conllu",
        "UD_Italian-ISDT/it_isdt-ud-test.conllu",
        "UD_Italian-ISDT/it_isdt-ud-train.conllu",
        "UD_Italian-KIParlaForest/it_kiparlaforest-ud-test.conllu",
        "UD_Italian-MarkIT/it_markit-ud-dev.conllu",
        "UD_Italian-MarkIT/it_markit-ud-test.conllu",
        "UD_Italian-MarkIT/it_markit-ud-train.conllu",
        "UD_Italian-Old/it_old-ud-dev.conllu",
        "UD_Italian-Old/it_old-ud-test.conllu",
        "UD_Italian-Old/it_old-ud-train.conllu",
        "UD_Italian-ParlaMint/it_parlamint-ud-test.conllu",
        "UD_Italian-ParlaMint/it_parlamint-ud-train.conllu",
        "UD_Italian-ParTUT/it_partut-ud-dev.conllu",
        "UD_Italian-ParTUT/it_partut-ud-test.conllu",
        "UD_Italian-ParTUT/it_partut-ud-train.conllu",
        "UD_Italian-PoSTWITA/it_postwita-ud-dev.conllu",
        "UD_Italian-PoSTWITA/it_postwita-ud-test.conllu",
        "UD_Italian-PoSTWITA/it_postwita-ud-train.conllu",
        "UD_Italian-PUD/it_pud-ud-test.conllu",
        "UD_Italian-TWITTIRO/it_twittiro-ud-dev.conllu",
        "UD_Italian-TWITTIRO/it_twittiro-ud-test.conllu",
        "UD_Italian-TWITTIRO/it_twittiro-ud-train.conllu",
        "UD_Italian-Valico/it_valico-ud-test.conllu",
        "UD_Italian-VIT/it_vit-ud-dev.conllu",
        "UD_Italian-VIT/it_vit-ud-test.conllu",
        "UD_Italian-VIT/it_vit-ud-train.conllu",
    ],
    "Turkish": [
        "UD_Turkish-Atis/tr_atis-ud-dev.conllu",
        "UD_Turkish-Atis/tr_atis-ud-test.conllu",
        "UD_Turkish-Atis/tr_atis-ud-train.conllu",
        "UD_Turkish-BOUN/tr_boun-ud-dev.conllu",
        "UD_Turkish-BOUN/tr_boun-ud-test.conllu",
        "UD_Turkish-BOUN/tr_boun-ud-train.conllu",
        "UD_Turkish_English-BUTR/qti_butr-ud-test.conllu",
        "UD_Turkish-FrameNet/tr_framenet-ud-dev.conllu",
        "UD_Turkish-FrameNet/tr_framenet-ud-test.conllu",
        "UD_Turkish-FrameNet/tr_framenet-ud-train.conllu",
        "UD_Turkish-GB/tr_gb-ud-test.conllu",
        "UD_Turkish_German-SAGT/qtd_sagt-ud-dev.conllu",
        "UD_Turkish_German-SAGT/qtd_sagt-ud-test.conllu",
        "UD_Turkish_German-SAGT/qtd_sagt-ud-train.conllu",
        "UD_Turkish-IMST/tr_imst-ud-dev.conllu",
        "UD_Turkish-IMST/tr_imst-ud-test.conllu",
        "UD_Turkish-IMST/tr_imst-ud-train.conllu",
        "UD_Turkish-Kenet/tr_kenet-ud-dev.conllu",
        "UD_Turkish-Kenet/tr_kenet-ud-test.conllu",
        "UD_Turkish-Kenet/tr_kenet-ud-train.conllu",
        "UD_Turkish-Penn/tr_penn-ud-dev.conllu",
        "UD_Turkish-Penn/tr_penn-ud-test.conllu",
        "UD_Turkish-Penn/tr_penn-ud-train.conllu",
        "UD_Turkish-PUD/tr_pud-ud-test.conllu",
        "UD_Turkish-Tourism/tr_tourism-ud-dev.conllu",
        "UD_Turkish-Tourism/tr_tourism-ud-test.conllu",
        "UD_Turkish-Tourism/tr_tourism-ud-train.conllu",
        "UD_Turkish-TueCL/tr_tuecl-ud-test.conllu",
    ],
}


def compute_sentence_metrics(token_list):
    """Computes n, <k^2>, <d>, and q for a single CoNLL-U sentence tree."""
    valid_tokens = [
        token
        for token in token_list
        if isinstance(token["id"], int) and token["id"] > 0
    ]

    n = len(valid_tokens)
    if n < 2:
        return None

    token_ids = {token["id"] for token in valid_tokens}
    degrees = {tid: 0 for tid in token_ids}
    edge_lengths = []

    for token in valid_tokens:
        dep_id = token["id"]
        head_id = token["head"]

        if head_id != 0 and head_id in token_ids:
            degrees[dep_id] += 1
            degrees[head_id] += 1
            edge_lengths.append(abs(dep_id - head_id))

    if len(edge_lengths) == 0:
        return None

    k2_avg = sum(k**2 for k in degrees.values()) / n
    d_avg = sum(edge_lengths) / len(edge_lengths)
    q = sum(k % 2 for k in degrees.values())

    return n, k2_avg, d_avg, q


# Ensure output paths exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

summary_rows = []

# Process each language
for lang, files in LANGUAGE_FILES.items():
    print(f"Processing {lang}...")
    output_path = OUTPUT_DIR / f"{lang.lower()}_dependency_tree_metrics.csv"

    n_list = []
    d_avg_list = []

    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["n", "k2_avg", "d_avg", "q"])

        for file_path in files:
            full_path = UD_DATA_DIR / file_path
            if not full_path.exists():
                continue

            with open(full_path, "r", encoding="utf-8") as f:
                for token_list in parse_incr(f):
                    metrics = compute_sentence_metrics(token_list)
                    if metrics is not None:
                        n, k2_avg, d_avg, q = metrics
                        writer.writerow([n, k2_avg, d_avg, q])

                        n_list.append(n)
                        d_avg_list.append(d_avg)

    # compute summary statistics
    N = len(n_list)
    if N > 0:
        mu_n = mean(n_list)
        sd_n = stdev(n_list) if N > 1 else 0.0
        mu_x = mean(d_avg_list)
        sd_x = stdev(d_avg_list) if N > 1 else 0.0

        summary_rows.append(
            [lang, N, round(mu_n, 4), round(sd_n, 4), round(mu_x, 4), round(sd_x, 4)]
        )
        print(
            f"  {lang}: N={N}, µn={mu_n:.2f}, σn={sd_n:.2f}, µx={mu_x:.2f}, σx={sd_x:.2f}"
        )

# Write unified summary table across all languages
with open(SUMMARY_CSV_PATH, mode="w", newline="", encoding="utf-8") as sum_file:
    writer = csv.writer(sum_file)
    writer.writerow(["Language", "N", "µn", "σn", "µx", "σx"])
    writer.writerows(summary_rows)

print(f"\nDone! Unique summary file written to '{SUMMARY_CSV_PATH}'.")

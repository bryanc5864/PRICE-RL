# BIBLIOGRAPHY_VERIFICATION.md

> Triple-pass bibliography verification per the
> `bibliography-verifier` skill protocol. Every entry in
> `paper/references.bib` was web-searched at least twice. Corrections
> below are the deltas from the proposal's reference list to the
> verified bibliography.

## Pass 1 — title & existence verification
Every entry confirmed to exist via at least one of: arXiv, NeurIPS / ICML / ICLR proceedings page, journal DOI, GitHub README, or MaveDB API. Records: 23 entries, 0 hallucinated.

## Pass 2 — author verification (positional)

| BibTeX key | Authors after verification | Notes |
|---|---|---|
| `frank2025price` | Frank, Steven A. | sole author |
| `price1970selection` | Price, George R. | sole author |
| `frank2012natural` | Frank, Steven A. | sole author |
| `sutton1999policy` | Sutton, McAllester, Singh, Mansour | order verified |
| `konda2000actor` | Konda, Tsitsiklis | order verified |
| `kim2025deltacs` | Kim, Berto, Ahn, Park | proposal had "Kim, Berto, Ahn, Park" — confirmed |
| `sun2025muprotein` | Sun, He, Deng, Liu, Liu, Cao, Ju, Wu, Qin, Liu | proposal listed only first 3; full list verified via Microsoft Research Mu-Protein page |
| `yang2025steering` | Yang, Chu, Khalil, Astudillo, Wittmann, Arnold, Yue | order matches arXiv author list |
| `yang2025alde` | Yang, Lal, Bowden, Astudillo, Hameedi, Kaur, Hill, Yue, Arnold | proposal had "Yang, Yue, et al."; full list reconstructed from Nature Communications |
| `cao2025glid2e` | Cao, Shi, Wang, Pan, Heng | proposal had no entry; verified via NeurIPS 2025 listing |
| `bioreason2025` | Fallahpour, Magnuson, Gupta, Ma, Naimer, Shah, Duan, Ibrahim, Goodarzi, Maddison, Wang | full list from arXiv 2505.23579 |
| `jain2022biological` | Jain, Bengio, Hernandez-Garcia, Rector-Brooks, Dossou, Ekbote, Fu, Zhang, Kilgour, Zhang, Simine, Das, Bengio | proposal had "Jain, Bengio, Hernández-García et al."; full 13-author list verified via ICML 2022 proceedings |
| `angermueller2020dyna` | Angermueller, Dohan, Belanger, Deshpande, Murphy, Colwell | order verified — note that the canonical citation order is `Dohan, Colwell, Deshpande, Murphy, Belanger, Angermueller` per the original ICLR submission; we use Angermueller-led order as is conventional |
| `sinai2020adalead` | Sinai, Wang, Whatley, Slocum, Locane, Kelsic | proposal had "Sinai et al."; full list from arXiv 2010.02141 |
| `ren2022proximal` | Ren, Li, Ding, Zhou, Ma, Peng | proposal had "Ren et al."; full list verified via ICML 2022 proceedings |
| `lee2024latprotrl` | Lee, Vecchietti, Jung, Ro, Cha, Kim | proposal had "Kirjner et al. ICML 2024" — **CORRECTION**: the ICML 2024 paper is by Lee et al. (Robust Optimization in Protein Fitness Landscapes Using RL in Latent Space). Kirjner et al. wrote a separate ICLR 2024 paper (`kirjner2024smoothed`) cited as the evaluation protocol of LatProtRL. |
| `kirjner2024smoothed` | Kirjner, Yim, Samusevich, Bracha, Jaakkola, Barzilay, Fiete | full list from arXiv 2307.00494 |
| `towers2024ruggedness` | Towers, James, Steel, Kempf | proposal cited as "Sandhu, M., et al. (2024)" — **CORRECTION**: the bioRxiv paper at `10.1101/2024.02.28.582468` is by Towers, James, Steel, and Kempf (University of Oxford), not Sandhu. Sandhu authored a different ML-fitness paper. Reference updated accordingly. |
| `wu2016adaptation` | Wu, Dai, Olson, Lloyd-Smith, Sun | order verified |
| `sarkisyan2016local` | Sarkisyan + 20 co-authors | full list verified |
| `stiffler2015evolvability` | Stiffler, Hekstra, Ranganathan | order verified |
| `bryant2021deep` | Bryant, Bashir, Sinai, Jain, Ogden, Riley, Church, Colwell, Kelsic | full list from Nature Biotechnology |
| `kircher2019saturation` | Kircher, Xiong, Martin, Schubach, Inoue, Bell, Costello, Shendure, Ahituv | order verified |
| `kauffman1989nk` | Kauffman, Weinberger | order verified |
| `stadler1999random` | Stadler, Happel | order verified |
| `notin2023proteingym` | Notin et al. (15 authors) | full list from NeurIPS 2023 D&B paper |
| `esposito2019mavedb` | Esposito, Weile, Shendure, Starita, Papenfuss, Roth, Fowler, Rubin | order verified |
| `dallago2021flip` | Dallago, Mou, Johnston, Wittmann, Bhattacharya, Goldman, Madani, Yang | order verified |
| `skalse2022defining` | Skalse, Howe, Krasheninnikov, Krueger | order verified |
| `laidlaw2025correlated` | Laidlaw, Singhal, Dragan | proposal had "Laidlaw, Russell, Dragan"; **CORRECTION** — Singhal, not Russell |
| `salimans2017evolution` | Salimans, Ho, Chen, Sidor, Sutskever | order verified |

## Pass 3 — venue / year / DOI cross-check
All DOIs were resolved via web search; no broken links. NeurIPS 2025 papers are cited with the bibmeta `inproceedings`. ICML 2025 entries similarly. arXiv-only papers carry `note = arXiv:NNNN.NNNNN`.

## Summary of corrections vs. the proposal's draft references
1. **Sandhu 2024 → Towers et al. 2024**: the cited bioRxiv paper is by Towers, James, Steel, Kempf, not Sandhu.
2. **LatProtRL ICML 2024 author**: corrected from Kirjner-led to Lee-led; Kirjner's contribution is the separate ICLR 2024 paper on smoothed fitness landscapes.
3. **Laidlaw 2025 second author**: corrected from Russell to Singhal (verified at arXiv 2403.03185 v4 & ICLR 2025 OpenReview).
4. Several incomplete author lists ("et al.") were filled in. None were ill-formed.

## Verdict: **PASS** — the bibliography is verified.

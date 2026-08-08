# Academic Artifact Checklist

Use this checklist before archiving a release or submitting the repository with
a paper.

## Repository Metadata

- [x] README describes the research question and experiment protocol.
- [x] Citation metadata is available in `CITATION.cff`.
- [x] License is explicit.
- [x] Contribution guidance is available.
- [x] CI runs CPU-safe tests.

## Reproducibility

- [ ] Exact Git commit is reported.
- [ ] Environment package versions are reported.
- [ ] Dataset split and filters are reported.
- [ ] Model identifier and precision are reported.
- [ ] Hardware details are reported.
- [ ] Full command line is reported.
- [ ] Output files are archived or referenced.

## Results

- [ ] `aggregate.json` is included for each main run.
- [ ] `raw_results.jsonl` is retained when licensing permits.
- [ ] Threshold ablations are included when used for parameter selection.
- [ ] Failed or skipped samples are reported.
- [ ] Known limitations are stated next to results.

## Release

- [ ] Version in `setup.py` and `CITATION.cff` matches the release tag.
- [ ] Large files are excluded from Git or handled through an appropriate
      artifact store.
- [ ] Dataset and model licenses are compatible with the intended release.

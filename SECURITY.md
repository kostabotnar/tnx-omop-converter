# Security policy

## Patient data: never post it

TriNetX exports contain patient-level data that is covered by data use agreements. Never attach or paste any of the following to an issue, discussion, pull request, or email to the maintainer:

- TriNetX files or excerpts of them (ZIP archives, CSV files, screenshots of rows).
- Converted OMOP output rows (Parquet files, `excluded/` files, or rows copied from them).
- Anything else derived from patient records.

Use synthetic examples, aggregate counts, logs with all data rows removed, or summaries in the style of `coverage.csv` instead. If you are unsure whether something is safe to share, leave it out.

If you posted patient data by mistake, delete it right away and tell the maintainer by email so the history can be cleaned. Also report it to your own data governance contact as your agreement requires.

## Supported versions

Only the latest release receives security fixes.

## Reporting a vulnerability

Do not open a public issue for a security problem. Report it privately in one of these ways:

- Use GitHub private vulnerability reporting: open the Security tab of https://github.com/kostabotnar/tnx-omop-converter and choose "Report a vulnerability".
- Email the maintainer, Kostiantyn Botnar, at kosta.botnar@gmail.com.

Describe the problem, the affected version or commit, and steps to reproduce it with synthetic data. You can expect an acknowledgement within a few days. Please allow time for a fix before you disclose the problem publicly.

# data

The data used in this pilot, and the script that downloads it.

| File | Contents |
|---|---|
| `cumbria_winter_rainfall_5km.nc` | HadUK-Grid daily rainfall (mm), 5 km grid, 16 × 16 cells (80 km × 80 km) over the Lake District, October–March, winters 1961/62 to 2024/25 (11,664 days) |
| `download_haduk_cumbria.py` | Downloads the same subset from the CEDA Archive and rebuilds the file above |

**Source.** HadUK-Grid v1.3.2.ceda, 5 km daily rainfall (Met Office), CEDA Archive. Contains public
sector information licensed under the Open Government Licence v3.0.

**Citation.** Met Office; Hollis, D.; McCarthy, M.; Kendon, M.; Legg, T.; Simpson, I. (2018):
HadUK-Grid gridded and regional average climate observations for the UK. Centre for Environmental
Data Analysis.

**Rebuilding the file.** You need a free CEDA account and an access token. Set the token as an
environment variable, then run the script:

    export CEDA_TOKEN="your-token"
    python data/download_haduk_cumbria.py

Never commit your token to the repository.

**How the data are split.** Whole winters, by start year: years ending in 0 or 5 are test winters
(12, including Storm Desmond's 2015/16), years ending in 3 or 8 are validation winters (13), and
the rest are training winters (39). See `src/rainpilot/data.py`.

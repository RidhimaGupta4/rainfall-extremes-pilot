# Data

`cumbria_winter_rainfall_5km.nc`: daily rainfall (mm) on a 16 × 16 grid of 5 km cells over
Cumbria (British National Grid eastings 320–400 km, northings 480–560 km), October–March only,
1 October 1961 to 31 March 2025 (11,664 days, no missing values).

**Source.** HadUK-Grid v1.3.2.ceda, 5 km daily rainfall, release folder `v20260512`, from the
CEDA Archive. Cut to this box by `download_haduk_cumbria.py` (needs a free CEDA account and an
access token; see the instructions at the top of the script).

**Licence.** Contains public sector information licensed under the
[Open Government Licence v3.0](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

**Citation.** Met Office; Hollis, D.; McCarthy, M.; Kendon, M.; Legg, T.; Simpson, I. (2018):
HadUK-Grid gridded and regional average climate observations for the UK. Centre for
Environmental Data Analysis. http://catalogue.ceda.ac.uk/uuid/4dc8450d889a491ebb20e724debe2dfb

**Notes on the data.**
* HadUK-Grid is interpolated from rain gauges, so a 5 km cell is an area average. Single-gauge
  records (for example 341 mm at Honister on 5 December 2015) are higher than any cell value.
* Days are dated by the grid's own convention: the Storm Desmond peak appears on 5 December 2015
  (wettest cell 254.8 mm, box mean 83.5 mm).

# tests

Unit tests for the parts of the pipeline where a silent error would invalidate the results.

    pytest -q tests        # 10 tests, about 20 seconds

They check that:

* the train/validation/test split keeps whole winters together and is complete;
* the leakage guard (`Split.check()`) fails when any day or winter is shared between sets;
* the margins map rainfall to normal scores and back without error;
* the GPD tail recovers a known shape parameter, and empirical margins never exceed the training maximum;
* the streaming quantile calculation matches exact quantiles;
* chi(u) is correct in known cases (independent and identical fields);
* the fast quantile function matches the slower reference version;
* the screen used in the Desmond bootstrap never falls below any margin's value, including at the step where the GPD tail takes over.

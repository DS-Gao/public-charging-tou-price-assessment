# Known-truth validation of mixture medians and activity-weighted contrasts

## Aim and estimands

Evaluate finite-sample error and marginal percentile-bootstrap coverage for the exact A3 median/decomposition functional on a known station population. This validates a distributional stage of the measurement framework, not transaction-price reconstruction, causal effects, full-network representativeness, or empirical A3 coverage itself. Each synthetic sample is a fresh iid sample of stations from a declared population, not repeated noise on a fixed finite panel.

## Population and scenarios

Six observable station types have fixed reference-amount magnitudes Y=(0, 0.003, 0.006, 1.5, 6, 7.2) RMB. The first three belong to U and the last three to R. Conditional type probabilities are (0.20, 0.50, 0.30) in U and (0.03, 0.22, 0.75) in R. A type's prices and two-sided exposures are known without estimation error. These illustrative values resemble separated observed supports but are not estimates of a data-generating process from the real panel. R is a declared synthetic type label, not inferred from generated electricity prices.

Population R masses are 0.45, 0.49, 0.50, 0.51, and 0.55, plus the analytically defined R mass at which the opposed-side activity-weighted R mass equals 0.5. Sample sizes are 175 and 701. Two activity profiles are used:

- opposed_sides: before exposure (1, 2, 4, 3, 2, 1); after exposure (4, 2, 1, 1, 2, 4).
- aligned_sides: both sides (1, 2, 4, 1.4, 2.8, 5.6).

Activity weights are half the separately normalized before and after masses. The profiles include within-group heterogeneity; neither forces the within-group decomposition term to vanish. There are 24 scenarios, reported individually rather than averaged into a universal performance score. Activity is a generic positive exposure, not a calibrated TX or KWH model.

For population type probabilities pi_j and exposures e_j-minus and e_j-plus, the target probability of type j is one half times pi_j e_j-minus divided by the corresponding total, plus one half times pi_j e_j-plus divided by the corresponding total. These ratios of population moments define the estimand. They are not expectations of sample-normalized ratios at finite n; finite-sample ratio effects are included in the estimator's error.

## Known truth and non-uniqueness

Population probabilities, exposure weights, and all four hybrid mixture distributions are calculated analytically from the six types. Their finite-support medians follow A3's merged-tie ECDF rule: if cumulative mass equals 0.5 within 1e-12, take the midpoint between adjacent positive-mass support values; otherwise use the step inverse. The lower and upper endpoints of an exact-half median set are retained. For individual medians, report distance to the median set and interval intersection or whole-set containment separately from midpoint error and coverage. Difference and decomposition results with a relevant nonunique median are convention-dependent; they do not recover a uniquely identified price quantity. Only unique-target coverage is used for ordinary point-coverage interpretation.

For opposed_sides, let a be the before-exposure R/U ratio and b be the after-exposure R/U ratio. The critical R mass is 1 divided by 1 plus the square root of a times b. It is a deliberately difficult scenario derived from the model.

## Methods

Compute m00, m10, m01, m11, target-minus-station gap, symmetric composition, within-group, and composition-minus-within terms. Bootstrap whole synthetic stations using multinomial type multiplicities, recomputing exposure normalization and all hybrid weights. This is equivalent to sampling individual stations with replacement because type fully determines Y, group, and exposure. It represents one supported layer and does not evaluate cross-layer dependence. For only one represented group, record the event and use A3's zero-composition convention.

Main marginal percentile intervals use 1,999 inner bootstrap samples and linear 2.5th and 97.5th percentile interpolation. A station-weighted median additionally receives a conservative binomial order-statistic sign interval and the exact conditional-bootstrap percentile interval obtained from the binomial CDF of the odd-sample median. The sign interval has a finite-sample coverage lower bound for any fixed population median; ties can make coverage conservative. These benchmarks apply only to the unweighted median, not to the activity-weighted gap or components.

## Historical execution and scope

The historical pilot used 200 outer samples and 399 inner resamples for all scenarios. The historical formal run used seed namespace 9132602, 2,000 outer samples, and 1,999 inner resamples per scenario. Conditional on true coverage near 95 percent, 2,000 outer samples give approximately 0.49 percentage points of Monte Carlo standard error. Arithmetic and truth verification are execution gates; nominal coverage is not a pass criterion.

The scenario 16 adaptive diagnostic uses 7,999 inner resamples over the saved 2,000 formal outer datasets. It is a post-result diagnostic and does not increase the formal outer sample count. This simulation specifies synthetic station types and requires no transaction-level original data. Frozen public summaries are available under data/reference/historical/window_and_mixture_validation/v2/formal/. The full saved formal replicate parquet files required to reproduce the adaptive diagnostic are not included in this package.

## Method references

The design uses the aims, data-generation, estimands, methods, and performance-reporting structure of Morris, White, and Crowther (2019), https://doi.org/10.1002/sim.8086. Binomial order-statistic confidence intervals follow the construction explained in Stark's statistical inference notes, https://www.stat.berkeley.edu/~stark/Teach/S240/Notes/ch5.htm. Binomial CDF, survival, and quantile calculations use SciPy's binomial distribution implementation.

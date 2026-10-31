# Datasheet for TE-GvA

TE-GvA is a dataset of gather-or-act decisions in fault-driven incident response on the Tennessee Eastman
process. Each record is a moment in a simulated incident at which a defender has to choose between collecting
more evidence and responding. For every candidate action at that moment, the dataset gives the cost that action
would have caused, estimated by rolling the full simulator forward from the same state. It accompanies the paper
"Gather or Act? Belief-Level Counterfactual Labels for Deadline-Aware Incident Response in Cyber-Physical Systems"
by A. Mohammadi Ruzbahani. The structure of this datasheet follows Gebru et al., "Datasheets for Datasets".

Counts of episodes, decisions, candidates and rollouts cover all three splits. Shares and other statistics cover
the training and validation splits together (the development pool): 800 episodes and 3,129 decisions in the
natural set, 800 episodes and 4,795 decisions in the stress set.

## At a glance

| | Natural set | Stress set |
|---|---|---|
| Role | main benchmark, drawn from the sampler's own incident mix | oversamples ambiguous, high-stakes incidents |
| Episodes (with labelled decisions) | 1,000 (990) | 1,000 (1,000) |
| Decisions: train / validation / test | 2,346 / 783 / 784 | 3,589 / 1,206 / 1,200 |
| Labelled candidate actions | 54,324 | 89,123 |
| Candidates per decision | 4 to 33, mean 13.8 | 4 to 35, mean 14.9 |
| Rollouts in the true world / in twin worlds | 217,296 / 114,320 | 356,492 / 238,656 |
| Ambiguous decisions (sensor link versus I/O module) | 42.3% | 58.2% |
| Episodes without a backup channel | 0% | 51.9% |
| High-stakes decisions (hindsight labels) | 9.7% | 22.4% |
| Gathering is optimal under hindsight labels, all decisions | 1.1% | 1.3% |
| Gathering is optimal under belief labels, ambiguous decisions | 15.5% | 23.0% (12.5% with, 33.1% without a backup channel) |

Both sets come from the same simulator, code, behaviour policy and labelling procedure. They differ in the
incident distribution, in the availability of backup channels, and in the number of labelled decision points per
episode (at most four in the natural set, one of which may come before the onset; at most six in the stress set,
all after the onset). Report the two sets separately. The stress set is not a natural distribution and should
not be pooled with the natural set.

### Terms used below

- *Hindsight labels* are computed in the world of the true cause. *Belief labels* (pair-conditioned belief
  labels) average over the two causes of the ambiguous pair, weighted by their posterior probability (Section
  2.5).
- An *ambiguous decision* is a decision of an episode whose incident is a fault of a sensor group's network link
  or of the same group's remote I/O module.
- The *observation actions* are `trace_link`, `query_historian` and `verify_sensor`. All other candidates,
  including waiting (`monitor`) and escalation, are *direct actions*.
- *Gathering is optimal* when the best observation candidate has a strictly lower label than every direct
  action.
- The *regret* of a candidate is its label minus the lowest label of its decision. *Belief regret* uses the
  belief labels.
- A decision is *high-stakes* when its candidates differ in trip probability.

## 1. Motivation

**For what purpose was the dataset created?** To learn or evaluate a response policy for cyber-physical
incidents, one needs to know what each available action would have led to from the same situation. Incident logs
record one action per situation, and public datasets for industrial control systems record detection labels
rather than the consequences of responses. TE-GvA provides such data for one recurring decision: whether to
gather discriminating evidence first or to act on the most likely cause, while evidence expires and the process
drifts toward a protective shutdown (a trip).

Labels computed in the world of the true cause almost never favour gathering, because once the cause is fixed
the right repair is among the candidates and its label already shows that it works. The dataset therefore also
contains belief labels for the look-alike pair that dominates the decisive cases: a failed sensor link versus a
frozen I/O module of the same sensor group.

**Who created the dataset?** Ali Mohammadi Ruzbahani (ORCID 0009-0004-3287-552X), Smart Cyber-Physical (SCPS) Lab,
Department of Electrical and Software Engineering, University of Calgary, with the code in this repository.

## 2. Composition

### 2.1 What an instance is

An instance is a decision point of one simulated episode. At decision time τ the defender holds an evidence log
in which every item has an arrival time and a validity window: window summaries of the measurements, detector
alerts, link-health reports, tool results, operator reports and the receipts of the defender's own actions. Each
instance contains:

- the evidence log around τ (valid, expired and soon-to-arrive items) as a tensor of 256 item slots;
- the candidate actions, each described by 90 features;
- for each candidate, labels aggregated over four rollouts: the mean and standard deviation of the cost, the
  rates of trip, resolution and first-action success, and, per rollout, the trip flag, trip time, horizon and the
  duration and censoring of the first action; for observation actions also the value of information;
- for each candidate, the belief labels (cost, trip probability, value of information);
- whether the behaviour policy explored at this decision, and the first action each of two reference playbooks
  would choose;
- hidden fields for scoring and analysis, such as the incident kind and the posterior probability of the true
  cause.

### 2.2 The two incident distributions

Each episode has a one-hour warm-up, an incident whose onset is drawn uniformly between 10 and 60 minutes after
the warm-up, and a six-hour horizon after the onset. The incident runs on top of a background disturbance. IDV
numbers refer to the standard Tennessee Eastman process disturbances.

| | Natural set | Stress set |
|---|---|---|
| Incident type: link / I/O module / sensor / disturbance | 0.35 / 0.20 / 0.25 / 0.20 | 0.35 / 0.35 / 0.10 / 0.20 |
| Units hit by link and module faults | I/O modules and sensor links of the reactor and stripper weight 2, all others 1 | I/O modules: reactor and stripper 3, feeds 2, others 0.5; sensor links: reactor and stripper 3, feeds 2, others 0.5; actuator links: reactor, separator and stripper 1.5, others 0.5 |
| Sensor faults | eight sensors; frozen / drift (2-8%/h) / bias (2-6%) with weights 0.5 / 0.3 / 0.2 | reactor temperature (weight 3/6), reactor pressure, reactor level and stripper level (1/6 each); frozen / drift / bias 0.6 / 0.3 / 0.1 |
| Process disturbances | IDV1, IDV2, IDV4, IDV5, IDV6, IDV7, IDV12, IDV13, IDV14 | IDV6 (1/2), IDV1 (1/3), IDV2 (1/6) |
| Background none / IDV8 / IDV1 | 0.4 / 0.3 / 0.3 | 0.2 / 0.4 / 0.4 |
| Backup channel for sensor links | always | missing with probability 0.5, drawn independently of the incident |

An incident disturbance that is already active in the background is drawn again. In the development pool the
natural episodes contain 278 link faults, 167 I/O module faults, 197 sensor faults and 158 disturbances, and the
stress episodes 264, 289, 89 and 158. Under the behaviour policy, 5.1% of natural and 19.0% of stress episodes end
in a trip.

In the ambiguous family, both causes leave the controller with the same held readings, so the process and its
telemetry look identical until the defender's actions or stochastic evidence separate them.

### 2.3 Size and splits

Splits are drawn by scenario, never by decision. The split follows from the episode seed s alone: with
u = (s × 2654435761 mod 2^32) / 2^32, an episode is in training if u < 0.6, in validation if u < 0.8 and in test
otherwise. All decisions of an episode, and of its twin, therefore stay in one split. The natural set uses seeds
1000 to 1999 and the stress set seeds 500000 to 500999. `meta.json` lists the seeds of the episodes that appear in
each split's arrays.

| Split | Natural episodes (with decisions) | Natural decisions | Natural candidates | Stress episodes | Stress decisions | Stress candidates |
|---|---|---|---|---|---|---|
| train | 600 (592) | 2,346 | 32,568 | 599 | 3,589 | 53,896 |
| validation | 200 (199) | 783 | 10,748 | 201 | 1,206 | 17,512 |
| test | 200 (199) | 784 | 11,008 | 200 | 1,200 | 17,715 |

Ten natural episodes (eight in training, one in validation, one in test) have no labelled decision, because the
defender never received an alert or other evidence it could cite. The nine in the development pool are three
sensor drifts, two sensor biases and four disturbances. Episodes without a labelled decision are not in the
arrays.

### 2.4 Files and fields

Each set is a directory with `train.npz`, `val.npz`, `test.npz`, `meta.json` and one `<split>_item_ids.json` per
split. The arrays load with `numpy.load(path, allow_pickle=False)`. In the tables, N is the number of decisions
and K the number of candidates in the split; candidates are stored as one flat list, and `cand_decision` gives
each candidate's decision row. M = 256 evidence slots, P = 95 payload columns, F = 90 action features, R = 4
rollouts per candidate and S = 23 evidence sources.

All times are in simulated seconds. `decision_time`, `dec_tau` and the item times use the episode clock, which
starts at the beginning of the warm-up (the warm-up ends at 3,600 s). `y_trip_time`, `y_horizon` and
`dec_oracle_tts` count from τ. Durations count from the start of the first action.

#### Evidence

| Field | Type, shape | Meaning |
|---|---|---|
| `decision_time` | float64 [N] | decision time τ |
| `segment` | int64 [N] | episode index within the split |
| `item_id` | int64 [N, M] | item identifier (see below); -1 for an empty slot |
| `present` | bool [N, M] | the slot holds an item |
| `source`, `kind` | int64 [N, M] | indices into `vocab.sources` (23) and `vocab.kinds` (8) of `meta.json` |
| `event_start`, `event_time` | float64 [N, M] | start and end of the summarized window; equal for point events |
| `arrival_time` | float64 [N, M] | when the item reached the defender |
| `valid_until` | float64 [N, M] | end of the item's validity; infinite for the receipt of a still-running action |
| `retracted_at` | float64 [N, M] | when the item was withdrawn; infinite for almost all items |
| `payload` | float32 [N, M, P] | scaled payload, zero beyond the width used by the source |
| `payload_dim` | int64 [S] | payload width used by each source |

An item is evidence at τ only if it is valid: `present` and `arrival_time ≤ τ < min(valid_until,
retracted_at)`. The tensors deliberately also hold expired items and items that arrive up to 120 s after τ;
about 72% of the stored slots are not valid at τ, so a model that ignores this rule sees future evidence. In this
release every row fills all 256 slots, and slots are ordered by arrival time. The payload column names are in
`vocab.payload_layout`, the scaled columns in `vocab.payload_scaled`, and the scaler in `scaler`.

#### Candidates and hindsight labels

| Field | Type, shape | Meaning |
|---|---|---|
| `cand_features` | float32 [K, F] | one-hot tool (11), family (4), target kind (5) and target (60); two arguments scaled to [0, 1]; log1p of the median and 90th-percentile duration; resource cost / 20; one-hot side-effect class (5); column names in `vocab.action_columns` |
| `cand_decision` | int64 [K] | row of the candidate's decision |
| `y_J`, `y_J_std` | float64 [K] | mean and population standard deviation (ddof = 0) of the cost J over the four rollouts |
| `y_regret` | float64 [K] | J minus the lowest J of the decision |
| `y_p_trip` | float64 [K] | share of rollouts that tripped |
| `y_trip_event`, `y_trip_time`, `y_horizon` | float64 [K, R] | per rollout: trip flag; trip time, equal to the horizon when there was no trip; horizon, 10,800 s unless the episode ended first |
| `y_p_resolved` | float64 [K] | share of rollouts in which every fault was cleared by the end; NaN for disturbances |
| `y_p_mitigated` | float64 [K] | share in which the faults were cleared or every failed link was on its backup channel; NaN for disturbances |
| `y_p_success` | float64 [K] | share of uncensored rollouts in which the first action succeeded; NaN if all are censored |
| `y_first_censored` | float64 [K, R] | 1 if the first action was still running, or accepted but not started, when the rollout ended |
| `y_duration` | float64 [K] | mean duration of the first action over rollouts in which it finished; NaN if none did |
| `y_duration_time`, `y_duration_censored` | float64 [K, R] | survival form: the finished duration, or for a censored rollout the time from the start to the end of the rollout (0 if it never started); NaN if the action was refused; and the censoring flag |
| `y_observe` | bool [K] | the candidate is an observation action |
| `y_voi` | float64 [K] | value of information: the lowest `y_J` among the decision's direct actions minus this candidate's `y_J`; positive means the observation beats every direct action; NaN for direct actions |

#### Belief labels

| Field | Type, shape | Meaning |
|---|---|---|
| `y_J_belief`, `y_p_trip_belief` | float64 [K] | cost and trip probability averaged over the two causes of the pair (Section 2.5) |
| `y_voi_belief` | float64 [K] | value of information computed from `y_J_belief` |

Outside the ambiguous family, and where deterministic evidence excluded the other cause, the belief labels equal
the hindsight labels.

#### Decisions

| Field | Type, shape | Meaning |
|---|---|---|
| `dec_tau`, `dec_episode_seed`, `dec_segment` | float64 / int64 / int64 [N] | decision time (equal to `decision_time`), episode seed, episode index (equal to `segment`) |
| `dec_n_candidates`, `dec_explored` | int64 / bool [N] | number of candidates; whether the behaviour policy explored at this decision |
| `dec_baseline_act_first`, `dec_baseline_investigate_first` | int64 [N] | index, among the decision's candidates, of the first action of each reference playbook; a playbook that would wait maps to `monitor(300)` |
| `dec_backup_available` | bool [N] | the episode's sensor links have a backup channel |
| `dec_kind` | int64 [N] | hidden: incident kind (none, link_fault, io_module_fault, sensor_fault, disturbance); none before the onset |
| `dec_onset` | bool [N] | hidden: the incident has started |
| `dec_oracle_tts` | float64 [N] | hidden: time to trip if nothing is done; NaN if no trip within three hours (about 80% of natural and 49% of stress decisions) |
| `dec_family` | bool [N] | hidden: the episode belongs to the ambiguous family; also true for its decisions before the onset (36 in the natural development pool) |
| `dec_posterior` | float64 [N] | hidden: posterior probability of the true cause; exactly 1 outside the family and where deterministic evidence excluded the other cause |
| `dec_twin_ok`, `dec_belief_fallback` | bool [N] | hidden: the twin was labelled at this decision; the belief labels are hindsight labels because the decision is outside the family. In this release `dec_twin_ok` equals `dec_family`, and `dec_belief_fallback` equals its negation |

The hidden fields reveal the cause. Use them for scoring and analysis, never as model inputs.
`dec_backup_available` describes the scenario, which a defender can observe, and is useful for splitting results.

#### Metadata

`meta.json` holds the vocabularies (sources, kinds, causes, payload layouts, tools, targets and action columns),
the payload scaler, the seeds and counts of each split, the label keys, and a provenance record of the original
build. The provenance record contains configuration and code checksums and internal run identifiers. The code
and configuration files in this repository had their comments removed before release; their content is
unchanged, but their checksums differ from those in the provenance record.

`<split>_item_ids.json` holds a list `keys` whose entry i is [episode index within the split, evidence
identifier] for `item_id` i.

### 2.5 How the labels are computed

**Hindsight labels.** For every candidate, the labeller restores a snapshot of the whole simulator at τ, submits
the candidate as the first action and lets a fixed follow-up policy act for three hours. This is repeated four
times with different random numbers. The cost J over these three hours is measured in cost points:

| Cost item | Cost points |
|---|---|
| trip, which also ends the rollout | 100 |
| operation with a shutdown margin below 10% of its nominal value | 5 per hour |
| lost production | 10 per hour at zero production, pro rata |
| any sensor link on its backup channel | 0.5 per hour |
| tool effort | see the table below |

| Tool | Nominal duration | Effort |
|---|---|---|
| `monitor(T)` (wait until τ + T) | T | 0 |
| `query_historian` | 10 s | 0.04 |
| `trace_link` (correct with probability 0.97) | 2 min | 0.2 |
| `verify_sensor` (field check of a sensor group) | 20 min | 1 |
| `switch_to_backup` | 1 min | 0.2 |
| `restore_primary` | 1 min | 0.2 |
| `repair_link` | 40 min | 2 |
| `recalibrate_sensor` | 60 min | 2 |
| `replace_io_module` | 60 min | 2 |
| `set_production` | 30 s | 0.2 |
| `escalate` (operator diagnosis) | 15 min | 4 |

Effort is charged when a tool starts; an action refused before it starts costs nothing. The cost weights are
illustrative design choices. Because a rollout stops at a trip, labels for another trip penalty L follow exactly
as J + (L − 100) times the trip probability.

**Belief labels.** For an ambiguous decision with true cause c and the other cause of the pair c′, the belief
label of a candidate a is

    J_belief(a) = π(c) J(a | c) + π(c′) J(a | c′, h)

where π is the posterior over the two causes given everything the defender observed by τ, and J(a | c′, h) is the
cost of a in a twin episode in which c′ is active and the defender's history h is replayed (Section 3.4). The trip
probability is mixed the same way. The twin cost itself is not stored, but where 0 < π(c) < 1 it can be recovered
as (`y_J_belief` − π(c) `y_J`) / (1 − π(c)), with π(c) = `dec_posterior`.

The posterior is computed analytically. It conditions on the incident being one of the pair, and it treats the
onset time as known.

- The prior is the sampler's probability of each incident, normalized within the pair: 0.519 for the link fault
  in the natural set and 0.388 in the stress set, the same for every sensor group.
- Stochastic evidence enters through its likelihood under each cause. A down link is reported by network
  monitoring with probability 0.05 per 36-second window and a healthy one with probability 0.0003; a report that
  is missing also counts, allowing for its delivery delay. Trace results are correct with probability 0.97, and
  the operator names the true cause with probability 0.85.
- All other evidence is deterministic. If any of it differs between the two worlds, for example fresh data after
  a repair, the other cause is excluded and the belief labels equal the hindsight labels. This happens in 15.3%
  (natural) and 13.6% (stress) of ambiguous decisions.

Because the likelihood uses the generator's own parameters, the posterior is calibrated by construction. The
measured expected calibration error is 0.015 (natural) and 0.014 (stress).

### 2.6 Selected statistics

| Development pool | Natural | Stress |
|---|---|---|
| cost J per candidate: mean, median, 95th percentile | 13.7, 7.2, 62.5 | 27.0, 8.8, 107.1 |
| candidates that never / always / sometimes trip | 89.2 / 2.9 / 7.9% | 72.8 / 13.3 / 13.9% |
| waiting is among the lowest-cost candidates (hindsight labels) | 66.5% | 57.6% |
| hindsight-best action has positive belief regret (ambiguous decisions) | 28.9% | 34.2% |
| mean belief regret of the hindsight-best action (ambiguous decisions) | 1.55 | 2.69 |
| pooled variance of paired differences between candidates, relative to independent rollouts | 0.140 | 0.221 |
| runner-up not separated from the best candidate (paired 95% interval over the rollouts) | 30.9% | 34.5% |

### 2.7 Other composition questions

**Is it a sample?** Yes. Scenarios are drawn from the two incident samplers, and decision points are the first
labelled points of each episode's behaviour trajectory, so the states follow the behaviour policy.

**Are instances related?** Decisions within an episode share a trajectory and stay in one split. The candidates
of a decision share their rollout random numbers by design, and the belief labels of an ambiguous decision
depend on its twin episode.

**Is there redundancy?** There are no duplicate decisions. Many candidates of a decision have identical outcomes;
see Section 8.

**Is it self-contained?** The arrays are. Regenerating or extending the data needs the code in this repository
and TEP Studio with the Tennessee Eastman kernel, which are not included (Section 6).

**Does it contain personal or sensitive data?** No. Everything is simulated and no people are involved.

## 3. Collection process

### 3.1 Simulator

The process model is the revised Tennessee Eastman model of Bathelt, Ricker and Jelali (2015). It runs through TEP
Studio, a Python wrapper of the model (https://github.com/khalidlabs/tep-studio), in mode 1 under Ricker's
decentralized multiloop controller, with a 1.8 s control interval, 41 measurements, 12 valves and the model's
shutdown interlocks.

| Component | Identifier |
|---|---|
| TEP Studio commit | `852f0c69ab974540750d61bd1dbfafd56bfd80bb` |
| kernel `temexd_mod.c`, sha256 | `eff57b75700e7472005785150c21f82b0c3d178ebceebe37e532794f23785aba` |

The code checks the kernel against this hash whenever the plant is built, provided the TEP Studio source tree is
present.

Around the process we simulate a control network and a tool executor. Sensor and actuator links are grouped by
process unit, with one remote I/O module per group. A failed link holds its last delivered values, and a sensor
link may have a backup channel that delivers the same readings 36 s late. Every measurement is summarized over 36 s
windows whose summaries arrive with log-normal delays and stay valid for 4 to 15 minutes. Staleness and deviation
detectors are calibrated on normal operation only. Tools have log-normal durations and pass through a queue with a
5 s latency; their cited evidence and the state are checked again when they start, and every step produces a
receipt that enters the evidence log. Every response action must cite at least one item of evidence from the
plant, the network or a tool; the defender's own receipts do not count.

### 3.2 Behaviour policy and decision points

The trajectories come from a behaviour policy: a rule-based playbook with an act-first and an investigate-first
mode, drawn per episode, plus uniform exploration over the candidates with probability 0.3 at each labelled point.
The raw episode records log the probability of each submitted action and of submitting nothing; they are not part
of this release, and the arrays keep only the exploration flag.

A decision point is labelled when a new alert arrives or, after the onset, when relevant evidence can be cited.
The candidates are the actions whose target relates to the current evidence, plus waiting, escalation and
production changes, out of 109 possible actions.

### 3.3 Branched rollouts

At every labelled point the labeller snapshots the complete environment: the plant with its noise generator, the
controller, network, telemetry, evidence log, executor queue, costs and random streams. For each candidate and
rollout it restores the snapshot, reseeds every random stream from the scenario seed, the decision point and the
rollout number, submits the candidate, and lets the follow-up policy act for three hours or until the episode
ends. The follow-up policy is the playbook in act-first mode, starting from what the behaviour policy had observed
and done. It acts on trace results and on staleness that persists after a backup switch or a repair, but not on
operator reports or field checks. A first action `monitor(T)` means that nothing is done before τ + T. All
candidates share the random numbers of a rollout, so differences between candidates are estimated with common
random numbers.

### 3.4 Twin episodes

Every episode of the ambiguous family is run a second time under the other cause of the pair. The twin shares the
seed and therefore every random stream, the background disturbance, the backup availability and the end of the
episode. The module fault starts one control step before the link fault, so that the delivered readings and all
window summaries are identical until the two causes diverge. The twin repeats each of the defender's actions at the
same control step. Where the defender cited an item that exists only in the original world, the twin cites a
shared item instead, and actions that fall after the end of the twin are not repeated (70 natural and 99 stress
actions in the development pool). At each labelled point the twin labels the same candidates with the same
rollouts, and its follow-up policy remembers what the defender actually observed.

The twin repeats actions chosen for the original cause, so it trips much more often than the original episode.
It represents the cost of the defender's actual history under the other cause, not an independent sample of that
cause.

### 3.5 When and how

The data were generated on 27 September 2026 with 15 parallel processes: 8.2 hours for the natural set and 13.3
hours for the stress set, including the twin episodes, with Python 3.12, NumPy 2.4 and SciPy 1.17 on Windows 11.
The process was fully automated; no people took part, and no ethical review was required.

## 4. Preprocessing and labelling

For each decision the export keeps the items that arrived by τ + 120 s and are at most 30 minutes old, together
with every item valid at τ, up to 256 items chosen round-robin over the sources with the most recent first.

The receipts and tool results of actions taken at or after the decision are removed from its row, together with
any item derived from them, and so are items that are valid by their time stamps but were not yet known at τ.
Automated checks over all rows of the training and validation splits find no such item and no receipt without an
action identifier.

Payloads are scaled robustly (median and a spread estimate) and clipped to ±10; the scaler is fitted on each set's
own training items. Flags, fractions and one-hot fields are not scaled. Hindsight labels are aggregated over the
four rollouts, keeping censoring for first actions that were still running. Belief labels are computed from the
evidence stored with each decision, using the posterior of Section 2.5.

No decision was removed. Before release we rebuilt the training and validation arrays from the raw episode
records and obtained identical values; the raw records themselves are not released (Section 6).

## 5. Uses

The accompanying paper uses the dataset to compare decision rules trained on hindsight and on belief labels, and
gradient-boosted and neural decision rules.

**Intended uses**

- training and evaluating decision rules that predict the cost of each candidate action and choose among them;
- studying the gather-or-act choice and the value of information at the defender's level of knowledge, with the
  belief labels;
- comparing labelling methods and their effect on learned decision rules;
- survival models of action durations, using the censoring fields.

**Uses to avoid**

- feeding the hidden fields to a model;
- ignoring the validity rule of the evidence, since the tensors contain items from after τ on purpose;
- pooling the natural and stress sets into one number;
- judging the value of gathering with hindsight labels, which by construction almost never favour it;
- treating the belief labels of decisions outside the ambiguous family as belief-level, since they are hindsight
  labels;
- reading the costs, trip rates or deadlines as properties of a real plant;
- drawing conclusions about attacks, since the incidents are faults and disturbances;
- selecting models on the test split. The paper evaluated it once, after all modelling decisions had been fixed on
  the validation split.

## 6. Distribution

The dataset is published at https://github.com/ruzbahani/TE-GvA together with the code and the results of the
paper. The data and this datasheet are also archived on Zenodo (https://doi.org/10.5281/zenodo.23830595). The arrays
are NumPy `.npz` files and the metadata JSON, all loadable without pickle; `SHA256SUMS` lists a
checksum for every file.

Licences: the code is released under the MIT licence, and the data and results under Creative Commons Attribution
4.0 (CC BY 4.0). Please cite the paper when you use them.

Not included:

- TEP Studio (BSD 3-Clause) and the Tennessee Eastman kernel it ships, which has no explicit licence. Obtain them
  from the TEP Studio repository at the commit in Section 3.1.
- The raw episode records, which hold the per-rollout costs and their components, the twin worlds' own labels and
  the behaviour policy's actions and probabilities. They are Python pickles, which can execute code when loaded
  from an untrusted source.
- Model weights.

## 7. Maintenance

The author maintains the dataset. Questions and reports of errors are welcome through the issue tracker of the
repository. If the data change, the change will be described there and released with new checksums.

## 8. Limitations and known issues

The data are simulated: one process model, one controller, idealized links and tools, and illustrative cost
weights. Branched and twin rollouts cannot be collected on a real plant, so the dataset is a testbed for methods,
not an estimate of real trip rates or deadlines. The incidents are component faults and process disturbances. The
labelling assumes evidence generated by a known observation model that does not adapt to the defender; an
adversary who forges or suppresses evidence would break that assumption.

The belief labels cover one confusion, a sensor link versus an I/O module on the same sensor group, and they are
conditioned on that pair. Other confusions, such as a sensor fault versus a link fault, a disturbance versus a
sensor fault, or whether an incident has started at all, carry hindsight labels; this applies to 57.7% of natural
and 41.8% of stress decisions. The posterior is that of an idealized observer: it uses the true observation
parameters and a known onset, and it conditions on everything the defender observed, including evidence that no
longer fits in the 256-slot tensor.

The labels describe one first action followed by the act-first playbook for three hours. Because that playbook does
not act on operator reports or field checks, escalation and field checks never pay off in the labels, and a
different follow-up policy would change the labels. A candidate that postpones a trip beyond three hours looks as
good as one that prevents it, and 64 natural and 4 stress decisions have a shorter horizon because the episode
ended. The candidates are a subset of the 109 possible actions, so the lowest-cost candidate is not a global
optimum.

With four rollouts per candidate, trip probabilities are multiples of 0.25. Exact ties at the lowest cost occur in
41.5% (natural) and 41.3% (stress) of decisions under hindsight labels. Nearly all of them are between candidates
with identical trajectories, mostly `monitor(300)` and `monitor(900)`, so metrics based on the best action must
handle ties.

Censoring is most common for module replacement (5.5% of its first-action rollouts in the natural set, 9.6% in the
stress set) and sensor recalibration (3.4% and 7.0%); use the survival fields where this matters. Results of
historian queries and field checks are rare in the evidence, and labelled decisions concentrate in the first hour
after the onset.

## Reproducing the data

Regenerating the data needs the Python dependencies in `code/pyproject.toml` and TEP Studio at the commit in
Section 3.1, installed from a source checkout so that the kernel can be checked against its hash:

```
git clone https://github.com/khalidlabs/tep-studio
git -C tep-studio checkout 852f0c69ab974540750d61bd1dbfafd56bfd80bb
pip install -e tep-studio
cd code
python scripts/generate_experience.py --set "seeds=[1000,2000]" workers=15 --name natural
python scripts/generate_experience.py --config experience/generate_stress.yaml --set "seeds=[500000,501000]" workers=15 --name stress
python scripts/build_dataset.py --episodes <episode directory> --out <output directory> --name natural
```

Seeds run from the first value up to, but not including, the second. `workers` is the number of parallel
processes. Each generation run prints the directory it wrote; pass it to the build step, once per set. The analysis
scripts read the arrays from the directories named in their settings (the `SETS` constants in `code/scripts` and
the `data` entry in `code/configs/model`); build into those directories or change the settings. We have not tested
whether regeneration on another platform reproduces the episodes bit for bit.

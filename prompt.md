You are working on my SIH26 Hyperlocal Weather Downscaling project.

PROJECT ROOT:
E:\SIH26_Downscaling

I want you to create ONE MASTER PIPELINE FILE that orchestrates the EXISTING pipeline.

The purpose of this master file is NOT to replace or redesign the existing scripts.

It should simply coordinate them dynamically using:

    State
    District
    Block

============================================================
CORE REQUIREMENT
============================================================

Create one new master/orchestrator Python file.

This will become the SINGLE ENTRY POINT for running the complete SIH26 pipeline for any block.

For example:

    python run_pipeline.py --state Maharashtra --district Nashik --block Sinnar

or:

    python run_pipeline.py --state Maharashtra --district Pune --block Baramati

The master file should then execute the EXISTING scripts in the correct order.

Do NOT rewrite the existing methodology into the master file.

The master file should orchestrate the existing scripts.

============================================================
VERY IMPORTANT — DO NOT CHANGE METHODOLOGY
============================================================

This task is ONLY about:

1. Dynamic State/District/Block handling
2. Pipeline orchestration
3. Checking whether required data already exists
4. Running only the missing/incomplete stages
5. Passing State/District/Block to existing scripts
6. Keeping outputs isolated by State/District/Block

DO NOT change the scientific methodology.

DO NOT change:

- feature engineering methodology
- feature definitions
- models
- model architecture
- model hyperparameters
- target definitions
- D1-D7 forecast horizons
- chronological splitting
- leakage prevention
- Stage-1 methodology
- Stage-2 methodology
- rainfall occurrence model
- rainfall conditional amount model
- rainfall 70/30 correction
- temperature policy
- humidity policy
- wind policy
- pressure exclusion
- ERA5-Land methodology
- GP spatial predictors
- LST availability rules
- temporal cutoff rules
- missing-value methodology
- imputation
- scaling
- evaluation methodology

Do not add new scientific features.

Do not remove scientific features.

Do not replace any model.

Do not introduce deep learning.

Do not "improve" the model.

If you discover a methodological issue, report it separately. Do not modify it.

============================================================
FIRST TASK — FULL REPOSITORY AUDIT
============================================================

Before creating the master file, inspect the existing repository.

Inspect:

    E:\SIH26_Downscaling\scripts
    E:\SIH26_Downscaling\models

Understand the actual existing execution order.

Identify:

- data collection scripts
- cleaning scripts
- feature dictionary/inventory scripts
- feature engineering scripts
- audits
- training dataset generation
- Stage-1 training
- Stage-1 evaluation/finalization
- Stage-2 GP downscaling
- model saving/loading
- current forecast generation
- dashboard/output generation

Also identify which scripts already accept:

    state
    district
    block

and which scripts currently use hard-coded:

    Maharashtra
    Nashik
    Sinnar

============================================================
MASTER FILE RESPONSIBILITY
============================================================

The master file should perform ONLY orchestration.

Conceptually:

    User gives:
        State
        District
        Block

             ↓

    Master Pipeline

             ↓

    Check required data

             ↓

    If missing:
        run existing collection script

             ↓

    Check cleaned data

             ↓

    If missing:
        run existing cleaning script

             ↓

    Check feature inventory/dictionary

             ↓

    If missing:
        run existing feature inventory script

             ↓

    Check engineered data

             ↓

    If missing:
        run existing feature engineering script

             ↓

    Run existing audits

             ↓

    Check training datasets

             ↓

    If missing:
        run existing training-data generation

             ↓

    Check Stage-1 model artifacts

             ↓

    If missing:
        run existing Stage-1 training

             ↓

    Check Stage-2 requirements

             ↓

    If missing:
        run existing Stage-2 pipeline

             ↓

    Generate current GP forecast

             ↓

    Complete

============================================================
DO NOT RECREATE EXISTING LOGIC
============================================================

If an existing script already performs a task, CALL THAT SCRIPT.

Do NOT copy its implementation into the master file.

For example, if:

    scripts/modeling/feature_engg/feature_engineering.py

already performs feature engineering,

the master file should execute that script.

It should NOT reproduce feature engineering code.

Likewise for:

- cleaning
- audits
- training data
- Stage 1
- Stage 2
- GEE extraction
- model training

The master is an ORCHESTRATOR, not a replacement implementation.

============================================================
DYNAMIC LOCATION
============================================================

The master file must receive:

    --state
    --district
    --block

Example:

    python run_pipeline.py \
        --state Maharashtra \
        --district Pune \
        --block Baramati

All downstream scripts that support these parameters should receive them.

For scripts that currently hard-code:

    Maharashtra/Nashik/Sinnar

modify those scripts ONLY as necessary so that the master can pass the requested location dynamically.

Do not change anything else in those scripts.

============================================================
DATA AVAILABILITY / INCREMENTAL PIPELINE
============================================================

This is one of the most important requirements.

The master pipeline must NOT blindly rerun everything.

For every stage, determine whether the required output already exists.

For example:

    Raw data
        ↓
    Cleaned data
        ↓
    Engineered data
        ↓
    Training data
        ↓
    Stage-1 model
        ↓
    Stage-2 training/cache
        ↓
    Current GP forecast

If a valid output already exists:

    REUSE IT

If it does not exist:

    RUN THE REQUIRED EXISTING SCRIPT

If it exists but is incomplete or invalid:

    RUN ONLY THE REQUIRED STAGE AGAIN

Do NOT delete the entire block's data.

Do NOT restart the complete pipeline unnecessarily.

============================================================
IMPORTANT: VALIDITY CHECKS
============================================================

Do not use file existence alone as the only criterion.

Where practical, verify:

- file exists
- file is not empty
- expected columns exist
- requested State/District/Block matches the data
- expected number/date coverage is reasonable
- required upstream dependency exists

Use existing project audits wherever available.

Do not create a completely new scientific validation methodology.

The checks are only for determining whether a pipeline stage needs to run.

============================================================
LOCATION ISOLATION
============================================================

Every block must remain completely isolated.

For example:

    Maharashtra / Nashik / Sinnar

must never accidentally use:

    Maharashtra / Pune / Baramati

data or model artifacts.

The master should resolve location-specific paths such as:

    E:\SIH26_Downscaling\...\Maharashtra\Nashik\Sinnar\

and:

    E:\SIH26_Downscaling\...\Maharashtra\Pune\Baramati\

automatically.

Never hard-code Sinnar as the default working location.

============================================================
MODEL ISOLATION
============================================================

Stage-1 and Stage-2 models are block-specific.

A model trained for:

    Maharashtra / Nashik / Sinnar

must not be loaded when running:

    Maharashtra / Pune / Baramati

unless the existing methodology explicitly requires it.

Store/load model artifacts using the State/District/Block hierarchy.

Do not overwrite Sinnar artifacts when running Baramati.

============================================================
ERA5-LAND STAGE-2 CACHE
============================================================

Stage-2 ERA5-Land reference data must also be block-specific.

For example:

    stage2_direct_gp_training/
        Maharashtra/
            Nashik/
                Sinnar/

and:

    stage2_direct_gp_training/
        Maharashtra/
            Pune/
                Baramati/

If the ERA5-Land cache already exists and is complete:

    reuse it.

If it is incomplete:

    extract only the missing data using the EXISTING Stage-2 extraction methodology.

If it does not exist:

    run the existing Stage-2 GEE extraction.

Do not change the ERA5-Land dataset or scientific target definition.

============================================================
EXISTING SCRIPTS MUST REMAIN THE SOURCE OF TRUTH
============================================================

The master file should discover and execute the existing scripts.

Do not make the master contain duplicate implementations.

The existing scripts remain responsible for:

    collection
    cleaning
    feature engineering
    audits
    training data
    model training
    Stage 1
    Stage 2

The master only decides:

    WHAT NEEDS TO RUN
    IN WHAT ORDER
    FOR WHICH LOCATION

============================================================
COMMAND LINE
============================================================

The final master should support:

    python run_pipeline.py --state Maharashtra --district Nashik --block Sinnar

and:

    python run_pipeline.py --state Maharashtra --district Pune --block Baramati

Prefer argparse.

Also support useful optional controls if they fit naturally into the existing architecture, such as:

    --from-stage
    --to-stage
    --force

BUT DO NOT add unnecessary complexity.

For example:

    --force

should mean:

    rerun the selected stage even if its output exists.

It must NOT delete unrelated existing data.

============================================================
PIPELINE STAGES
============================================================

Use the ACTUAL stages discovered in the repository.

Do not blindly assume the filenames.

Determine the real dependency order from the current code.

Then define clear master stages such as:

    1. Block data
    2. GP data
    3. Cleaning
    4. Feature inventory/dictionary
    5. Feature engineering
    6. Audits
    7. Training dataset
    8. Stage 1
    9. Stage 2
    10. Current GP forecast

Adjust these names/order to match the actual repository.

============================================================
LOGGING
============================================================

The master should clearly show what it is doing.

Example:

============================================================
SIH26 HYPERLOCAL WEATHER DOWNSCALING
============================================================

State    : Maharashtra
District : Pune
Block    : Baramati

============================================================
PIPELINE STATUS
============================================================

[1/10] Block data
      Status : AVAILABLE
      Action : REUSING

[2/10] GP data
      Status : AVAILABLE
      Action : REUSING

[3/10] Cleaning
      Status : AVAILABLE
      Action : REUSING

[4/10] Feature Engineering
      Status : MISSING
      Action : RUNNING

...

At the end:

============================================================
PIPELINE COMPLETE
============================================================

Location:
    Maharashtra / Pune / Baramati

Outputs:
    ...

============================================================
ERROR HANDLING
============================================================

If an existing script fails:

- stop the pipeline
- show which stage failed
- show the command that was executed
- show the relevant error
- do NOT silently continue to later stages

Do not hide errors.

============================================================
IMPORTANT: PRESERVE EXISTING SINNAR WORK
============================================================

Sinnar is currently the validated/reference block.

Do not delete or overwrite its existing:

- datasets
- engineered files
- training datasets
- models
- Stage-2 cache
- Stage-2 outputs
- forecasts

The first validation should be:

    Maharashtra / Nashik / Sinnar

and the master should detect and reuse existing data rather than recollecting everything.

Then test:

    Maharashtra / Pune / Baramati

to verify that the same pipeline works dynamically.

============================================================
BACKWARD COMPATIBILITY
============================================================

Do not break the existing individual scripts.

They should still be executable independently where they currently are.

The master is an additional orchestration layer.

============================================================
FINAL VALIDATION
============================================================

After implementation, perform these tests.

TEST 1:

    Maharashtra / Nashik / Sinnar

Expected:

- existing data detected
- existing outputs reused
- no unnecessary downloads
- correct Sinnar models loaded
- correct Sinnar Stage-2 cache loaded
- no Sinnar data overwritten

TEST 2:

    Maharashtra / Pune / Baramati

Expected:

- paths dynamically generated
- no Nashik/Sinnar data accidentally used
- missing data stages are collected
- existing Baramati data is reused when available
- block-specific models/artifacts are created
- Stage-2 operates on Baramati's GP data
- outputs stored under Baramati

TEST 3:

Run Sinnar again after completion.

Expected:

    mostly REUSING existing outputs

rather than rerunning the entire pipeline.

============================================================
FINAL REPORT
============================================================

After implementation, provide:

1. Name and location of the new master file.

2. Complete list of existing files modified to support dynamic parameters.

3. Complete list of files NOT modified.

4. Actual pipeline execution order discovered from the repository.

5. How each stage checks whether data already exists.

6. How partial/incomplete data is handled.

7. How State/District/Block paths are constructed.

8. How model artifacts are isolated by block.

9. How Stage-2 ERA5-Land cache is isolated by block.

10. Sinnar validation result.

11. Baramati validation result.

12. Any hard-coded location assumptions that remain, and why.

13. Any methodology/scientific issue discovered but intentionally NOT changed.

============================================================
MOST IMPORTANT INSTRUCTION
============================================================

This project already has a scientifically defined and working methodology.

DO NOT redesign it.

DO NOT improve it.

DO NOT simplify it.

DO NOT replace it.

DO NOT introduce new algorithms.

The only objective is:

    EXISTING PIPELINE
          +
    ONE MASTER ORCHESTRATOR
          +
    DYNAMIC STATE/DISTRICT/BLOCK
          +
    INCREMENTAL DATA CHECKING
          +
    AUTOMATIC EXECUTION OF MISSING STAGES

The master file should make it possible to take a completely new block and run the same existing SIH26 methodology for that block without manually editing paths throughout the project.


The Baramati test confirms that the master pipeline is dynamically resolving the State/District/Block paths correctly.

Do NOT bypass the GP target preflight and do NOT create any synthetic/fake GP targets.

The pipeline must continue to stop when a genuine GP target is unavailable.

However, fix the orchestration/argument propagation issue visible in the audit logs.

The audit scripts print:

    State: District: Block:

even though the resolved paths correctly contain:

    Maharashtra / Pune / Baramati

Make the master pipeline pass the requested:

    --state
    --district
    --block

to every existing downstream script that supports these arguments, including the audit scripts.

Do not change the audit methodology.

Do not change feature engineering.

Do not change leakage rules.

Do not change target definitions.

Do not bypass any safety/preflight check.

The only change should be correct propagation of the location arguments.

Then rerun:

    python run_pipeline.py --state Maharashtra --district Pune --block Baramati

The expected result is that audit logs explicitly show:

    State: Maharashtra
    District: Pune
    Block: Baramati

while continuing to use:

    datasets/.../Maharashtra/Pune/Baramati

and the pipeline should STILL stop at GP target preflight if:

    datasets/targets/Maharashtra/Pune/Baramati/gp_weather_targets.csv

does not exist.

Do NOT create this target file artificially.

Report which scripts were missing the location arguments and exactly how they were fixed.

IMPORTANT CORRECTION TO THE PREVIOUS TASK:

DO NOT hard-code Baramati.

Baramati is ONLY a TEST CASE to prove that the pipeline is dynamic.

The final system must NOT contain special-case logic such as:

    if block == "Baramati":
        ...

or:

    Maharashtra/Pune/Baramati

or any equivalent hard-coded Baramati path, filename, model location, target location, or execution branch.

Likewise, do not special-case:

    Sinnar
    Nashik
    Pune
    Maharashtra

These are only examples/test locations.

============================================================
FINAL REQUIREMENT
============================================================

The master pipeline must be completely parameter-driven:

    --state
    --district
    --block

For example:

    python run_pipeline.py --state Maharashtra --district Nashik --block Sinnar

    python run_pipeline.py --state Maharashtra --district Pune --block Baramati

    python run_pipeline.py --state Maharashtra --district Nashik --block Niphad

    python run_pipeline.py --state <any> --district <any> --block <any>

All four commands must follow the SAME code path.

There must be NO:

    if Sinnar
    if Baramati
    if Nashik
    if Pune

logic.

============================================================
DYNAMIC PATH CONSTRUCTION
============================================================

Every location-specific path must be constructed from:

    state
    district
    block

For example:

    base / state / district / block

NOT:

    base / "Maharashtra" / "Nashik" / "Sinnar"

and NOT:

    base / "Maharashtra" / "Pune" / "Baramati"

The master must generate these paths at runtime.

============================================================
DYNAMIC DATA CHECKING
============================================================

For EVERY NEW BLOCK:

1. Receive state/district/block.
2. Resolve the corresponding directories.
3. Check whether required data already exists.
4. If valid data exists → REUSE.
5. If missing → call the existing collector.
6. If partially available → collect/process only what is missing.
7. Continue to the next dependency.
8. Never use another block's data.

The logic must be identical for every block.

============================================================
DYNAMIC MODEL STORAGE
============================================================

Models must also be separated automatically:

    models/
        <state>/
            <district>/
                <block>/

Do not create a special folder for Baramati.

Do not create a special folder for Sinnar.

The location hierarchy must be generated dynamically.

============================================================
DYNAMIC STAGE-2 CACHE
============================================================

Stage-2 ERA5-Land cache must automatically resolve to:

    stage2_direct_gp_training/
        <state>/
            <district>/
                <block>/

For example, the same code should automatically produce:

    .../Maharashtra/Nashik/Sinnar/

or:

    .../Maharashtra/Pune/Baramati/

or:

    .../Maharashtra/Nashik/Niphad/

based solely on the command-line arguments.

============================================================
NO HARDCODED TEST LOGIC
============================================================

After implementation, search the ACTIVE pipeline for:

    "Baramati"
    "Sinnar"
    "Nashik"
    "Pune"
    "Maharashtra"

Any remaining occurrence must be classified as one of:

1. Documentation/example/help text — acceptable.
2. Historical/legacy data — leave untouched if not part of execution.
3. Actual runtime logic — MUST be removed.

There must be no active runtime dependency on a particular block.

============================================================
MOST IMPORTANT TEST
============================================================

Do NOT only test:

    Sinnar

and:

    Baramati

The master should be designed so that a completely new block can be supplied tomorrow without modifying ANY Python code.

For example:

    python run_pipeline.py \
        --state Maharashtra \
        --district Nashik \
        --block Niphad

The user should not need to edit:

    run_pipeline.py

or any downstream script just because the block changed.

============================================================
PIPELINE ARCHITECTURE
============================================================

The architecture should be:

                    USER
                     │
                     ▼
              run_pipeline.py
                     │
            ┌────────┴────────┐
            │                 │
          state             district
                              │
                             block
                              │
                              ▼
                  Dynamic Location Context
                              │
                              ▼
                 Existing Pipeline Scripts
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
          Collection       Processing       Modeling
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                  <state>/<district>/<block>
                              │
                              ▼
                       Final GP Forecast

There must be ONE generic pipeline.

============================================================
DO NOT MODIFY SCIENTIFIC METHODOLOGY
============================================================

This remains strictly an orchestration/parameterization task.

Do NOT modify:

- feature engineering
- features
- models
- hyperparameters
- targets
- D1-D7
- Stage-1 methodology
- Stage-2 methodology
- ERA5-Land methodology
- leakage controls
- temporal controls
- LST rules
- precipitation methodology
- wind methodology
- pressure exclusion
- validation methodology

Only make the pipeline dynamic.

============================================================
FINAL ACCEPTANCE CRITERIA
============================================================

The implementation is correct only if:

✓ One master file exists.

✓ The master accepts:
      --state
      --district
      --block

✓ No block is hard-coded.

✓ Baramati is NOT hard-coded.

✓ Sinnar is NOT hard-coded.

✓ New blocks require NO code modification.

✓ Existing data is reused.

✓ Missing data is collected automatically.

✓ Partial data is handled incrementally.

✓ Models are isolated by State/District/Block.

✓ Stage-2 caches are isolated by State/District/Block.

✓ One block can never accidentally load another block's data.

✓ Existing scientific methodology is unchanged.

✓ Existing individual scripts remain usable.

✓ Running a new block requires only a command such as:

    python run_pipeline.py \
        --state Maharashtra \
        --district Pune \
        --block <NEW_BLOCK>

and nothing needs to be edited in the code.

IMPORTANT:

Baramati is NOT the destination.

Baramati is only the test proving that the system works.

The actual goal is:

    ONE GENERIC SIH26 PIPELINE
    FOR ANY STATE / DISTRICT / BLOCK.

    IMPORTANT FINAL REQUIREMENT — INCREMENTAL PIPELINE MUST COMPLETE MISSING STAGES

The master pipeline must behave as an INCREMENTAL BUILD SYSTEM.

Do NOT stop simply because a file, dataset, model, cache, or training artifact is missing.

A missing artifact means:

    "THIS STAGE HAS NOT BEEN COMPLETED YET"

and therefore the master must RUN THE REQUIRED EXISTING SCRIPT to create it.

============================================================
CORE BEHAVIOR
============================================================

For any State / District / Block:

    Check stage
        │
        ├── COMPLETE
        │       ↓
        │    REUSE
        │
        └── MISSING / INCOMPLETE
                ↓
             RUN STAGE
                ↓
          VERIFY OUTPUT
                ↓
          CONTINUE TO NEXT STAGE

The pipeline must continue until the requested final stage is complete.

============================================================
NEW LOCATION EXAMPLE
============================================================

Suppose I run:

    python run_pipeline.py \
        --state Maharashtra \
        --district Pune \
        --block A NEW BLOCK

and absolutely nothing exists for that location.

The master must NOT say:

    "data missing"
    "target missing"
    "pipeline stopped"

and exit.

Instead it must start building the location from the beginning:

    1. Collect block data
    2. Collect GP spatial data
    3. Clean data
    4. Build feature dictionary/inventory
    5. Feature engineering
    6. Run required audits
    7. Build required training data
    8. Train Stage-1 model
    9. Build/run Stage-2 ERA5-Land reference data
    10. Train Stage-2 model
    11. Generate current GP forecast
    12. Verify final outputs
    13. COMPLETE

The exact order must be determined from the existing repository's real dependencies.

============================================================
VERY IMPORTANT
============================================================

DO NOT bypass safety checks.

There is a difference between:

    "required input is missing and can be generated"
    
and:

    "scientifically invalid input must never be fabricated."

If a required GP target/reference is missing, determine whether the CURRENT active pipeline has an existing legitimate way to generate it.

For our current Stage-2 methodology, this is the existing ERA5-Land GP reference workflow.

Therefore:

    missing ERA5-Land GP cache
        ↓
    RUN EXISTING ERA5-LAND EXTRACTION
        ↓
    VERIFY CACHE
        ↓
    CONTINUE TO STAGE-2 TRAINING

Do NOT create:

    fake GP observations
    copied block observations
    interpolated GP targets
    synthetic weather targets

The master should invoke the legitimate existing collection/extraction process instead.

============================================================
NO PREMATURE STOPPING
============================================================

The master must NOT stop simply because:

- a raw dataset is missing
- cleaned data is missing
- engineered data is missing
- training data is missing
- a model is missing
- an ERA5 cache is missing
- a Stage-2 output is missing

These are NORMAL CONDITIONS for a new location.

The master should identify the missing stage and execute the existing script responsible for that stage.

============================================================
WHEN SHOULD THE PIPELINE ACTUALLY STOP?
============================================================

The master should stop only when:

1. An existing required script actually fails.
2. The script exits with a non-zero exit code.
3. Required external infrastructure/authentication is unavailable and the stage genuinely cannot execute.
4. A scientific safety check explicitly determines that proceeding would create invalid data and there is NO legitimate existing process capable of producing the required input.

When it stops, clearly report:

    Stage:
    Command:
    Error:
    Required input:
    Suggested next action:

Do NOT silently continue with incomplete or invalid data.

============================================================
AFTER EVERY STAGE
============================================================

After running a missing stage:

    RUN SCRIPT
       ↓
    CHECK EXIT CODE
       ↓
    CHECK EXPECTED OUTPUT
       ↓
    CHECK OUTPUT IS NON-EMPTY/VALID
       ↓
    MARK STAGE COMPLETE
       ↓
    CONTINUE

Do NOT assume that because a script exited successfully, the stage necessarily produced the required artifact.

Verify the expected output.

============================================================
EXAMPLE: COMPLETELY NEW LOCATION
============================================================

Command:

    python run_pipeline.py \
        --state Maharashtra \
        --district Nashik \
        --block Niphad

Assume NOTHING exists.

Expected behavior:

    [1/9] Block data
          Status : MISSING
          Action : COLLECTING
          ↓
          collection script runs
          ↓
          output verified

    [2/9] GP spatial data
          Status : MISSING
          Action : COLLECTING
          ↓
          GP scripts run
          ↓
          output verified

    [3/9] Cleaning
          Status : MISSING
          Action : RUNNING
          ↓
          cleaning completes

    [4/9] Feature dictionary
          Status : MISSING
          Action : BUILDING
          ↓
          dictionary created

    [5/9] Feature engineering
          Status : MISSING
          Action : RUNNING
          ↓
          engineered datasets created

    [6/9] Audits
          Status : MISSING
          Action : RUNNING
          ↓
          audits completed

    [7/9] Training dataset
          Status : MISSING
          Action : BUILDING
          ↓
          required training data generated

    [8/9] Stage-1
          Status : MISSING
          Action : TRAINING
          ↓
          block-specific model trained

    [9/9] Stage-2
          Status : MISSING
          Action : BUILDING/TRAINING
          ↓
          ERA5-Land reference extraction
          ↓
          Stage-2 model training
          ↓
          current GP forecast

    PIPELINE COMPLETE

This is the behavior we want.

============================================================
PARTIAL LOCATION EXAMPLE
============================================================

Suppose:

    Block data       = AVAILABLE
    GP data          = AVAILABLE
    Cleaning         = AVAILABLE
    Feature engineer = AVAILABLE
    Audits           = AVAILABLE
    Training data    = MISSING
    Stage-1 model    = MISSING
    Stage-2          = MISSING

The master must NOT restart from Stage 1.

It should do:

    [1] REUSE
    [2] REUSE
    [3] REUSE
    [4] REUSE
    [5] REUSE
    [6] REUSE
    [7] BUILD
    [8] TRAIN
    [9] TRAIN

Then finish.

============================================================
PARTIALLY COMPLETE STAGE
============================================================

If a stage has partially collected data and the existing collector supports incremental collection:

    reuse existing valid records
    determine missing range
    run existing incremental collector
    merge according to existing methodology
    verify final output

Do NOT delete the complete dataset and start from zero unless the existing methodology explicitly requires rebuilding it.

============================================================
STAGE DEPENDENCY GRAPH
============================================================

The master should understand dependencies.

Conceptually:

    RAW DATA
       ↓
    CLEANED DATA
       ↓
    FEATURE DICTIONARY / INVENTORY
       ↓
    FEATURE ENGINEERING
       ↓
    AUDITS
       ↓
    TRAINING DATA
       ↓
    STAGE-1 MODEL
       ↓
    STAGE-2 REFERENCE DATA
       ↓
    STAGE-2 MODEL
       ↓
    CURRENT GP FORECAST

If Stage N is missing, run Stage N.

Do not declare the whole pipeline unavailable just because Stage N is missing.

============================================================
IMPORTANT: CURRENT GP TARGET PREFLIGHT
============================================================

The current run is stopping at:

    training_data/00_target_preflight.py

because:

    gp_weather_targets.csv

does not exist.

DO NOT simply suppress this error.

First determine whether this preflight belongs to the CURRENT active training path.

Our current Stage-2 methodology uses:

    stage2_direct_gp_training/<state>/<district>/<block>/
    gp_era5_land_reference_targets.csv

If the direct ERA5-Land Stage-2 workflow is the current valid source of GP reference targets, then the master must invoke that workflow when the Stage-2 reference cache is missing.

For a new block:

    missing Stage-2 reference cache
             ↓
    run existing ERA5-Land extraction
             ↓
    cache created
             ↓
    Stage-2 training
             ↓
    GP forecast

Do NOT create gp_weather_targets.csv artificially.

============================================================
LOCATION MUST REMAIN 100% DYNAMIC
============================================================

There must be NO special handling for:

    Baramati
    Sinnar
    Nashik
    Pune
    Maharashtra

All locations must follow the same code path.

The ONLY inputs that change are:

    --state
    --district
    --block

============================================================
FINAL GOAL
============================================================

The user should be able to take ANY new block and execute:

    python run_pipeline.py \
        --state <STATE> \
        --district <DISTRICT> \
        --block <BLOCK>

and the system should automatically:

    CHECK
       ↓
    COLLECT WHAT IS MISSING
       ↓
    PROCESS WHAT IS MISSING
       ↓
    TRAIN WHAT IS MISSING
       ↓
    GENERATE WHAT IS MISSING
       ↓
    VERIFY
       ↓
    COMPLETE

No manual editing of Python files.

No hard-coded locations.

No manual copying of datasets.

No manual creation of target files.

No fake GP targets.

No methodology changes.

============================================================
FINAL SUCCESS CONDITION
============================================================

A completely new location should be able to go from:

    NOTHING

to:

    DATA
      ↓
    CLEAN DATA
      ↓
    ENGINEERED DATA
      ↓
    AUDITED DATA
      ↓
    TRAINING DATA
      ↓
    TRAINED STAGE-1 MODEL
      ↓
    ERA5-LAND GP REFERENCE
      ↓
    TRAINED STAGE-2 MODEL
      ↓
    CURRENT GP FORECAST

using ONLY:

    python run_pipeline.py \
        --state <STATE> \
        --district <DISTRICT> \
        --block <BLOCK>

That is the final objective.

The master is an INCREMENTAL BUILD/ORCHESTRATION SYSTEM, not merely a status checker.

We now have the desired master-pipeline behavior.

For a new/incomplete location, run_pipeline.py correctly:

- reuses completed stages
- identifies missing Stage-2
- launches the existing Stage-2 script dynamically
- creates the correct State/District/Block-specific cache
- detects missing ERA5-Land GP/day records
- attempts to collect the missing records

For Baramati it reached:

    Expected GP-day records : 95,931
    Missing GP-day records  : 95,931

and correctly attempted incremental ERA5-Land extraction.

DO NOT change the master pipeline behavior.

DO NOT hard-code Baramati.

DO NOT hard-code Sinnar.

DO NOT add any location-specific branch.

The current failure is inside:

    models/main_model/02_stage2_direct_gp_training.py

during the ERA5-Land incremental collection stage.

ERROR:

    The truth value of a DatetimeIndex is ambiguous.
    Use a.empty, a.bool(), a.item(), a.any() or a.all().

============================================================
TASK
============================================================

Find the exact expression in 02_stage2_direct_gp_training.py that is treating a pandas DatetimeIndex as a boolean.

Typical problematic patterns may look like:

    if dates:
    if not dates:
    if datetime_index:
    if not some_datetime_index:
    
or an equivalent indirect condition.

Replace ONLY the erroneous boolean test with the appropriate explicit pandas check, such as:

    if dates.empty:
    
or:

    if not dates.empty:

depending on the intended logic.

Do NOT blindly replace all conditions involving dates.

Trace the actual variable and preserve the existing intended behavior.

============================================================
CRITICAL METHODOLOGY REQUIREMENT
============================================================

Do NOT change:

- ERA5-Land dataset
- ERA5-Land target definition
- GP target methodology
- date range logic
- latest-source-date detection
- incremental cache logic
- GP/day completeness logic
- GEE extraction methodology
- batching methodology
- train/validation/test split
- Stage-2 model
- features
- hyperparameters
- inference
- forecast horizons
- any scientific methodology

This is ONLY a Python runtime bug fix.

============================================================
IMPORTANT: PRESERVE INCREMENTAL BEHAVIOR
============================================================

The existing intended behavior is:

    cache missing
        ↓
    determine requested period
        ↓
    determine available GEE date
        ↓
    identify missing GP/day records
        ↓
    download ONLY missing records
        ↓
    save cache
        ↓
    continue Stage-2 training
        ↓
    generate current GP forecast

Do not replace this with a full redownload every time.

If the cache is partially populated:

    existing valid records → KEEP
    missing records → COLLECT
    merge → VERIFY

Do not delete the existing cache.

============================================================
EARTH ENGINE QUOTA
============================================================

The run also produced:

    Your project has exceeded its noncommercial compute quota
    and is now in restricted mode.

Do NOT try to bypass Earth Engine quota restrictions.

Do NOT change the data source.

Do NOT switch to a fabricated/local target.

Do NOT alter the scientific methodology to avoid GEE.

First fix the Python DatetimeIndex error.

After that, if Earth Engine itself refuses the extraction because of quota restrictions, report that as a separate external service limitation.

============================================================
VALIDATION
============================================================

After fixing the DatetimeIndex error:

1. Run the Stage-2 script directly:

    python -u models/main_model/02_stage2_direct_gp_training.py ^
        --state Maharashtra ^
        --district Pune ^
        --block Baramati

2. Verify that the previous:

    DatetimeIndex truth value is ambiguous

error is gone.

3. Verify that the incremental ERA5-Land logic still prints:

    Expected GP-day records
    Missing GP-day records

4. Verify that an existing partial cache is preserved.

5. If Earth Engine quota permits, allow the extraction to continue.

6. If quota prevents extraction, stop with the actual GEE quota error rather than fabricating data.

7. Then run:

    python run_pipeline.py ^
        --state Maharashtra ^
        --district Pune ^
        --block Baramati

and verify that the master correctly recognizes whatever Stage-2 artifacts were successfully created.

============================================================
FINAL REQUIREMENT
============================================================

The fix must be generic.

It must work for:

    Maharashtra / Pune / Baramati

but MUST NOT contain Baramati-specific code.

It must also work for:

    Maharashtra / Nashik / Sinnar

and any future:

    <state> / <district> / <block>

using the same code path.

Report:

1. Exact line/function containing the DatetimeIndex boolean error.
2. Exact minimal code change made.
3. Confirmation that incremental cache behavior was preserved.
4. Confirmation that no methodology changed.
5. Confirmation that no location-specific branch was added.
6. Whether the remaining blocker, if any, is Earth Engine quota rather than Python code.
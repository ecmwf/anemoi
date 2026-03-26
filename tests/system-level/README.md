# System-level Tests

The `anemoi_test` suite provides system-level testing for the anemoi packages, covering the entire workflow from dataset creation to model training and inference.

For details on adding a test case or triggering suite deployment via GitHub, see the main
[Testing Documentation](https://anemoi.readthedocs.io/en/latest/contributing/testing.html).

## Updating Training Configs

The training configs for the global use case and for the lam use case are based on the configuration of the respective integration tests in training. We use full configs in the system-level test suite in order to track
changes in the configurations.

There is a helper script in anemoi-training to update these configs. It requires an up to date training environment.
Check out the branch in anemoi-core based on which you would like to update the configs. You'll need to
provide the path to your local anemoi repo, and can run the script like so:

```
python training/tests/integration/generate_slt_configs.py PATH_TO_YOUR_ANEMOI_REPO
```

It will update the training config files in the system-level tests folder, according ot the path that you provided. You can review the configs, and then commit them.

## Running Tests Locally During Development

To build and deploy the anemoi_test suite locally, you will need:
 - An ecflow server
 - [pyflow-wellies](https://pyflow-wellies.readthedocs.io/latest/) version ≥ 1.2.0

### Building the Suite

Run the build script with your desired output directory:

./build.sh -s OUTPUT_ROOT_SUITE=desired-output-dir

Set `OUTPUT_ROOT_SUITE` to the path you want to use as the root output directory of the suite, e.g. `$SCRATCH/workdir`.

This directory is used to store the suite's outputs and to build the virtual environments using uv. Therefore, the suite will only use the UV cache if it is located in the same file system as `OUTPUT_ROOT_SUITE`.

### Testing Configuration Changes

If you modify configuration files in `anemoi_test/configs`, you need to point to a committed branch of the `anemoi` repo so that the deployed suite can pull your config files from that branch.

- Commit your changes to a branch.
- Push the branch to anemoi.
- Run the build script pointing to that branch:

```
./build.sh -s OUTPUT_ROOT_SUITE=$SCRATCH/workdir anemoi_branch=name-of-your-branch
```

### Additional Build Options

You can configure additional build options defined in `configs/user.yaml`, e.g. the `anemoi-datasets` branch used to run the tests, by passing additional overrides:

```
./build.sh -s OUTPUT_ROOT_SUITE=$SCRATCH/workdir anemoi_datasets_branch=name-of-your-branch
```

### Deploying the Suite to the Ecflow Server

By default, the suite will be built in your home directory `$HOME/pyflow/anemoi_tests/{USER}`. To deploy the suite to the ecflow server,
- Navigate to this directory -- it should contain a definition file `{USER}.def`.
- Follow the [ecflow documentation](https://ecflow.readthedocs.io/en/5.14.1/quickstart.html) to start the server and configure the host.
- Load your suite definition to the server

    ```
    ecflow_client --load={USER}.def
    ```
- Follow the ecflow documentation to start and monitor the suite.

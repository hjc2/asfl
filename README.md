# asfl


## create and activate venv
```bash
python -m venv venv

source venv/bin/activate # linux
# OR 
source venv/Scripts/activate # windows
```

## Install dependencies

```bash
pip install . --no-cache-dir
```

## Setup the toml configuration

```toml
[tool.flwr.app.config]
num-server-rounds = 100 # number of server rounds
local-epochs = 3 # number of local epochs for each client
strat-mode = "fedcustom" # fedcustom or fedavg
inplace = true
file-writing = true

[tool.flwr.federations.local-simulation]
options.num-supernodes = 2 # number of vehicles
```



## Run (Simulation Engine)

In the `asfl` directory, use `flwr run` to run a local simulation:

```bash
flwr run
```

Advanced simulation

```bash
flower-simulation --app . --num-supernodes 200 --run-config 'num-server-rounds=400 strat-mode="fed_fuzz"'
```

## Fun Simulation Results

![screenshot](.github/Figure_1.svg)
![screenshot](.github/Figure_2.svg)


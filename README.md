# asfl


## create and activate venv

```bash
source venv/Scripts/activate
```

## Install dependencies

```bash
pip install .
```

## Setup the toml configuration

```toml
[tool.flwr.app.config]
num-server-rounds = 100 # number of server rounds
local-epochs = 3 # number of local epochs for each client
strat-mode = "fedcustom" # fedcustom or fedavg
inplace = true
file-writing = true
```



## Run (Simulation Engine)

In the `asfl` directory, use `flwr run` to run a local simulation:

```bash
flwr run
```


output 09-11-2024 (3:07)
```
INFO :      [SUMMARY]
INFO :      Run finished 50 round(s) in 551.49s
INFO :          History (loss, distributed):
INFO :                  round 1: 16.13967028947977
INFO :                  round 2: 16.13518502162053
INFO :                  round 3: 16.13680888215701
INFO :                  round 4: 16.132071803013485
INFO :                  round 5: 16.131488116163958
INFO :                  round 6: 16.13021508852641
INFO :                  round 7: 16.119215506773727
INFO :                  round 8: 16.124419540166855
INFO :                  round 9: 16.126892069975536
INFO :                  round 10: 16.129603173997666
INFO :                  round 11: 16.12129138946533
INFO :                  round 12: 16.112248886199225
INFO :                  round 13: 16.115063846111298
INFO :                  round 14: 16.11443340336835
INFO :                  round 15: 16.10862729890006
INFO :                  round 16: 16.098564604054328
INFO :                  round 17: 16.105502128601074
INFO :                  round 18: 16.10065786043803
INFO :                  round 19: 16.092070865631104
INFO :                  round 20: 16.090341687202454
INFO :                  round 21: 16.08925759792328
INFO :                  round 22: 16.086024701595306
INFO :                  round 23: 16.085633631105775
INFO :                  round 24: 16.085415482521057
INFO :                  round 25: 16.08057975769043
INFO :                  round 26: 16.07998494748716
INFO :                  round 27: 16.071973204612732
INFO :                  round 28: 16.068379592895507
INFO :                  round 29: 16.061994068084225
INFO :                  round 30: 16.057707885216022
INFO :                  round 31: 16.04799817908894
INFO :                  round 32: 16.042170188643716
INFO :                  round 33: 16.037260911681436
INFO :                  round 34: 16.029848566422096
INFO :                  round 35: 16.021213836669922
INFO :                  round 36: 16.010917360966022
INFO :                  round 37: 15.99938119541515
INFO :                  round 38: 15.983445843060812
INFO :                  round 39: 15.977712306109341
INFO :                  round 40: 15.96019535798293
INFO :                  round 41: 15.945386060078938
INFO :                  round 42: 15.92681804725102
INFO :                  round 43: 15.907257863453456
INFO :                  round 44: 15.880901822677025
INFO :                  round 45: 15.85268830259641
INFO :                  round 46: 15.837819942406245
INFO :                  round 47: 15.805375488599141
INFO :                  round 48: 15.770546674728394
INFO :                  round 49: 15.739253029227257
INFO :                  round 50: 15.671293914318085
INFO :          History (metrics, distributed, evaluate):
INFO :          {'accuracy': [(1, 0.10057692307692308),
INFO :                        (2, 0.09807692307692308),
INFO :                        (3, 0.09895833333333333),
INFO :                        (4, 0.09854166666666667),
INFO :                        (5, 0.09802631578947368),
INFO :                        (6, 0.101),
INFO :                        (7, 0.09807692307692308),
INFO :                        (8, 0.1040625),
INFO :                        (9, 0.10125),
INFO :                        (10, 0.09694444444444444),
INFO :                        (11, 0.0992),
INFO :                        (12, 0.10261904761904762),
INFO :                        (13, 0.10208333333333333),
INFO :                        (14, 0.10351851851851852),
INFO :                        (15, 0.111),
INFO :                        (16, 0.11521739130434783),
INFO :                        (17, 0.11428571428571428),
INFO :                        (18, 0.12981481481481483),
INFO :                        (19, 0.131),
INFO :                        (20, 0.13318181818181818),
INFO :                        (21, 0.12942307692307692),
INFO :                        (22, 0.1305),
INFO :                        (23, 0.13148148148148148),
INFO :                        (24, 0.13083333333333333),
INFO :                        (25, 0.13473684210526315),
INFO :                        (26, 0.13185185185185186),
INFO :                        (27, 0.14225),
INFO :                        (28, 0.14066666666666666),
INFO :                        (29, 0.14016129032258065),
INFO :                        (30, 0.1367241379310345),
INFO :                        (31, 0.1475),
INFO :                        (32, 0.14363636363636365),
INFO :                        (33, 0.1475),
INFO :                        (34, 0.1523076923076923),
INFO :                        (35, 0.1552),
INFO :                        (36, 0.15173076923076922),
INFO :                        (37, 0.15954545454545455),
INFO :                        (38, 0.16055555555555556),
INFO :                        (39, 0.15886363636363637),
INFO :                        (40, 0.1648076923076923),
INFO :                        (41, 0.163),
INFO :                        (42, 0.16642857142857143),
INFO :                        (43, 0.17595238095238094),
INFO :                        (44, 0.1701923076923077),
INFO :                        (45, 0.17291666666666666),
INFO :                        (46, 0.17107142857142857),
INFO :                        (47, 0.178),
INFO :                        (48, 0.18265625),
INFO :                        (49, 0.183125),
INFO :                        (50, 0.18803571428571428)],
INFO :           'count': [(1, 26),
INFO :                     (2, 26),
INFO :                     (3, 24),
INFO :                     (4, 24),
INFO :                     (5, 38),
INFO :                     (6, 30),
INFO :                     (7, 26),
INFO :                     (8, 16),
INFO :                     (9, 24),
INFO :                     (10, 18),
INFO :                     (11, 25),
INFO :                     (12, 21),
INFO :                     (13, 24),
INFO :                     (14, 27),
INFO :                     (15, 35),
INFO :                     (16, 23),
INFO :                     (17, 28),
INFO :                     (18, 27),
INFO :                     (19, 20),
INFO :                     (20, 22),
INFO :                     (21, 26),
INFO :                     (22, 20),
INFO :                     (23, 27),
INFO :                     (24, 24),
INFO :                     (25, 19),
INFO :                     (26, 27),
INFO :                     (27, 20),
INFO :                     (28, 30),
INFO :                     (29, 31),
INFO :                     (30, 29),
INFO :                     (31, 22),
INFO :                     (32, 22),
INFO :                     (33, 22),
INFO :                     (34, 26),
INFO :                     (35, 25),
INFO :                     (36, 26),
INFO :                     (37, 22),
INFO :                     (38, 18),
INFO :                     (39, 22),
INFO :                     (40, 26),
INFO :                     (41, 30),
INFO :                     (42, 28),
INFO :                     (43, 21),
INFO :                     (44, 26),
INFO :                     (45, 24),
INFO :                     (46, 28),
INFO :                     (47, 30),
INFO :                     (48, 32),
INFO :                     (49, 32),
INFO :                     (50, 28)]}
```
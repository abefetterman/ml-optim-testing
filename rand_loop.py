from densenet import ex
import random

try:
    while True:
        etam=random.uniform(0,1)
        etad=random.uniform(0,1)
        weight_decay=10 ** (0.8*etad + random.uniform(0,0.5) - 4.0)
        ex.run(config_updates={
            "etam":etam,
            "etad":etad,
            "weight_decay":weight_decay
        })
except KeyboardInterrupt:
    print('Interrupted!')
